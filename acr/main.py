import os

# MPS performance/compatibility switches must be set before importing torch.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('PYTORCH_MPS_FAST_MATH', '1')
os.environ.setdefault('PYTORCH_MPS_PREFER_METAL', '1')

import sys
import time
import cv2
from tqdm import tqdm
import logging
import torch
import torch.nn as nn


def select_inference_device():
    """Select CUDA first, then Apple MPS, with CPU as the final fallback."""
    if torch.cuda.is_available():
        return torch.device('cuda')

    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device('mps')

    return torch.device('cpu')


def move_to_device(value, device):
    """Recursively move tensors while preserving non-tensor metadata."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def redirect_legacy_tensor_cuda(device):
    """Route legacy Tensor.cuda() calls to MPS/CPU when CUDA is unavailable."""
    if device.type == 'cuda':
        return
    if getattr(torch.Tensor.cuda, '_acr_device_redirect', False):
        return

    selected_device = device

    def tensor_cuda(tensor, device=None, non_blocking=False,
                    memory_format=torch.preserve_format):
        return tensor.to(
            selected_device,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    tensor_cuda._acr_device_redirect = True
    torch.Tensor.cuda = tensor_cuda
    logging.info('Redirecting legacy Tensor.cuda() calls to %s', selected_device)


def load_model_on_cpu(path, model, prefix='module.', drop_prefix='',
                      fix_loaded=False):
    """Load checkpoint storage on CPU before moving the model to its backend."""
    logging.info('using fine_tune model: %s', path)
    if not os.path.exists(path):
        logging.warning('model %s not exist!', path)
        raise ValueError('model {} not exist'.format(path))

    pretrained_model = torch.load(path, map_location='cpu')
    if isinstance(pretrained_model, dict):
        if 'model_state_dict' in pretrained_model:
            pretrained_model = pretrained_model['model_state_dict']
        if 'state_dict' in pretrained_model:
            pretrained_model = pretrained_model['state_dict']

    copy_state_dict(
        model.state_dict(),
        pretrained_model,
        prefix=prefix,
        drop_prefix=drop_prefix,
        fix_loaded=fix_loaded,
    )
    return model


INFERENCE_DEVICE = select_inference_device()
redirect_legacy_tensor_cuda(INFERENCE_DEVICE)

##################
# config and utils
##################
import acr.config as config
from acr.config import args, parse_args, ConfigContext
from acr.utils import *
from acr.utils import justify_detection_state, reorganize_results, collect_image_list, img_preprocess, WebcamVideoStream, split_frame, save_results

########################
# models and runtime
########################
from acr.model import ACR as ACR_v1
from acr.mano_wrapper import MANOWrapper
from acr.mps_realtime import (
    ExponentialRate,
    LatestFrameWorker,
    apply_realtime_options,
    build_realtime_options,
    draw_keypoint_overlay,
    extract_keypoint_overlay,
    patch_model_for_realtime,
)


class ACR(nn.Module):
    def __init__(self, args_set=None):
        super(ACR, self).__init__()
        self.demo_cfg = {'mode': 'parsing', 'calc_loss': False}
        self.project_dir = config.project_dir
        self.device = INFERENCE_DEVICE
        self.cached_points = None
        self.cached_hand_types = None
        self.inference_rate = ExponentialRate()
        self.display_rate = ExponentialRate()
        self._last_display_timestamp = None
        self._mesh_render_index = 0
        self._initialize_(vars(args() if args_set is None else args_set))

        self.runtime = build_realtime_options(args(), self.device, argv=sys.argv[1:])
        apply_realtime_options(args(), self.runtime)
        self.input_size = self.runtime.input_size
        self.render_size = self.runtime.render_size
        self.live_visualization = self.runtime.live_visualization
        self.inference_stride = self.runtime.inference_stride
        self.render_every = self.runtime.render_every
        self.mps_amp_enabled = self.runtime.mps_amp

        # The imported research model has a CUDA-only fp16 branch. Keep that
        # branch disabled on MPS and place a device-generic MPS autocast around
        # the complete model call instead.
        if self.device.type != 'cuda':
            self.model_precision = 'fp32'
            args().model_precision = 'fp32'

        logging.info(
            'Runtime profile=%s device=%s input=%d visualization=%s '
            'stride=%d MPS_AMP=%s',
            self.runtime.profile,
            self.device,
            self.input_size,
            self.live_visualization,
            self.inference_stride,
            self.mps_amp_enabled,
        )

        self.visualizer = None
        if self.live_visualization == 'mesh' or args().demo_mode != 'webcam':
            from acr.visualization import Visualizer
            logging.info(
                'Loading %s renderer as visualizer, rendering size: %s',
                self.renderer,
                self.render_size,
            )
            self.visualizer = Visualizer(
                resolution=(self.render_size, self.render_size),
                renderer_type=self.renderer,
            )

        print('building model')
        self._build_model_()
        print('Initialization finished!')

    def _initialize_(self, config_dict):
        hparams_dict = {}
        for key, value in config_dict.items():
            setattr(self, key, value)
            hparams_dict[key] = value

        logging.basicConfig(level=logging.INFO)
        logging.info(hparams_dict)
        logging.info('-' * 66)

        if self.temporal_optimization:
            self.filter_dict = {
                0: create_OneEuroFilter(args().smooth_coeff),
                1: create_OneEuroFilter(args().smooth_coeff),
            }
        return hparams_dict

    def _build_model_(self):
        model = ACR_v1().eval()
        model = load_model_on_cpu(
            self.model_path,
            model,
            prefix='module.',
            drop_prefix='',
            fix_loaded=False,
        )
        model = model.to(self.device)
        if self.device.type == 'cuda':
            model = nn.DataParallel(model)

        patch_model_for_realtime(
            model,
            input_size=self.input_size,
            device=self.device,
            compute_camera_translation=self.runtime.compute_camera_translation,
        )
        self.model = model.eval()
        self.mano_regression = MANOWrapper().to(self.device).eval()

    @torch.inference_mode()
    def process_results(self, outputs, build_result_dict=True):
        if self.temporal_optimization:
            out_hand = []
            for index, present in enumerate(outputs['detection_flag_cache']):
                out_hand.append(index if present else -1)

            assert len(outputs['params_dict']['poses']) == 2
            for source_id, target_id in enumerate(out_hand):
                if target_id == -1:
                    continue
                outputs['params_dict']['poses'][source_id], outputs['params_dict']['betas'][source_id] = \
                    smooth_results(
                        self.filter_dict[target_id],
                        outputs['params_dict']['poses'][source_id],
                        outputs['params_dict']['betas'][source_id],
                    )

        outputs = self.mano_regression(outputs, outputs['meta_data'])
        if not build_result_dict:
            return outputs, {}

        reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
        new_results = reorganize_results(
            outputs,
            outputs['meta_data']['imgpath'],
            reorganize_idx,
        )
        return outputs, new_results

    def _runtime_label(self):
        precision = 'FP16' if self.mps_amp_enabled else 'FP32'
        return '{} {} {}'.format(self.device.type.upper(), precision, self.input_size)

    def _record_display(self):
        now = time.perf_counter()
        if self._last_display_timestamp is not None:
            self.display_rate.update_duration(now - self._last_display_timestamp)
        self._last_display_timestamp = now

    def record_inference_duration(self, seconds):
        self.inference_rate.update_duration(seconds)

    def _show_frame(self, frame):
        self._record_display()
        if self.runtime.show_fps:
            frame = draw_keypoint_overlay(
                frame,
                None,
                None,
                display_fps=self.display_rate.value,
                inference_fps=self.inference_rate.value,
                label=self._runtime_label(),
                show_fps=True,
            )
        cv2.imshow('render_output', frame)

    def display_cached(self, frame):
        display = draw_keypoint_overlay(
            frame,
            self.cached_points,
            self.cached_hand_types,
            display_fps=self.display_rate.value,
            inference_fps=self.inference_rate.value,
            label=self._runtime_label(),
            show_fps=self.runtime.show_fps,
        )
        self._record_display()
        cv2.imshow('render_output', display)

    def _cache_keypoints(self, outputs):
        self.cached_points, self.cached_hand_types = extract_keypoint_overlay(outputs)

    def _render_mesh_frame(self, outputs, bgr_frame):
        results_dict, img_names = self.visualizer.visulize_result_live(
            outputs,
            bgr_frame,
            outputs['meta_data'],
            show_items=['mesh'],
            vis_cfg={'settings': ['put_org']},
            save2html=False,
        )
        return img_names[0], results_dict['mesh_rendering_orgimgs']['figs'][0]

    @torch.inference_mode()
    def infer_live_payload(self, bgr_frame):
        """Return a CPU-only keypoint payload for the asynchronous webcam loop."""
        outputs = self.single_image_forward(bgr_frame, '0')
        if outputs is None or not outputs['detection_flag']:
            return None, None
        outputs, _ = self.process_results(outputs, build_result_dict=False)
        return extract_keypoint_overlay(outputs)

    @torch.inference_mode()
    def forward(self, bgr_frame, path):
        outputs = self.single_image_forward(bgr_frame, path)
        live_mode = args().demo_mode == 'webcam'
        build_result_dict = not live_mode or bool(args().save_dict_results)

        if outputs is not None and outputs['detection_flag']:
            outputs, results = self.process_results(
                outputs,
                build_result_dict=build_result_dict,
            )
            self._cache_keypoints(outputs)

            if live_mode and self.live_visualization == 'keypoints':
                self.display_cached(bgr_frame)
            elif live_mode and self.live_visualization == 'none':
                self._show_frame(bgr_frame)
            else:
                should_render_mesh = (
                    not live_mode
                    or self._mesh_render_index % self.render_every == 0
                )
                self._mesh_render_index += 1
                if should_render_mesh:
                    img_name, mesh_rendering_orgimg = self._render_mesh_frame(outputs, bgr_frame)
                    if self.save_visualization_on_img and not live_mode:
                        save_name = os.path.join(self.output_dir + os.path.basename(img_name))
                        cv2.imwrite(save_name, mesh_rendering_orgimg[:, :, ::-1])
                    else:
                        self._record_display()
                        cv2.imshow('render_output', mesh_rendering_orgimg[:, :, ::-1])
                elif live_mode:
                    self.display_cached(bgr_frame)
        else:
            results = {path: {}} if build_result_dict else {}
            self.cached_points = None
            self.cached_hand_types = None
            if self.save_visualization_on_img and not live_mode:
                save_name = os.path.join(self.output_dir + os.path.basename(path))
                cv2.imwrite(save_name, bgr_frame)
            else:
                self._show_frame(bgr_frame)

        return results

    @torch.inference_mode()
    def single_image_forward(self, bgr_frame, path):
        meta_data = img_preprocess(
            bgr_frame,
            path,
            input_size=self.input_size,
            single_img_input=True,
        )

        ds_org, imgpath_org = get_remove_keys(meta_data, keys=['data_set', 'imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.device.type != 'cuda':
            meta_data = move_to_device(meta_data, self.device)

        if self.mps_amp_enabled:
            try:
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = self.model(meta_data, **self.demo_cfg)
            except RuntimeError as error:
                logging.warning(
                    'MPS FP16 inference failed (%s); retrying and remaining in FP32.',
                    error,
                )
                self.mps_amp_enabled = False
                if hasattr(torch, 'mps'):
                    torch.mps.empty_cache()
                outputs = self.model(meta_data, **self.demo_cfg)
        else:
            outputs = self.model(meta_data, **self.demo_cfg)

        outputs['detection_flag'], outputs['reorganize_idx'] = justify_detection_state(
            outputs['detection_flag'],
            outputs['reorganize_idx'],
        )
        meta_data.update({'imgpath': imgpath_org, 'data_set': ds_org})
        outputs['meta_data']['imgpath'] = [path]
        return outputs


def main():
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        print('Loading the configurations from {}'.format(args_set.configs_yml))
        acr = ACR(args_set=args_set)

    results_dict = {}
    if args().demo_mode == 'image':
        acr.output_dir = './demos_outputs/single_images_output/'
        print('output dir:', acr.output_dir)
        os.makedirs(acr.output_dir, exist_ok=True)

        imgpath = args().inputs
        if not imgpath:
            raise ValueError('--inputs is required in image mode')
        image = cv2.imread(imgpath)
        if image is None:
            raise ValueError('failed to read image: {}'.format(imgpath))
        outputs = acr(image, imgpath)
        results_dict.update(outputs)

        if args().save_dict_results:
            save_results(imgpath, acr.output_dir, results_dict)

    elif args().demo_mode == 'video' or args().demo_mode == 'folder':
        if not os.path.isdir(args().inputs):
            image_folder = split_frame(args().inputs)
        else:
            image_folder = args().inputs[:-1] if args().inputs.endswith('/') else args().inputs

        print('running on: ', image_folder)
        acr.output_dir = './demos_outputs/' + os.path.basename(image_folder) + \
            f'_results_{args().centermap_conf_thresh}' + '/' + \
            args().model_path.split('/')[-1] + '/'
        print('output dir:', acr.output_dir)
        os.makedirs(acr.output_dir, exist_ok=True)

        file_list = collect_image_list(image_folder=image_folder)
        try:
            file_list = sorted(
                file_list,
                key=lambda item: int(os.path.basename(item).split('.')[0]),
            )
        except (ValueError, IndexError):
            print('warning: image filename is not in order.')

        for imgpath in tqdm(file_list):
            image = cv2.imread(imgpath)
            outputs = acr(image, imgpath)
            results_dict.update(outputs)

        if args().save_visualization_on_img:
            save_video(
                acr.output_dir,
                os.path.basename(image_folder) + '_output_' +
                os.path.basename(args().model_path.replace('.pkl', '')),
            )

        if args().save_dict_results:
            save_results(image_folder, acr.output_dir, results_dict)

    elif args().demo_mode == 'webcam':
        cap = WebcamVideoStream(args().cam_id)
        cap.start()
        frame_index = 0
        worker = None
        last_generation = 0

        # Mesh rendering owns an OpenGL context and remains on the foreground
        # path. Keypoint/none modes use a latest-frame inference worker so the
        # macOS GUI loop can refresh at the camera rate while stale frames are
        # dropped instead of queued.
        if acr.device.type == 'mps' and acr.live_visualization != 'mesh':
            worker = LatestFrameWorker(acr.infer_live_payload).start()

        try:
            while True:
                frame = cap.read()
                if frame is None:
                    continue

                if worker is not None:
                    if frame_index % acr.inference_stride == 0:
                        worker.submit(frame)
                    result = worker.latest()
                    if result is not None and result.generation != last_generation:
                        last_generation = result.generation
                        if result.error is not None:
                            raise result.error
                        acr.cached_points, acr.cached_hand_types = result.payload
                        acr.record_inference_duration(result.duration)

                    if acr.live_visualization == 'keypoints':
                        acr.display_cached(frame)
                    else:
                        acr._show_frame(frame)
                else:
                    start = time.perf_counter()
                    acr(frame, '0')
                    acr.record_inference_duration(time.perf_counter() - start)

                frame_index += 1
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
        finally:
            if worker is not None:
                worker.stop()
            cap.stop()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
