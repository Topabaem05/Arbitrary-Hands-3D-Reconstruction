import os

# Allow individual MPS operators that are not implemented yet to run on CPU.
# This must be set before importing torch.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import sys
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
    """Route legacy Tensor.cuda() calls to MPS/CPU when CUDA is unavailable.

    The upstream inference stack contains several direct Tensor.cuda() calls
    outside this entry point. Replacing those calls in every research module
    would produce a wider, harder-to-review patch, so the demo entry point
    installs this narrow compatibility redirect only on non-CUDA systems.
    """
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
from acr.visualization import Visualizer
if args().model_precision=='fp16':
    from torch.cuda.amp import autocast

########################
# models and dataloader
########################
from acr.model import ACR as ACR_v1
from acr.mano_wrapper import MANOWrapper


class ACR(nn.Module):
    def __init__(self, args_set=None):
        super(ACR, self).__init__()
        self.demo_cfg = {'mode':'parsing', 'calc_loss': False}
        self.project_dir = config.project_dir
        self.device = INFERENCE_DEVICE
        self._initialize_(vars(args() if args_set is None else args_set))

        logging.info('Using inference device: %s', self.device)
        logging.info('Loading {} renderer as visualizer, rendering size: {}'.format(self.renderer, self.render_size))
        self.visualizer = Visualizer(resolution=(self.render_size,self.render_size), renderer_type=self.renderer)

        print('building model')
        self._build_model_()
        print('Initialization finished!')

    def _initialize_(self, config_dict):
        # configs
        hparams_dict = {}
        for i, j in config_dict.items():
            setattr(self,i,j)
            hparams_dict[i] = j

        logging.basicConfig(level=logging.INFO)

        # CUDA AMP is not used on MPS/CPU in this compatibility path. Keeping
        # inference in FP32 avoids the imported model's CUDA-only autocast.
        if self.device.type != 'cuda' and self.model_precision == 'fp16':
            logging.warning(
                'FP16 is CUDA-only in this repository; using FP32 on %s.',
                self.device,
            )
            self.model_precision = 'fp32'
            args().model_precision = 'fp32'
            hparams_dict['model_precision'] = 'fp32'

        logging.info(hparams_dict)
        logging.info('-'*66)

        # optimizations parameters
        if self.temporal_optimization:
            self.filter_dict = {}
            self.filter_dict[0] = create_OneEuroFilter(args().smooth_coeff)
            self.filter_dict[1] = create_OneEuroFilter(args().smooth_coeff)

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
        self.model = model.eval()
        self.mano_regression = MANOWrapper().to(self.device).eval()

    @torch.no_grad()
    def process_results(self, outputs):

        # temporal optimization
        if self.temporal_optimization:
            out_hand = [] # [0],[1],[0,1]
            for idx, i in enumerate(outputs['detection_flag_cache']):
                if i:
                    out_hand.append(idx) # idx is also hand type, 0 for left, 1 for right
                else:
                    out_hand.append(-1)

            assert len(outputs['params_dict']['poses']) == 2
            for sid, tid in enumerate(out_hand):
                if tid == -1:
                    continue
                outputs['params_dict']['poses'][sid], outputs['params_dict']['betas'][sid] = \
                    smooth_results(self.filter_dict[tid], \
                    outputs['params_dict']['poses'][sid], outputs['params_dict']['betas'][sid])

        outputs = self.mano_regression(outputs, outputs['meta_data'])
        reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
        new_results = reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)

        return outputs, new_results

    @torch.no_grad()
    def forward(self, bgr_frame, path):
        with torch.no_grad():
            outputs = self.single_image_forward(bgr_frame, path)

        if outputs is not None and outputs['detection_flag']:
            outputs, results = self.process_results(outputs)

            # visualization: render to raw image
            show_items_list = ['mesh'] # ['org_img', 'mesh', 'pj2d', 'centermap']
            results_dict, img_names = self.visualizer.visulize_result_live(outputs, bgr_frame, outputs['meta_data'], \
                show_items=show_items_list, vis_cfg={'settings':['put_org']}, save2html=False)

            img_name, mesh_rendering_orgimg = img_names[0], results_dict['mesh_rendering_orgimgs']['figs'][0]

            if self.save_visualization_on_img and args().demo_mode!='webcam':
                save_name = os.path.join(self.output_dir + os.path.basename(img_name))
                cv2.imwrite(save_name, mesh_rendering_orgimg[:,:,::-1])
            else:
                cv2.imshow('render_output', mesh_rendering_orgimg[:,:,::-1])
                cv2.waitKey(1)
        else:
            print('no hand detected!')
            results = {}
            results[path] = {}
            if self.save_visualization_on_img and args().demo_mode!='webcam':
                save_name = os.path.join(self.output_dir + os.path.basename(path))
                cv2.imwrite(save_name, bgr_frame)
            else:
                cv2.imshow('render_output', bgr_frame)
                cv2.waitKey(1)

        return results

    @torch.no_grad()
    def single_image_forward(self, bgr_frame, path):
        meta_data = img_preprocess(bgr_frame, path, input_size=args().input_size, single_img_input=True)

        ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.device.type != 'cuda':
            meta_data = move_to_device(meta_data, self.device)

        if self.model_precision=='fp16':
            with autocast():
                outputs = self.model(meta_data, **self.demo_cfg)
        else:
            outputs = self.model(meta_data, **self.demo_cfg)

        outputs['detection_flag'], outputs['reorganize_idx'] = justify_detection_state(outputs['detection_flag'], outputs['reorganize_idx'])
        meta_data.update({'imgpath':imgpath_org, 'data_set':ds_org})
        outputs['meta_data']['imgpath'] = [path]

        return outputs


def main():
    ################## Model Initialization ####################
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        print('Loading the configurations from {}'.format(args_set.configs_yml))
        acr = ACR(args_set=args_set)

    ################## RUN on image forlder ####################
    results_dict = {}
    if args().demo_mode == 'image':

        acr.output_dir = './demos_outputs/single_images_output/'
        print('output dir:', acr.output_dir)
        os.makedirs(acr.output_dir, exist_ok=True)

        image = cv2.imread(imgpath)
        outputs = acr(image, imgpath)
        results_dict.update(outputs)

        if args().save_dict_results:
            save_results(imgpath, acr.output_dir, results_dict)

    ###################### RUN on video ########################
    elif args().demo_mode == 'video' or args().demo_mode == 'folder':
        if not os.path.isdir(args().inputs):
            image_folder = split_frame(args().inputs)
        else:
            image_folder = args().inputs[:-1] if args().inputs.endswith('/') else args().inputs

        print('running on: ', image_folder)
        acr.output_dir = './demos_outputs/' + os.path.basename(image_folder) + f'_results_{args().centermap_conf_thresh}' + '/' + args().model_path.split('/')[-1] + '/'
        print('output dir:', acr.output_dir)
        os.makedirs(acr.output_dir, exist_ok=True)

        file_list = collect_image_list(image_folder=image_folder)
        try:
            file_list = sorted(file_list, key=lambda x:int(os.path.basename(x).split('.')[0])) # please ensure the image name is something like '000000.jpg'
        except:
            print('warning: image filename is not in order.')

        bar = tqdm(file_list)
        for imgpath in bar:
            image = cv2.imread(imgpath)
            outputs = acr(image, imgpath)
            results_dict.update(outputs)

        if args().save_visualization_on_img:
            save_video(acr.output_dir, os.path.basename(image_folder) + '_output_' + os.path.basename(args().model_path.replace('.pkl','')))

        if args().save_dict_results:
            save_results(image_folder, acr.output_dir, results_dict)

    ###################### RUN on webcame ########################
    elif args().demo_mode == 'webcam':
        cap = WebcamVideoStream(args().cam_id)
        cap.start()
        while True:
            frame = cap.read()
            outputs = acr(frame, '0')
        cap.stop()

if __name__ == '__main__':
    main()
