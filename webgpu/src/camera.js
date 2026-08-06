export class CameraController {
  constructor(video) {
    this.video = video;
    this.stream = null;
  }

  async start({ facingMode = 'user', width = 1280, height = 720 } = {}) {
    await this.stop();
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode,
        width: { ideal: width },
        height: { ideal: height },
        frameRate: { ideal: 60, min: 24 },
      },
    });
    this.video.srcObject = this.stream;
    this.video.muted = true;
    this.video.playsInline = true;
    await this.video.play();
    if (!(this.video.videoWidth > 0)) {
      await new Promise((resolve) => this.video.addEventListener('loadedmetadata', resolve, { once: true }));
    }
    return { width: this.video.videoWidth, height: this.video.videoHeight };
  }

  async stop() {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    this.video.pause();
    this.video.srcObject = null;
  }

  get active() {
    return Boolean(this.stream);
  }
}

export async function captureVideoFrame(video) {
  if (typeof VideoFrame === 'function') {
    let frame = null;
    try {
      frame = new VideoFrame(video, { timestamp: Math.round(performance.now() * 1000) });
      return await createImageBitmap(frame);
    } catch {
      // Fall through to the broadly supported ImageBitmap path.
    } finally {
      frame?.close?.();
    }
  }
  return createImageBitmap(video);
}

export function scheduleVideoFrames(video, callback) {
  let cancelled = false;
  let requestId = 0;
  const onFrame = (now, metadata) => {
    if (cancelled) return;
    callback(now, metadata);
    if (typeof video.requestVideoFrameCallback === 'function') {
      requestId = video.requestVideoFrameCallback(onFrame);
    } else {
      requestId = requestAnimationFrame((time) => onFrame(time, { mediaTime: video.currentTime }));
    }
  };
  if (typeof video.requestVideoFrameCallback === 'function') {
    requestId = video.requestVideoFrameCallback(onFrame);
  } else {
    requestId = requestAnimationFrame((time) => onFrame(time, { mediaTime: video.currentTime }));
  }
  return () => {
    cancelled = true;
    if (typeof video.cancelVideoFrameCallback === 'function') video.cancelVideoFrameCallback(requestId);
    else cancelAnimationFrame(requestId);
  };
}
