function mean(total, count) {
  return count > 0 ? total / count : 0;
}

export class RuntimePerformance {
  constructor({ windowMs = 5000 } = {}) {
    if (!(windowMs >= 500)) throw new RangeError('windowMs must be at least 500.');
    this.windowMs = windowMs;
    this.windowStarted = null;
    this.renderFrames = 0;
    this.inferenceFrames = 0;
    this.captureTotal = 0;
    this.captureCount = 0;
    this.schedulerTotal = 0;
    this.inferenceTotal = 0;
    this.detectorTotal = 0;
    this.landmarkTotal = 0;
    this.inferenceCount = 0;
    this.renderTotal = 0;
  }

  recordCapture(milliseconds) {
    if (Number.isFinite(milliseconds) && milliseconds >= 0) {
      this.captureTotal += milliseconds;
      this.captureCount += 1;
    }
  }

  recordInference(timing = {}, schedulerMilliseconds = 0) {
    this.inferenceFrames += 1;
    this.inferenceCount += 1;
    this.schedulerTotal += Number.isFinite(schedulerMilliseconds) ? schedulerMilliseconds : 0;
    this.inferenceTotal += Number.isFinite(timing.totalMilliseconds) ? timing.totalMilliseconds : 0;
    this.detectorTotal += Number.isFinite(timing.detectorMilliseconds) ? timing.detectorMilliseconds : 0;
    this.landmarkTotal += Number.isFinite(timing.landmarkMilliseconds) ? timing.landmarkMilliseconds : 0;
  }

  recordRender(milliseconds) {
    this.renderFrames += 1;
    if (Number.isFinite(milliseconds) && milliseconds >= 0) this.renderTotal += milliseconds;
  }

  sample(now = performance.now()) {
    if (this.windowStarted == null) {
      this.windowStarted = now;
      return null;
    }
    const elapsed = now - this.windowStarted;
    if (elapsed < this.windowMs) return null;
    const seconds = elapsed / 1000;
    const report = {
      windowSeconds: seconds,
      displayFps: this.renderFrames / seconds,
      freshInferenceFps: this.inferenceFrames / seconds,
      captureMs: mean(this.captureTotal, this.captureCount),
      schedulerMs: mean(this.schedulerTotal, this.inferenceCount),
      inferenceMs: mean(this.inferenceTotal, this.inferenceCount),
      detectorMs: mean(this.detectorTotal, this.inferenceCount),
      landmarkMs: mean(this.landmarkTotal, this.inferenceCount),
      renderMs: mean(this.renderTotal, this.renderFrames),
    };
    this.windowStarted = now;
    this.renderFrames = 0;
    this.inferenceFrames = 0;
    this.captureTotal = 0;
    this.captureCount = 0;
    this.schedulerTotal = 0;
    this.inferenceTotal = 0;
    this.detectorTotal = 0;
    this.landmarkTotal = 0;
    this.inferenceCount = 0;
    this.renderTotal = 0;
    return report;
  }
}
