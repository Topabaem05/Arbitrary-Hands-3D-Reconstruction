export class ExponentialRate {
  constructor(smoothing = 0.15) {
    if (!(smoothing > 0 && smoothing <= 1)) {
      throw new RangeError('smoothing must be in (0, 1].');
    }
    this.smoothing = smoothing;
    this.value = 0;
  }

  updateDuration(seconds) {
    if (!(seconds > 0)) return this.value;
    const sample = 1 / seconds;
    this.value = this.value === 0
      ? sample
      : this.value * (1 - this.smoothing) + sample * this.smoothing;
    return this.value;
  }
}

export class LatestFrameScheduler {
  constructor(worker, { dispose = null } = {}) {
    if (typeof worker !== 'function') throw new TypeError('worker must be a function.');
    this.worker = worker;
    this.dispose = dispose;
    this.pending = undefined;
    this.busy = false;
    this.latest = null;
    this.stopped = false;
    this.idleWaiters = [];
    this.generation = 0;
  }

  submit(value) {
    if (this.stopped) {
      this.dispose?.(value);
      return;
    }
    if (this.pending !== undefined) this.dispose?.(this.pending);
    this.pending = value;
    if (!this.busy) void this.#drain();
  }

  async #drain() {
    this.busy = true;
    try {
      while (!this.stopped && this.pending !== undefined) {
        const value = this.pending;
        this.pending = undefined;
        const started = performance.now();
        try {
          const output = await this.worker(value);
          this.generation += 1;
          this.latest = {
            generation: this.generation,
            value: output,
            duration: (performance.now() - started) / 1000,
            error: null,
          };
        } catch (error) {
          this.generation += 1;
          this.latest = {
            generation: this.generation,
            value: null,
            duration: (performance.now() - started) / 1000,
            error,
          };
        } finally {
          this.dispose?.(value);
        }
      }
    } finally {
      this.busy = false;
      if (this.pending !== undefined && !this.stopped) {
        void this.#drain();
      } else {
        for (const resolve of this.idleWaiters.splice(0)) resolve();
      }
    }
  }

  whenIdle() {
    if (!this.busy && this.pending === undefined) return Promise.resolve();
    return new Promise((resolve) => this.idleWaiters.push(resolve));
  }

  async stop() {
    this.stopped = true;
    if (this.pending !== undefined) {
      this.dispose?.(this.pending);
      this.pending = undefined;
    }
    await this.whenIdle();
  }
}
