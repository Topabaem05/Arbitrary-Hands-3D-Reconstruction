import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimePerformance } from '../src/performance.js';

test('reports display, fresh inference, and timing breakdown independently', () => {
  const metrics = new RuntimePerformance({ windowMs: 1000 });
  assert.equal(metrics.sample(0), null);
  for (let frame = 0; frame < 60; frame += 1) metrics.recordRender(2);
  for (let frame = 0; frame < 30; frame += 1) {
    metrics.recordCapture(1);
    metrics.recordInference({
      totalMilliseconds: 20,
      detectorMilliseconds: frame % 12 === 0 ? 5 : 0,
      landmarkMilliseconds: 15,
    }, 21);
  }
  const report = metrics.sample(1000);
  assert.equal(report.displayFps, 60);
  assert.equal(report.freshInferenceFps, 30);
  assert.equal(report.captureMs, 1);
  assert.equal(report.schedulerMs, 21);
  assert.equal(report.inferenceMs, 20);
  assert.equal(report.landmarkMs, 15);
  assert.equal(report.renderMs, 2);
});

test('does not emit a partial performance window', () => {
  const metrics = new RuntimePerformance({ windowMs: 2000 });
  metrics.sample(100);
  metrics.recordRender(1);
  assert.equal(metrics.sample(1900), null);
});
