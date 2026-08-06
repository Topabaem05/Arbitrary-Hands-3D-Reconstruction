import test from 'node:test';
import assert from 'node:assert/strict';
import { ExponentialRate, LatestFrameScheduler } from '../src/scheduler.js';

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));


test('latest frame scheduler replaces stale queued work', async () => {
  const seen = [];
  const scheduler = new LatestFrameScheduler(async (value) => {
    seen.push(value);
    await sleep(15);
    return value * 10;
  });

  scheduler.submit(1);
  scheduler.submit(2);
  scheduler.submit(3);
  await scheduler.whenIdle();

  assert.deepEqual(seen, [1, 3]);
  assert.equal(scheduler.latest.value, 30);
});


test('exponential rate reports reciprocal duration', () => {
  const rate = new ExponentialRate(1);
  assert.equal(rate.updateDuration(0.02), 50);
  assert.equal(rate.value, 50);
});
