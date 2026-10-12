'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { AEC_CONSTRAINTS, probeConstraints, runProbe } = require('../aec_probe.js');

test('constraints request echoCancellation:true', () => {
  assert.deepStrictEqual(probeConstraints(), { audio: { echoCancellation: true } });
  assert.strictEqual(AEC_CONSTRAINTS.audio.echoCancellation, true);
});

function fakeGum(ec, calls) {
  return async (c) => {
    calls.push(c);
    return {
      getAudioTracks: () => [{ getSettings: () => ({ echoCancellation: ec }) }],
      getTracks: () => [{ stop() { calls.push('stop'); } }],
    };
  };
}

test('runProbe passes constraints to getUserMedia and reports setting', async () => {
  const calls = [];
  const r = await runProbe(fakeGum(true, calls));
  assert.deepStrictEqual(calls[0], { audio: { echoCancellation: true } });
  assert.strictEqual(r.echoCancellation, true);
  assert.ok(calls.includes('stop'));
});

test('negative: runProbe reports false when AEC not applied', async () => {
  const r = await runProbe(fakeGum(false, []));
  assert.strictEqual(r.echoCancellation, false);
});
