'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const W = require('../worklet.js');
const C = require('../capture_check.js');

const sine = (n) => Float32Array.from({ length: n }, (_, i) => Math.sin(i / 10) * 0.5);
const frameWith = (seq, samples = 320) => W.packFrame(seq, new Float32Array(samples));

test('frame: 320 samples -> 4+640 bytes with u32 LE seq from 0', () => {
  const fr = W.createFramer({ inRate: 16000, frameSamples: 320 });
  const frames = fr.push(sine(960));
  assert.equal(frames.length, 3);
  frames.forEach((f, i) => {
    assert.equal(f.length, 644);
    assert.equal(W.readSeq(f), i);
    assert.equal(f[0], i & 0xff);
  });
  assert.equal(W.readSeq(fr.push(sine(320))[0]), 3);
});

test('frame: length 20-40ms accepted, 304 and 656 throw', () => {
  W.createFramer({ frameSamples: 320 });
  W.createFramer({ frameSamples: 640 });
  assert.throws(() => W.createFramer({ frameSamples: 304 }), RangeError);
  assert.throws(() => W.createFramer({ frameSamples: 656 }), RangeError);
});

test('frame: 48000Hz 480 samples -> 160 samples', () => {
  const rs = W.createResampler(48000);
  assert.equal(rs.process(sine(480)).length, 160);
  const fr = W.createFramer({ inRate: 48000, frameSamples: 320 });
  assert.equal(fr.push(sine(960 * 2)).length, 2);
});

test('frame: s16le conversion', () => {
  const f = W.packFrame(7, Float32Array.from([1, -1, 0]));
  const dv = new DataView(f.buffer);
  assert.equal(dv.getInt16(4, true), 32767);
  assert.equal(dv.getInt16(6, true), -32768);
  assert.equal(dv.getInt16(8, true), 0);
});

const stubGUM = (ec) => async () => ({
  getAudioTracks: () => [{ getSettings: () => ({ echoCancellation: ec }) }],
});

test('checks: C1 echoCancellation', async () => {
  assert.equal(await C.checkC1(stubGUM(true)), true);
  assert.equal(await C.checkC1(stubGUM(false)), false);
});

test('checks: C2 worklet delivers frames', async () => {
  assert.equal(await C.checkC2(async () => [frameWith(0)]), true);
  assert.equal(await C.checkC2(async () => []), false);
});

test('checks: C3 sequence', () => {
  assert.equal(C.checkC3([0, 1, 2].map((s) => frameWith(s))), true);
  assert.equal(C.checkC3([0, 2].map((s) => frameWith(s))), false);
  assert.equal(C.checkC3([frameWith(0)]), false);
});

test('checks: C4 frame length', () => {
  assert.equal(C.checkC4(320), true);
  assert.equal(C.checkC4(640), true);
  assert.equal(C.checkC4(304), false);
  assert.equal(C.checkC4(656), false);
});

test('checks: serializeResult has 11 keys', () => {
  const j = JSON.parse(C.serializeResult(
    { C1: true, C2: true, C3: true, C4: true },
    { ts: 't', userAgent: 'u', host: 'localhost', started_at: 1, ended_at: 2, app_commit: 'x', frame_count: 1 }));
  assert.deepEqual(Object.keys(j), ['C1', 'C2', 'C3', 'C4', 'ts', 'userAgent', 'host',
    'started_at', 'ended_at', 'app_commit', 'frame_count']);
});

test('frame: seqBase (drama) sets the top bit of seq', () => {
  const fr = W.createFramer({ inRate: 16000, frameSamples: 320, seqBase: W.DRAMA_FLAG });
  const frames = fr.push(sine(640));
  assert.deepEqual(frames.map(W.readSeq), [0x80000000, 0x80000001]);
});
