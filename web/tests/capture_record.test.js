'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { validateRecord, ProvenanceError } = require('../capture_check.js');

const good = () => ({
  C1: true, C2: true, C3: true, C4: true, ts: 't', userAgent: 'u', host: 'localhost',
  started_at: 1000, ended_at: 3000, app_commit: 'abc', frame_count: 100,
});

test('fixture record passes validateRecord', () => {
  assert.doesNotThrow(() => validateRecord(good()));
});

test('negative: missing key throws ProvenanceError', () => {
  for (const k of ['host', 'started_at', 'ended_at', 'app_commit', 'frame_count']) {
    const r = good();
    delete r[k];
    assert.throws(() => validateRecord(r), ProvenanceError, k);
  }
});

test('negative: ended_at <= started_at throws', () => {
  assert.throws(() => validateRecord({ ...good(), ended_at: 1000 }), ProvenanceError);
  assert.throws(() => validateRecord({ ...good(), ended_at: 999 }), ProvenanceError);
});

test('negative: frame_count above (ended-started)/20+1 throws', () => {
  assert.doesNotThrow(() => validateRecord({ ...good(), frame_count: 101 }));
  assert.throws(() => validateRecord({ ...good(), frame_count: 102 }), ProvenanceError);
});

const CC = require('../capture_check.js');
const W = require('../worklet.js');

const fr = (seq, n = 320) => W.packFrame(seq, new Float32Array(n));

test('capture_check checkC1 true when echoCancellation setting true', async () => {
  const gum = async () => ({ getAudioTracks: () => [{ getSettings: () => ({ echoCancellation: true }) }] });
  assert.strictEqual(await CC.checkC1(gum), true);
});

test('capture_check checkC1 false on false setting, no track, or rejection', async () => {
  const mkGum = (tracks) => async () => ({ getAudioTracks: () => tracks });
  assert.strictEqual(await CC.checkC1(mkGum([{ getSettings: () => ({ echoCancellation: false }) }])), false);
  assert.strictEqual(await CC.checkC1(mkGum([])), false);
  assert.strictEqual(await CC.checkC1(async () => { throw new Error('denied'); }), false);
});

test('capture_check checkC2 true with >=1 frame, false for empty/non-array/throw', async () => {
  assert.strictEqual(await CC.checkC2(async () => [fr(0)]), true);
  assert.strictEqual(await CC.checkC2(async () => []), false);
  assert.strictEqual(await CC.checkC2(async () => null), false);
  assert.strictEqual(await CC.checkC2(async () => { throw new Error('x'); }), false);
});

test('capture_check checkC3 requires contiguous seq from 0 and >=2 frames', () => {
  assert.strictEqual(CC.checkC3([fr(0), fr(1), fr(2)]), true);
  assert.strictEqual(CC.checkC3([fr(0)]), false);
  assert.strictEqual(CC.checkC3([fr(0), fr(2)]), false);
  assert.strictEqual(CC.checkC3([fr(1), fr(2)]), false);
  assert.strictEqual(CC.checkC3(null), false);
});

test('capture_check checkC4 accepts 320..640 integers only', () => {
  for (const n of [320, 480, 640]) assert.strictEqual(CC.checkC4(n), true, String(n));
  for (const n of [319, 641, 320.5, NaN, '320', undefined]) assert.strictEqual(CC.checkC4(n), false, String(n));
});

test('capture_check serializeResult emits RESULT_KEYS in order as JSON', () => {
  const g = good();
  const json = CC.serializeResult({ C1: true, C2: false, C3: true, C4: false }, g);
  const o = JSON.parse(json);
  assert.deepStrictEqual(Object.keys(o), CC.RESULT_KEYS);
  assert.deepStrictEqual([o.C1, o.C2, o.C3, o.C4], [true, false, true, false]);
  assert.strictEqual(o.host, 'localhost');
  assert.strictEqual(o.frame_count, 100);
  assert.doesNotThrow(() => CC.serializeResult({ C1: true, C2: true, C3: true, C4: true }));
});

test('capture_check serializeResult output round-trips through validateRecord', () => {
  const json = CC.serializeResult({ C1: true, C2: true, C3: true, C4: true }, good());
  assert.strictEqual(CC.validateRecord(JSON.parse(json)), true);
});

test('capture_check validateRecord negative: non-object, empty/null value, non-number fields', () => {
  assert.throws(() => CC.validateRecord(null), CC.ProvenanceError);
  assert.throws(() => CC.validateRecord('x'), CC.ProvenanceError);
  assert.throws(() => CC.validateRecord({ ...good(), host: '' }), CC.ProvenanceError);
  assert.throws(() => CC.validateRecord({ ...good(), app_commit: null }), CC.ProvenanceError);
  assert.throws(() => CC.validateRecord({ ...good(), started_at: '1000' }), CC.ProvenanceError);
  assert.throws(() => CC.validateRecord({ ...good(), frame_count: NaN }), CC.ProvenanceError);
});
