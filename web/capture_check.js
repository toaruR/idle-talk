'use strict';
// C1..C4 judgement logic for M0 browser capture check (design 6.2, 3.1).
// Browser APIs are injected so everything is testable with stubs.

const W = (typeof module !== 'undefined' && module.exports)
  ? require('./worklet.js') : globalThis.IdleTalkWorklet;

const FRAME_MS = 20;
const REQUIRED_META_KEYS = ['host', 'started_at', 'ended_at', 'app_commit', 'frame_count'];
const RESULT_KEYS = ['C1', 'C2', 'C3', 'C4', 'ts', 'userAgent', 'host',
  'started_at', 'ended_at', 'app_commit', 'frame_count'];

class ProvenanceError extends Error {
  constructor(message) { super(message); this.name = 'ProvenanceError'; }
}

// C1: getUserMedia({audio:{echoCancellation:true}}) yields a track with echoCancellation==true.
async function checkC1(getUserMedia) {
  try {
    const stream = await getUserMedia({ audio: { echoCancellation: true } });
    const track = stream.getAudioTracks()[0];
    return !!track && track.getSettings().echoCancellation === true;
  } catch (_e) {
    return false;
  }
}

// C2: the worklet delivers at least one frame. deliver: async () => frames[].
async function checkC2(deliver) {
  try {
    const frames = await deliver();
    return Array.isArray(frames) && frames.length >= 1;
  } catch (_e) {
    return false;
  }
}

// C3: sequence numbers are 0,1,2,... with no gaps and at least 2 frames.
function checkC3(frames) {
  if (!Array.isArray(frames) || frames.length < 2) return false;
  return frames.every((f, i) => W.readSeq(f) === i);
}

// C4: frame length (samples) is within 320..640 (20-40ms @16kHz).
function checkC4(samples) {
  return Number.isInteger(samples) && samples >= W.MIN_FRAME_SAMPLES && samples <= W.MAX_FRAME_SAMPLES;
}

function serializeResult(checks, meta) {
  const m = meta || {};
  const out = {
    C1: checks.C1, C2: checks.C2, C3: checks.C3, C4: checks.C4,
    ts: m.ts, userAgent: m.userAgent, host: m.host,
    started_at: m.started_at, ended_at: m.ended_at,
    app_commit: m.app_commit, frame_count: m.frame_count,
  };
  return JSON.stringify(out, null, 2);
}

function validateRecord(record) {
  if (record === null || typeof record !== 'object') throw new ProvenanceError('record must be an object');
  for (const k of REQUIRED_META_KEYS) {
    if (!(k in record) || record[k] === undefined || record[k] === null || record[k] === '') {
      throw new ProvenanceError(`missing key: ${k}`);
    }
  }
  const { started_at: s, ended_at: e, frame_count: n } = record;
  if (![s, e, n].every((v) => typeof v === 'number' && Number.isFinite(v))) {
    throw new ProvenanceError('started_at/ended_at/frame_count must be numbers');
  }
  if (!(e > s)) throw new ProvenanceError('ended_at must be greater than started_at');
  if (n > (e - s) / FRAME_MS + 1) throw new ProvenanceError('frame_count exceeds (ended_at-started_at)/20ms+1');
  return true;
}

// Browser runner (real getUserMedia / AudioContext injected by the page).
async function runCapture({ getUserMedia, AudioContextCtor, workletUrl, durationMs = 2000,
  frameSamples = 320, now = Date.now, appCommit = 'unknown', host = '', userAgent = '' }) {
  const started_at = now();
  const result = { C1: false, C2: false, C3: false, C4: false };
  const frames = [];
  let stream = null;
  let ctx = null;
  try {
    result.C1 = await checkC1(getUserMedia);
    stream = await getUserMedia({ audio: { echoCancellation: true } });
    ctx = new AudioContextCtor();
    await ctx.audioWorklet.addModule(workletUrl);
    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'capture-processor', {
      processorOptions: { inRate: ctx.sampleRate, frameSamples },
    });
    node.port.onmessage = (ev) => frames.push(new Uint8Array(ev.data));
    src.connect(node);
    result.C2 = await checkC2(async () => {
      await new Promise((r) => setTimeout(r, durationMs));
      return frames;
    });
    result.C3 = checkC3(frames);
    result.C4 = frames.length > 0 && checkC4(W.frameSampleCount(frames[0]));
  } finally {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (ctx) await ctx.close();
  }
  const ended_at = Math.max(now(), started_at + 1);
  const json = serializeResult(result, {
    ts: new Date(ended_at).toISOString(), userAgent, host,
    started_at, ended_at, app_commit: appCommit, frame_count: frames.length,
  });
  return { result, json };
}

const api = {
  ProvenanceError, checkC1, checkC2, checkC3, checkC4, serializeResult, validateRecord,
  runCapture, RESULT_KEYS, REQUIRED_META_KEYS, FRAME_MS,
};
if (typeof module !== 'undefined' && module.exports) module.exports = api;
else globalThis.IdleTalkCapture = api;
