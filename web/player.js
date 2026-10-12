'use strict';
// Playback queue with epoch discard / duck / resume (design 3.1, 3.2, 12.1).
// The audio backend is injected as `sink` so the logic is testable without a browser:
//   sink.play(chunk, onEnded)  start playing chunk.pcm (Int16Array); call onEnded() when done
//   sink.stop()                abort current playback immediately
//   sink.setGain(g)            output gain
//   sink.positionMs()          position inside the currently playing chunk (ms)

const HEADER_SIZE = 16;
const MAX_CHUNK_PCM = 1048576;
const DUCK_GAIN = 0.2;
const MAX_LOOKAHEAD = 1; // chunks held behind the one being played
const MAX_POSITION_MS = 600000;
const DEFAULT_PCM_RATE = 24000;

function parseDownFrame(buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (u8.byteLength < HEADER_SIZE) return null;
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const epoch = dv.getUint32(0, true);
  const utterance_id = dv.getUint32(4, true);
  const chunk_id = dv.getUint32(8, true);
  const length = dv.getUint32(12, true);
  if (length > MAX_CHUNK_PCM || u8.byteLength - HEADER_SIZE !== length || (length & 1)) return null;
  const pcm = new Int16Array(length / 2);
  for (let i = 0; i < pcm.length; i++) pcm[i] = dv.getInt16(HEADER_SIZE + i * 2, true);
  return { epoch, utterance_id, chunk_id, length, pcm };
}

function createPlayer({ sink, send = () => {}, duckGain = DUCK_GAIN, pcmRate = DEFAULT_PCM_RATE } = {}) {
  let minEpoch = 0;
  let queue = [];
  let current = null;
  let gain = 1.0;
  let ducked = false;
  let lastPos = 0;
  let uttBaseMs = 0; // finished-chunk duration accumulated inside current utterance
  let uttId = null;

  const clamp = (ms) => Math.max(0, Math.min(MAX_POSITION_MS, Math.round(ms)));

  function report(chunk, state, pos) {
    send({
      t: 'playback', epoch: chunk.epoch, utterance_id: chunk.utterance_id,
      chunk_id: chunk.chunk_id, position_ms: clamp(pos), state,
    });
  }

  function positionMs() {
    if (!current) return lastPos;
    const p = clamp(uttBaseMs + sink.positionMs());
    if (p > lastPos) lastPos = p;
    return lastPos;
  }

  function startNext() {
    if (current || queue.length === 0) return;
    const chunk = queue.shift();
    if (chunk.utterance_id !== uttId) { uttId = chunk.utterance_id; uttBaseMs = 0; lastPos = 0; }
    current = chunk;
    sink.setGain(gain);
    report(chunk, 'started', positionMs());
    sink.play(chunk, () => onEnded(chunk));
  }

  function onEnded(chunk) {
    if (current !== chunk) return; // stale callback after cancel
    const dur = (chunk.pcm.length / pcmRate) * 1000;
    uttBaseMs += dur;
    current = null;
    lastPos = Math.max(lastPos, clamp(uttBaseMs));
    report(chunk, 'finished', lastPos);
    startNext();
  }

  return {
    // Returns true if accepted.
    pushChunk(chunk) {
      if (!chunk || chunk.epoch < minEpoch) return false;
      if (queue.length >= MAX_LOOKAHEAD && current) {
        report(chunk, 'dropped', positionMs());
        return false;
      }
      queue.push(chunk);
      startNext();
      return true;
    },
    pushBinary(buf) {
      const c = parseDownFrame(buf);
      return c ? this.pushChunk(c) : false;
    },
    cancel(newEpoch) {
      if (newEpoch > minEpoch) minEpoch = newEpoch;
      const keep = [];
      for (const c of queue) {
        if (c.epoch < minEpoch) report(c, 'dropped', positionMs());
        else keep.push(c);
      }
      queue = keep;
      if (current && current.epoch < minEpoch) {
        const c = current;
        const pos = positionMs();
        current = null;
        sink.stop();
        report(c, 'dropped', pos);
      }
      startNext();
    },
    duck(epoch, g = duckGain) {
      if (epoch < minEpoch) return;
      ducked = true;
      gain = g;
      sink.setGain(gain);
    },
    resume(epoch) {
      if (epoch < minEpoch) return;
      ducked = false;
      gain = 1.0;
      sink.setGain(gain);
    },
    handleMessage(msg) {
      if (!msg || typeof msg !== 'object') return;
      if (msg.t === 'cancel') this.cancel(msg.new_epoch);
      else if (msg.t === 'duck') this.duck(msg.epoch, msg.gain);
      else if (msg.t === 'resume') this.resume(msg.epoch);
    },
    positionMs,
    get gain() { return gain; },
    get ducked() { return ducked; },
    get minEpoch() { return minEpoch; },
    get queueLength() { return queue.length; },
    get current() { return current; },
  };
}

// Browser sink built on Web Audio.
function createWebAudioSink(ctx, pcmRate = DEFAULT_PCM_RATE) {
  const gainNode = ctx.createGain();
  gainNode.connect(ctx.destination);
  let src = null;
  let startedAt = 0;
  return {
    play(chunk, onEnded) {
      const buf = ctx.createBuffer(1, chunk.pcm.length, pcmRate);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < chunk.pcm.length; i++) ch[i] = chunk.pcm[i] / 32768;
      const s = ctx.createBufferSource();
      s.buffer = buf;
      s.connect(gainNode);
      s.onended = () => { if (src === s) { src = null; onEnded(); } };
      src = s;
      startedAt = ctx.currentTime;
      s.start();
    },
    stop() {
      const s = src;
      src = null;
      if (s) { s.onended = null; try { s.stop(); } catch (_e) { /* already stopped */ } }
    },
    setGain(g) { gainNode.gain.value = g; },
    positionMs() { return src ? (ctx.currentTime - startedAt) * 1000 : 0; },
  };
}

const api = {
  HEADER_SIZE, MAX_CHUNK_PCM, DUCK_GAIN, MAX_LOOKAHEAD,
  parseDownFrame, createPlayer, createWebAudioSink,
};
if (typeof module !== 'undefined' && module.exports) module.exports = api;
else if (typeof globalThis !== 'undefined') globalThis.IdleTalkPlayer = api;
