'use strict';
// ブラウザでは classic script として共通のグローバルスコープに読まれるため、IIFE で閉じる。
(function () {
  // 16kHz mono s16le framing + AudioWorklet processor (design 3.1).
  // Frame = u32 LE sequence (4 byte) + frameSamples * 2 byte PCM.

  const TARGET_RATE = 16000;
  const MIN_FRAME_SAMPLES = 320; // 20ms
  const MAX_FRAME_SAMPLES = 640; // 40ms

  function validateFrameSamples(n) {
    if (!Number.isInteger(n) || n < MIN_FRAME_SAMPLES || n > MAX_FRAME_SAMPLES) {
      throw new RangeError(
        `frameSamples must be ${MIN_FRAME_SAMPLES}..${MAX_FRAME_SAMPLES} (20-40ms @16kHz), got ${n}`);
    }
    return n;
  }

  // Streaming linear resampler (state carried across calls).
  function createResampler(inRate, outRate = TARGET_RATE) {
    if (!(inRate > 0)) throw new RangeError('inRate must be > 0');
    const step = inRate / outRate;
    let buf = new Float32Array(0);
    let pos = 0;
    return {
      process(input) {
        if (inRate === outRate) return Float32Array.from(input);
        const merged = new Float32Array(buf.length + input.length);
        merged.set(buf, 0);
        merged.set(input, buf.length);
        const out = [];
        const eps = 1e-9;
        for (;;) {
          const i = Math.floor(pos + eps);
          if (i > merged.length - 1) break;
          const frac = Math.max(0, pos - i);
          if (i + 1 > merged.length - 1 && frac > eps) break;
          const a = merged[i];
          const b = i + 1 < merged.length ? merged[i + 1] : a;
          out.push(a + (b - a) * frac);
          pos += step;
        }
        const consumed = Math.min(Math.floor(pos + eps), merged.length);
        buf = merged.slice(consumed);
        pos -= consumed;
        return Float32Array.from(out);
      },
    };
  }

  function floatToS16(x) {
    const c = Math.max(-1, Math.min(1, x));
    return c < 0 ? Math.round(c * 32768) : Math.round(c * 32767);
  }

  function packFrame(seq, samples) {
    const out = new Uint8Array(4 + samples.length * 2);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, seq >>> 0, true);
    for (let i = 0; i < samples.length; i++) dv.setInt16(4 + i * 2, floatToS16(samples[i]), true);
    return out;
  }

  function readSeq(frame) {
    return new DataView(frame.buffer, frame.byteOffset, frame.byteLength).getUint32(0, true);
  }

  function frameSampleCount(frame) {
    return (frame.byteLength - 4) / 2;
  }

  function createFramer({ inRate = TARGET_RATE, frameSamples = 320 } = {}) {
    validateFrameSamples(frameSamples);
    const resampler = createResampler(inRate, TARGET_RATE);
    let pending = new Float32Array(0);
    let seq = 0;
    return {
      push(input) {
        const r = resampler.process(input);
        const merged = new Float32Array(pending.length + r.length);
        merged.set(pending, 0);
        merged.set(r, pending.length);
        const frames = [];
        let off = 0;
        while (merged.length - off >= frameSamples) {
          frames.push(packFrame(seq++, merged.subarray(off, off + frameSamples)));
          off += frameSamples;
        }
        pending = merged.slice(off);
        return frames;
      },
    };
  }

  // AudioWorkletGlobalScope only.
  if (typeof registerProcessor === 'function' && typeof AudioWorkletProcessor !== 'undefined') {
    class CaptureProcessor extends AudioWorkletProcessor {
      constructor(options) {
        super();
        const po = (options && options.processorOptions) || {};
        this.framer = createFramer({
          inRate: po.inRate || sampleRate,
          frameSamples: po.frameSamples || 320,
        });
      }
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (ch && ch.length) {
          for (const f of this.framer.push(ch)) this.port.postMessage(f, [f.buffer]);
        }
        return true;
      }
    }
    registerProcessor('capture-processor', CaptureProcessor);
  }

  const api = {
    TARGET_RATE, MIN_FRAME_SAMPLES, MAX_FRAME_SAMPLES,
    validateFrameSamples, createResampler, packFrame, readSeq, frameSampleCount, createFramer,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else if (typeof globalThis !== 'undefined') globalThis.IdleTalkWorklet = api;
})();
