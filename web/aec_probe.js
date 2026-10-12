'use strict';
// ブラウザでは classic script として共通のグローバルスコープに読まれるため、IIFE で閉じる。
(function () {
  // Speaker-mode AEC probe (design 6.2): measure getUserMedia({echoCancellation:true}).
  // Browser APIs are injected; the app records echo_chain / cancel events separately.

  const AEC_CONSTRAINTS = Object.freeze({ audio: Object.freeze({ echoCancellation: true }) });

  function probeConstraints() {
    return { audio: { echoCancellation: true } };
  }

  async function runProbe(getUserMedia) {
    const stream = await getUserMedia(probeConstraints());
    try {
      const track = stream.getAudioTracks()[0];
      const s = track && track.getSettings ? track.getSettings() : {};
      return { echoCancellation: s.echoCancellation === true };
    } finally {
      if (stream.getTracks) stream.getTracks().forEach((t) => t.stop());
    }
  }

  const api = { AEC_CONSTRAINTS, probeConstraints, runProbe };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else globalThis.IdleTalkAecProbe = api;
})();
