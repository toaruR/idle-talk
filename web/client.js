'use strict';
// ブラウザでは classic script として共通のグローバルスコープに読まれるため、IIFE で閉じる。
(function () {
  // Browser client: WebSocket <-> Director (design 3.1, 12.1). Browser APIs are injected.

  const W = (typeof module !== 'undefined' && module.exports)
    ? require('./worklet.js') : globalThis.IdleTalkWorklet;
  const P = (typeof module !== 'undefined' && module.exports)
    ? require('./player.js') : globalThis.IdleTalkPlayer;

  const PROTOCOL_VERSION = 1;
  const SAMPLE_RATE = 16000;
  const FRAME_SAMPLES = 320; // 20ms @16kHz
  const RECONNECT_MS = 2000; // 切断から再接続までの間隔
  // ドラマモードの画面 (設計書 6.5): サーバに求められたら 1 枚をこの大きさに収めて JPEG で返す (640x360 で約 106 トークン)
  const SNAP_W = 640;
  const SNAP_H = 360;
  const SNAP_QUALITY = 0.7;
  // 画面モード (設計書 6.7): サーバが前の画面との差を測るための縮小画像 (明るさ 1 バイトずつ)。src/idle_talk/screen.py と一致させる
  const THUMB_W = 32;
  const THUMB_H = 18;

  function wsUrl(port) {
    return `ws://127.0.0.1:${port}/ws`;
  }

  function micConstraints() {
    // AGC off: the server's VAD calibrates its threshold from the noise floor, which AGC would keep shifting.
    return { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1 } };
  }

  // ドラマモード: 共有したタブ (画面) の音だけを使う。Chrome は video 無しの共有を受け付けないので映像は取ってすぐ止める
  function displayConstraints() {
    return {
      video: true,
      // restrictOwnAudio: 画面全体 + システム音声で共有したとき、このタブが鳴らす AI の声を入れない (Chrome 141+。未対応なら無視される)
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, restrictOwnAudio: true },
      systemAudio: 'include',
    };
  }

  function makeHello(clientId) {
    return { t: 'hello', version: PROTOCOL_VERSION, sample_rate: SAMPLE_RATE, client_id: clientId };
  }

  function makeUi(settings) {
    const out = { t: 'ui' };
    for (const k of ['mute', 'on', 'radio', 'interval_s', 'quiet_start', 'quiet_end', 'speaker', 'drama', 'drama_mic', 'screen', 'camera', 'pc', 'persona']) {
      if (settings && settings[k] !== undefined) out[k] = settings[k];
    }
    return out;
  }

  function stateLabel(msg, names) {
    const label = (names && names[msg.state]) || msg.state;
    // 休止のチェック中は状態も「休止」になるので重ねて書かない (状態がまだ切り替わる前だけ添える)
    return msg.muted && msg.state !== 'rest' ? `${label} (休止)` : label;
  }

  // 一時的な失敗 (LLM・TTS・ASR) のエラー表示は、次に音声の再生が始まったら (応答が通ったら) 消す
  const TRANSIENT_ERRORS = [5001, 5002, 5003];
  function errorResolved(code, playback) {
    return TRANSIENT_ERRORS.includes(code) && Boolean(playback) && playback.t === 'playback' && playback.state === 'started';
  }

  // 縦横比を保って SNAP_W x SNAP_H に収める (拡大はしない)
  function snapSize(w, h) {
    const s = Math.min(SNAP_W / w, SNAP_H / h, 1);
    return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
  }

  // RGBA の画素 (THUMB_W x THUMB_H) を明るさ 1 バイトずつにして base64 にする
  function grayThumb(rgba) {
    let bin = '';
    for (let i = 0; i < rgba.length; i += 4) {
      bin += String.fromCharCode(Math.round(0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]));
    }
    return btoa(bin);
  }

  // 映像トラックの今の 1 コマを { jpeg, thumb } (どちらも base64) にする。描けなければ理由つきで投げる。
  // Browser only (ImageCapture / canvas)
  async function grabSnap(track, createCanvas) {
    if (!track) throw new Error('no video track');
    if (track.readyState !== 'live') throw new Error(`video track ${track.readyState}`);
    if (typeof ImageCapture === 'undefined') throw new Error('ImageCapture unsupported');
    let frame;
    try {
      frame = await new ImageCapture(track).grabFrame();
    } catch (e) {
      throw new Error(`grabFrame${track.muted ? ' (track muted)' : ''}: ${e && e.name ? `${e.name}: ${e.message}` : e}`);
    }
    try {
      const [w, h] = snapSize(frame.width, frame.height);
      const canvas = createCanvas(w, h);
      canvas.getContext('2d').drawImage(frame, 0, 0, w, h);
      const url = canvas.toDataURL('image/jpeg', SNAP_QUALITY);
      const small = createCanvas(THUMB_W, THUMB_H).getContext('2d');
      small.drawImage(frame, 0, 0, THUMB_W, THUMB_H);
      const thumb = grayThumb(small.getImageData(0, 0, THUMB_W, THUMB_H).data);
      return { jpeg: url.slice(url.indexOf(',') + 1), thumb };
    } finally {
      frame.close();
    }
  }

  // opts: { port, WebSocketCtor, player, onState(msg), onError(msg), onSearch(msg), onLinks(msg), onSnap(msg), onOpen(), onClose(),
  //         clientId, reconnectMs, setTimeoutFn }
  // 接続が切れた・つながらなかったら reconnectMs 後につなぎ直す (サーバの再起動中に開始を押しても待てば動く)。close() で止める
  function createClient(opts) {
    const { port, WebSocketCtor, player, onState, onError, onSearch, onLinks, onSnap, onOpen, onClose } = opts;
    const clientId = opts.clientId || `web-${Math.random().toString(36).slice(2, 10)}`;
    const reconnectMs = opts.reconnectMs || RECONNECT_MS;
    const setTimeoutFn = opts.setTimeoutFn || setTimeout;
    let ws = null;
    let closed = false;
    // hello 前・切断中は送らない (CONNECTING の send は例外、サーバは hello 前のメッセージを受け付けない)
    let ready = false;
    const sendJson = (m) => { if (ready) ws.send(JSON.stringify(m)); };

    function connect() {
      ws = new WebSocketCtor(wsUrl(port));
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => {
        if (closed) return;
        ready = true;
        sendJson(makeHello(clientId));
        if (onOpen) onOpen(); // hello の後 (サーバは hello 前のメッセージを受け付けない)
      };
      ws.onclose = () => {
        ready = false;
        if (closed) return;
        if (player && player.reset) player.reset(); // 前の接続の音は鳴らさない。新しい接続の epoch は 0 から
        if (onClose) onClose();
        setTimeoutFn(() => { if (!closed) connect(); }, reconnectMs);
      };
      ws.onmessage = onMessage;
    }

    function onMessage(ev) {
      if (closed) return;
      const d = ev.data;
      if (typeof d !== 'string') { if (player) player.pushBinary(d); return; }
      let msg;
      try { msg = JSON.parse(d); } catch (_e) { return; }
      if (!msg || typeof msg !== 'object') return;
      if (msg.t === 'ping') sendJson({ t: 'pong', ts: msg.ts });
      else if (msg.t === 'state') { if (onState) onState(msg); }
      else if (msg.t === 'error') { if (onError) onError(msg); }
      else if (msg.t === 'search') { if (onSearch) onSearch(msg); }
      else if (msg.t === 'links') { if (onLinks) onLinks(msg); }
      else if (msg.t === 'snap') { if (onSnap) onSnap(msg); }
      else if (player) player.handleMessage(msg);
    }

    connect();
    return {
      get ws() { return ws; },
      clientId,
      sendFrame(frame) { if (ready) ws.send(frame); },
      sendUi(settings) { sendJson(makeUi(settings)); },
      sendJson,
      close() {
        if (closed) return;
        closed = true;
        ready = false;
        if (player && player.reset) player.reset();
        ws.close();
      },
    };
  }

  // Start mic capture -> worklet frames -> client.sendFrame. Browser only.
  async function startCapture({ getUserMedia, AudioContextCtor, workletUrl, client }) {
    const stream = await getUserMedia(micConstraints());
    let ctx, node, source, stopped = false;
    const stop = () => {
      if (stopped) return;
      stopped = true;
      if (node) { node.port.onmessage = null; node.disconnect(); }
      if (source) source.disconnect();
      for (const track of stream.getTracks()) track.stop();
      if (ctx) ctx.close();
    };
    try {
      ctx = new AudioContextCtor();
      await ctx.audioWorklet.addModule(workletUrl);
      node = new AudioWorkletNode(ctx, 'capture-processor', {
        processorOptions: { inRate: ctx.sampleRate, frameSamples: FRAME_SAMPLES },
      });
      node.port.onmessage = (ev) => client.sendFrame(ev.data);
      source = ctx.createMediaStreamSource(stream);
      source.connect(node);
      return { ctx, stream, node, stop };
    } catch (e) {
      stop();
      throw e;
    }
  }

  // 共有したタブの音 -> worklet (seq に DRAMA_FLAG) -> client.sendFrame。共有が終わったら onEnded。Browser only.
  // vision (真偽値か Promise。共有の操作を待たせないよう共有の後で待つ) なら映像トラックを残し、snapshot() で今の画面を
  // { jpeg } (JPEG の base64) で返す。描けなければ { reason } (サーバのログで原因を見るため)
  // 失敗した段階をエラーに付ける (ブラウザの例外は "Not supported" のように段階が分からない)
  async function step(name, fn) {
    try {
      return await fn();
    } catch (e) {
      throw new Error(`${name}: ${e && e.name ? `${e.name}: ${e.message}` : e}`);
    }
  }

  async function startDramaCapture({ getDisplayMedia, AudioContextCtor, workletUrl, client, onEnded, vision, createCanvas }) {
    const stream = await step('タブの共有', async () => {
      try {
        return await getDisplayMedia(displayConstraints());
      } catch (e) {
        // 音声の詳細な制約や systemAudio を受け付けないブラウザ: 最小の制約で取り直す
        if (!e || (e.name !== 'NotSupportedError' && e.name !== 'TypeError' && e.name !== 'OverconstrainedError')) throw e;
        return getDisplayMedia({ video: true, audio: true });
      }
    });
    const audio = stream.getAudioTracks();
    if (!audio.length) {
      for (const t of stream.getVideoTracks()) t.stop();
      throw new Error('音声が共有されていません。共有の画面で「タブの音声も共有する」をオンにしてください');
    }
    const keep = Boolean(await vision);
    const video = keep ? stream.getVideoTracks() : [];
    if (!keep) for (const t of stream.getVideoTracks()) t.stop();
    const ctx = await step('音声の準備', () => new AudioContextCtor());
    await step('音声処理の読み込み', () => ctx.audioWorklet.addModule(workletUrl));
    const node = await step('音声処理の作成', () => new AudioWorkletNode(ctx, 'capture-processor', {
      processorOptions: { inRate: ctx.sampleRate, frameSamples: FRAME_SAMPLES, seqBase: W.DRAMA_FLAG },
    }));
    node.port.onmessage = (ev) => client.sendFrame(ev.data);
    await step('共有した音の接続', () => ctx.createMediaStreamSource(new MediaStream(audio)).connect(node));
    let stopped = false;
    const stop = () => {
      if (stopped) return;
      stopped = true;
      for (const t of [...audio, ...video]) t.stop();
      ctx.close();
      if (onEnded) onEnded();
    };
    audio[0].addEventListener('ended', stop); // ブラウザの「共有を停止」
    const snapshot = snapshotter(video, () => stopped, keep ? 'no video track' : 'vision off at share start', createCanvas);
    return { ctx, stream, node, stop, snapshot };
  }

  // snapshot(): 今の画面を { jpeg, thumb } で返す。止めた後・映像なし・描けなければ { reason }
  function snapshotter(video, isStopped, noVideo, createCanvas) {
    const canvas = createCanvas || ((w, h) => Object.assign(document.createElement('canvas'), { width: w, height: h }));
    return async () => {
      if (isStopped()) return { reason: 'capture stopped' };
      if (!video.length) return { reason: noVideo };
      try {
        return await grabSnap(video[0], canvas);
      } catch (e) {
        // 描けなければ画像なしで一言を言わせる
        return { reason: String((e && e.message) || e).slice(0, 200) };
      }
    };
  }

  // 画面モード (設計書 6.7): 共有した画面 (映像だけ) を残し、求められたら snapshot() で返す。共有が終わったら onEnded。
  // vision (真偽値か Promise) が偽なら LLM が画像を読めないので共有を止めてエラーにする。Browser only.
  async function startScreenCapture({ getDisplayMedia, onEnded, vision, createCanvas }) {
    const stream = await step('画面の共有', () => getDisplayMedia({ video: true, audio: false }));
    const video = stream.getVideoTracks();
    for (const t of stream.getAudioTracks()) t.stop();
    if (!(await vision)) {
      for (const t of video) t.stop();
      throw new Error('LLM が画像を読めないため使えません (idle_talk.toml の [llm].mmproj を指定してください)');
    }
    let stopped = false;
    const stop = () => {
      if (stopped) return;
      stopped = true;
      for (const t of video) t.stop();
      if (onEnded) onEnded();
    };
    if (video.length) video[0].addEventListener('ended', stop); // ブラウザの「共有を停止」
    return { stream, stop, snapshot: snapshotter(video, () => stopped, 'no video track', createCanvas) };
  }

  // カメラモード (設計書 6.9): Web カメラの映像を残し、画面モードと同じく求められたら snapshot() で返す。Browser only.
  async function startCameraCapture({ getUserMedia, onEnded, vision, createCanvas }) {
    if (!(await vision)) {
      throw new Error('LLM が画像を読めないため使えません (idle_talk.toml の [llm].mmproj を指定してください)');
    }
    const stream = await step('カメラ', () => getUserMedia({ video: { width: { ideal: SNAP_W }, height: { ideal: SNAP_H } }, audio: false }));
    const video = stream.getVideoTracks();
    let stopped = false;
    const stop = () => {
      if (stopped) return;
      stopped = true;
      for (const t of video) t.stop();
      if (onEnded) onEnded();
    };
    if (video.length) video[0].addEventListener('ended', stop); // カメラの取り外し・権限の取り消し
    return { stream, stop, snapshot: snapshotter(video, () => stopped, 'no video track', createCanvas) };
  }

  // 画面モードの [screen].follow_window:サーバが前面のウィンドウを撮るので共有はせず、オン・オフだけ持つ。
  // 戻り値は startScreenCapture と同じ形 (snapshot は求められない)
  async function startWindowWatch({ onEnded, vision }) {
    if (!(await vision)) {
      throw new Error('LLM が画像を読めないため使えません (idle_talk.toml の [llm].mmproj を指定してください)');
    }
    let stopped = false;
    const stop = () => {
      if (stopped) return;
      stopped = true;
      if (onEnded) onEnded();
    };
    return { stop, snapshot: async () => ({ reason: 'follow_window: captured by the server' }) };
  }

  const api = {
    PROTOCOL_VERSION, SAMPLE_RATE, FRAME_SAMPLES, SNAP_W, SNAP_H, THUMB_W, THUMB_H, W, P,
    wsUrl, micConstraints, displayConstraints, makeHello, makeUi, stateLabel, snapSize, grayThumb, errorResolved,
    createClient, startCapture, startDramaCapture, startScreenCapture, startCameraCapture, startWindowWatch,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else if (typeof globalThis !== 'undefined') globalThis.IdleTalkClient = api;
})();
