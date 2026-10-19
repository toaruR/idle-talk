"""相手モードのモニター (docs/plans/2026-10-18-partner-mode.md §2.3)。AI の声を Chrome の外からスピーカーに鳴らす。

相手 (ChatGPT 等) のタブのエコー除去は、Chrome が鳴らした声を、別のタブの音でも消した (実測 2026-10-18)。
Chrome の外のプロセスが鳴らした声は消さないので、相手モード中は Director が鳴らす。
再生の報告はブラウザの playback と同じ形 (started / finished) で返し、Director は今と同じ処理で履歴を確定する。
"""
from __future__ import annotations

import array
import logging
import threading
from collections import deque
from collections.abc import Callable

from .clients import TTS_RATE

log = logging.getLogger("idle_talk.monitor")
BLOCK_MS = 20  # この単位で書き込み、取消と音量の変更をすぐ効かせる (マイクもこの単位で読む)
MIC_RATE = 16000  # 上りと同じ 16kHz mono s16le


def open_device(name: str, rate: int = TTS_RATE):
    """name を含む出力機器 (空なら Windows の既定) を開く。sounddevice が無い・機器が無ければ OSError。"""
    try:
        import sounddevice as sd
    except ImportError as e:  # 相手モードだけで使う。普段の動作には要らない
        raise OSError("sounddevice is not installed (pip install sounddevice)") from e
    device = None
    if name:
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0 and name in d["name"]:
                device = i
                break
        else:
            raise OSError(f"output device not found: {name!r}")
    try:
        stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16", device=device)
        stream.start()
    except Exception as e:  # noqa: BLE001  PortAudioError 等は機器を開けなかったとして扱う
        raise OSError(f"cannot open output device {name or '(default)'}: {e}") from e
    return stream


def open_input(name: str, rate: int = MIC_RATE):
    """name を含む入力機器 (空なら Windows の既定の録音機器) を 16kHz mono s16le で開く。開けなければ OSError。"""
    try:
        import sounddevice as sd
    except ImportError as e:
        raise OSError("sounddevice is not installed (pip install sounddevice)") from e
    device = None
    if name:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and name in d["name"]:
                device = i
                break
        else:
            raise OSError(f"input device not found: {name!r}")
    try:
        stream = sd.RawInputStream(samplerate=rate, channels=1, dtype="int16", device=device,
                                   blocksize=rate * BLOCK_MS // 1000)
        stream.start()
    except Exception as e:  # noqa: BLE001
        raise OSError(f"cannot open input device {name or '(default)'}: {e}") from e
    return stream


class MicCapture:
    """PC と話すモードのマイク。ブラウザ (Chrome) のエコー除去を通さずに録る。

    Chrome のエコー除去は Chrome が鳴らした音を消すので、ブラウザの ChatGPT の声は idle-talk のタブのマイクから
    ほぼ消えた (実機 2026-10-19)。feed(pcm) は録音スレッドから 20ms ごとに呼ばれる (呼び出し側でループへ渡す)。"""

    def __init__(self, feed: Callable[[bytes], None], open_stream: Callable[[], object],
                 on_lost: Callable[[], None] = lambda: None) -> None:
        self._feed = feed
        self._on_lost = on_lost
        self._open = open_stream
        self._stream = None
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """機器を開いて録音スレッドを始める。開けなければ OSError。"""
        if self._running:
            return
        self._stream = self._open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="mic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:  # noqa: BLE001
                log.info("mic close failed: %s", e)
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None

    def _loop(self) -> None:
        n = MIC_RATE * BLOCK_MS // 1000
        while self._running:
            stream = self._stream
            if stream is None:
                return
            try:
                data, _overflow = stream.read(n)
            except Exception as e:  # noqa: BLE001  機器が抜けた等: 録音をやめて知らせる (E6)
                if self._running:
                    log.warning("mic read failed: %s", e)
                    self._on_lost()
                return
            self._feed(bytes(data))


class Monitor:
    """report(msg) は再生スレッドから呼ばれる (呼び出し側でイベントループへ渡す)。is_current(epoch) で旧世代を捨てる。"""

    def __init__(self, report: Callable[[dict], None], is_current: Callable[[int], bool],
                 open_stream: Callable[[], object], rate: int = TTS_RATE) -> None:
        self._report = report
        self._is_current = is_current
        self._open = open_stream
        self.rate = rate
        self.gain = 1.0
        self._q: deque[tuple[int, int, int, bytes]] = deque()
        self._cv = threading.Condition()
        self._stream = None
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """機器を開いて再生スレッドを始める。開けなければ OSError (呼び出し側で相手モードを始めない)。"""
        if self._running:
            return
        self._stream = self._open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._running = False
            self._q.clear()
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:  # noqa: BLE001  閉じられなくても困らない
                log.info("monitor close failed: %s", e)

    def play(self, ep: int, utt: int, chunk: int, pcm: bytes) -> None:
        with self._cv:
            self._q.append((ep, utt, chunk, pcm))
            self._cv.notify_all()

    def duck(self, gain: float) -> None:
        self.gain = gain

    def resume(self) -> None:
        self.gain = 1.0

    def cancel(self) -> None:
        """取消: 待ちのチャンクを捨てる。再生中のチャンクは次のブロックで旧世代として止まる。音量は戻す。"""
        with self._cv:
            self._q = deque(c for c in self._q if self._is_current(c[0]))
        self.gain = 1.0

    def _loop(self) -> None:
        block = self.rate * BLOCK_MS // 1000 * 2  # bytes
        while True:
            with self._cv:
                while self._running and not self._q:
                    self._cv.wait()
                if not self._running:
                    return
                ep, utt, chunk, pcm = self._q.popleft()
            if not self._is_current(ep):
                continue  # 旧世代: 報告もしない (ブラウザの報告もサーバで捨てられる)
            self._send(ep, utt, chunk, "started", 0)
            done = 0
            for i in range(0, len(pcm), block):
                if not self._running or not self._is_current(ep):
                    break
                self._write(pcm[i:i + block])
                done = i + block
            else:
                self._send(ep, utt, chunk, "finished", len(pcm) // 2 * 1000 // self.rate)
                continue
            log.info("monitor stopped chunk %d/%d/%d at %d ms", ep, utt, chunk, done // 2 * 1000 // self.rate)

    def _write(self, data: bytes) -> None:
        g = self.gain
        if g != 1.0:
            a = array.array("h", data[: len(data) - len(data) % 2])
            for i, s in enumerate(a):
                a[i] = int(s * g)
            data = a.tobytes()
        try:
            self._stream.write(data)
        except Exception as e:  # noqa: BLE001  機器が抜けた等: このチャンクは鳴らせなかったが報告は進める
            log.warning("monitor write failed: %s", e)

    def _send(self, ep: int, utt: int, chunk: int, state: str, pos: int) -> None:
        self._report({"t": "playback", "epoch": ep, "utterance_id": utt, "chunk_id": chunk,
                      "position_ms": min(pos, 600000), "state": state})
