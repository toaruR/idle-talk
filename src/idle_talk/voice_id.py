"""話者識別 (設計書 6.4)。声の特徴 (話者埋め込み) で利用者を見分け、初めての声なら名前を聞いて覚える。

埋め込みは WeSpeaker の ONNX モデル (入力は Kaldi 互換の 80 次元 fbank、平均を引いたもの) を CPU で計算する。
保存済みの声とのコサイン類似度が threshold 以上なら同じ人。短い発話 (MIN_VOICE_MS 未満) は判定しない。
知らない声が ENROLL_N 回続けて同じ声なら登録し、名前は応答の中で LLM に尋ねさせ、返事から抜き出す。
同じ名前を名乗った登録はまとめる (マイクが変わって別人扱いになったときの戻し方)。声は外部に送らない。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from .store import Store

log = logging.getLogger("idle_talk.voice_id")

SAMPLE_RATE = 16000
MIN_VOICE_MS = 1500  # これより短い発話 (相づち等) は埋め込みが不安定なので判定せず、直前の話者を引き継ぐ
THRESHOLD = 0.6  # 同じ人とみなすコサイン類似度の既定値 ([voice_id].threshold)
ENROLL_N = 2  # 知らない声がこの回数続けて互いに一致したら登録する (物音・テレビの 1 回で登録しない)
MAX_WEIGHT = 20  # 平均の更新で過去の埋め込みに掛ける重みの上限 (声の変化に追従できるよう頭打ちにする)
ASK_MAX = 2  # 名前を尋ねるのは 1 人あたり起動中にこの回数まで (答えたくない人にしつこく聞かない)
NAME_MAX = 12  # 名前として受け付ける長さの上限 (超えたら抜き出しの失敗とみなす)

ASK_PROMPT = "\n\n今話している相手の名前をまだ知りません。返答の最後に、自然な流れで一度だけ名前 (呼び名) を尋ねてください。"
ASKED_PROMPT = "\n\n直前にあなたは相手の名前を尋ねました。相手が名乗っていれば、その名前で呼んで軽く挨拶してください。"
NAME_PROMPT = ("あなたは会話記録から名前を抜き出す係です。以下は、名前を尋ねられた利用者の返事です。"
               "利用者が自分の名前や呼び名を名乗っていれば、その名前だけを 1 行で出力してください (敬称は付けない)。"
               "名乗っていなければ「なし」とだけ出力してください。")
_HONORIFIC = re.compile(r"(?:さん|様|さま|です|だよ|と申します|といいます|と言います|って言います|って呼んで(?:ください)?)$")
_STRIP = "「」『』\"'。、．，.,!！?？ 　"

# ---- 特徴量 (Kaldi fbank 互換: 25ms 窓 / 10ms シフト / povey 窓 / プリエンファシス 0.97 / 80 メル) ----
_FRAME, _SHIFT, _NFFT, _NMEL = 400, 160, 512, 80


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)


def _mel_banks() -> np.ndarray:
    """(NMEL, NFFT/2+1) の三角フィルタ。low 20Hz、high はナイキスト。"""
    lo, hi = _mel(20.0), _mel(SAMPLE_RATE / 2)
    edges = lo + (hi - lo) * np.arange(_NMEL + 2) / (_NMEL + 1)
    m = _mel(np.arange(_NFFT // 2) * SAMPLE_RATE / _NFFT)
    left, center, right = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    w = np.maximum(0.0, np.minimum((m - left) / (center - left), (right - m) / (right - center)))
    return np.pad(w, ((0, 0), (0, 1)))  # ナイキストの bin は使わない (Kaldi と同じ)


_BANKS = _mel_banks()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_FRAME) / (_FRAME - 1))) ** 0.85


def fbank(pcm: bytes) -> np.ndarray:
    """16kHz mono s16le PCM -> (T, 80) の対数メル特徴 (時間方向の平均を引いたもの)。1 フレームに満たなければ (0, 80)。"""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)  # WeSpeaker は int16 の振幅のまま入れる
    n = 1 + (len(x) - _FRAME) // _SHIFT if len(x) >= _FRAME else 0
    if n == 0:
        return np.zeros((0, _NMEL), dtype=np.float32)
    idx = np.arange(_FRAME)[None, :] + _SHIFT * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr = np.concatenate([fr[:, :1] * (1 - 0.97), fr[:, 1:] - 0.97 * fr[:, :-1]], axis=1)
    power = np.abs(np.fft.rfft(fr * _WINDOW, _NFFT)) ** 2
    feats = np.log(np.maximum(power @ _BANKS.T, np.finfo(np.float32).eps))
    return (feats - feats.mean(axis=0)).astype(np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


class Embedder:
    """ONNX の話者埋め込みモデル。呼び出しは ASR と並行して別スレッドで行う。"""

    def __init__(self, model_path: str | Path, threads: int = 2) -> None:
        import onnxruntime as ort

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"voice model not found: {model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads  # TTS (AivisSpeech) も CPU なので取り合わない程度に抑える
        self._sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name

    def __call__(self, pcm: bytes) -> np.ndarray | None:
        feats = fbank(pcm)
        if len(feats) == 0:
            return None
        return _unit(self._sess.run(None, {self._input: feats[None]})[0])


# ---- 声と名前の対応 (speakers テーブル) ----
class VoiceBook:
    def __init__(self, store: Store, threshold: float = THRESHOLD) -> None:
        self.store = store
        self.threshold = threshold
        self._rows: dict[int, dict] = {
            r["id"]: {"name": r["name"], "emb": np.frombuffer(r["embedding"], dtype=np.float32).copy(), "n": r["n"]}
            for r in store.query("SELECT id, name, embedding, n FROM speakers")}
        self._pending: list[np.ndarray] = []  # 登録前の知らない声

    def name(self, sid: int | None) -> str | None:
        row = self._rows.get(sid)
        return row["name"] if row else None

    def identify(self, emb: np.ndarray, now_ms: int) -> int | None:
        """一致した (または今回登録した) 話者 id。登録前の知らない声なら None。一致したら声の平均を更新する。"""
        emb = _unit(emb)
        best, score = None, -1.0
        for sid, row in self._rows.items():
            s = float(row["emb"] @ emb)
            if s > score:
                best, score = sid, s
        if best is not None and score >= self.threshold:
            row = self._rows[best]
            row["emb"] = _unit(row["emb"] * min(row["n"], MAX_WEIGHT) + emb)
            row["n"] += 1
            self.store.execute("UPDATE speakers SET embedding=?, n=?, updated_at=? WHERE id=?",
                               (row["emb"].tobytes(), row["n"], now_ms, best))
            return best
        if self._pending and float(_unit(np.mean(self._pending, axis=0)) @ emb) >= self.threshold:
            self._pending.append(emb)
        else:
            self._pending = [emb]
        if len(self._pending) < ENROLL_N:
            return None
        mean, self._pending = _unit(np.mean(self._pending, axis=0)), []
        sid = self.store.execute("INSERT INTO speakers(name, embedding, n, updated_at) VALUES (NULL,?,?,?)",
                                 (mean.tobytes(), ENROLL_N, now_ms))
        self._rows[sid] = {"name": None, "emb": mean, "n": ENROLL_N}
        log.info("new voice enrolled: speaker %d (score to nearest %.2f)", sid, score)
        return sid

    def set_name(self, sid: int, name: str, now_ms: int) -> int:
        """名前を付ける。同じ名前の登録が既にあれば声をそちらへまとめて sid を消し、残った id を返す。"""
        same = next((k for k, r in self._rows.items() if k != sid and r["name"] == name), None)
        if same is None:
            self._rows[sid]["name"] = name
            self.store.execute("UPDATE speakers SET name=?, updated_at=? WHERE id=?", (name, now_ms, sid))
            return sid
        a, b = self._rows[same], self._rows.pop(sid)
        a["emb"] = _unit(a["emb"] * min(a["n"], MAX_WEIGHT) + b["emb"] * min(b["n"], MAX_WEIGHT))
        a["n"] += b["n"]
        self.store.execute("UPDATE speakers SET embedding=?, n=?, updated_at=? WHERE id=?",
                           (a["emb"].tobytes(), a["n"], now_ms, same))
        self.store.execute("DELETE FROM speakers WHERE id=?", (sid,))
        return same


# ---- プロンプト ----
def speaker_prompt(name: str | None) -> str:
    """システムプロンプトに添える、今の話し相手の節。名前が無ければ空文字。"""
    if not name:
        return ""
    return f"\n\n今話している相手は「{name}」さんです。ときどき名前で呼びかけてください。"


def name_messages(utterances: list[str]) -> list[dict]:
    body = "\n".join(f"利用者: {u}" for u in utterances)
    return [{"role": "system", "content": NAME_PROMPT}, {"role": "user", "content": body}]


def parse_name(text: str) -> str | None:
    """LLM 出力の最初の行から名前を取り出す。「なし」・長すぎる・語の無いものは None。"""
    line = next((s.strip() for s in text.splitlines() if s.strip()), "")
    line = re.sub(r"^(?:名前|呼び名)\s*[:：]\s*", "", line).strip(_STRIP)
    line = _HONORIFIC.sub("", line).strip(_STRIP)
    if not line or "なし" in line or len(line) > NAME_MAX or not any(ch.isalnum() for ch in line):
        return None
    return line
