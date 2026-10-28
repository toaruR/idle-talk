"""話者識別 (設計書 6.4)。声の特徴 (話者埋め込み) で利用者を見分け、初めての声なら名前を聞いて覚える。

埋め込みは WeSpeaker の ONNX モデル (入力は Kaldi 互換の 80 次元 fbank、平均を引いたもの) を CPU で計算する。
保存済みの声とのコサイン類似度が threshold 以上なら同じ人。短い発話 (MIN_VOICE_MS 未満) は判定しない。
知らない声が ENROLL_N 回続けて同じ声なら登録し、名前は応答の中で LLM に尋ねさせ、返事から抜き出す。
同じ名前を名乗った登録はまとめる (マイクが変わって別人扱いになったときの戻し方)。声は外部に送らない。
文字起こしモード (設計書 6.10) では、名前の無い声に声の高さと抑揚から「低く落ち着いた声3」のような仮名を付ける。
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
FEW_N = 3  # 一致がこの回数以下の名前の無い声は [voice_id].forget_few_days で早めに消す
DAY_MS = 86_400_000

# 4B は「名前を尋ねて」という指示を無視することが多い (実機 2026-10-14)。質問は返答の後に定型文 ASK_QUESTION で読む
UNNAMED_PROMPT = "\n\n今話している相手の名前はまだ知りません (知っているふりはしないでください)。"
ASK_PROMPT = UNNAMED_PROMPT + "返答のあとで名前を尋ねるので、返答の最後に質問はしないでください。"
ASK_QUESTION = "ところで、まだ名前を聞いてなかったね。なんて呼べばいい？"
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


def forget_unnamed(store: Store, now_ms: int, days: int, few_days: int) -> int:
    """最後に一致してから days 日 (一致 FEW_N 回以下なら few_days 日) 経った名前の無い声を消し、消した数を返す。
    文字起こし・ドラマのテレビの声が溜まると、利用者の声がそれに一致して名前をまた尋ねる (実機 2026-10-26)。"""
    big = now_ms + DAY_MS  # 0 日 (消さない) は未来の時刻にして何も当たらないようにする
    where = ("name IS NULL AND (updated_at < ? OR (n <= ? AND updated_at < ?))",
             (now_ms - days * DAY_MS if days else -big, FEW_N, now_ms - few_days * DAY_MS if few_days else -big))
    n = store.query(f"SELECT COUNT(*) FROM speakers WHERE {where[0]}", where[1])[0][0]
    if n:
        store.execute(f"DELETE FROM speakers WHERE {where[0]}", where[1])
        log.info("forgot %d unnamed voice(s)", n)
    return n


# ---- 声と名前の対応 (speakers テーブル) ----
class VoiceBook:
    def __init__(self, store: Store, threshold: float = THRESHOLD) -> None:
        self.store = store
        self.threshold = threshold
        self._rows: dict[int, dict] = {
            r["id"]: {"name": r["name"], "emb": np.frombuffer(r["embedding"], dtype=np.float32).copy(), "n": r["n"],
                      "alias": r["alias"]}
            for r in store.query("SELECT id, name, embedding, n, alias FROM speakers")}
        self._pending: list[np.ndarray] = []  # 登録前の知らない声
        self.enrolled: int | None = None  # 直前の identify で登録した id (ENROLL_N 回続けて一致済み)

    def name(self, sid: int | None) -> str | None:
        row = self._rows.get(sid)
        return row["name"] if row else None

    def alias(self, sid: int | None) -> str | None:
        row = self._rows.get(sid)
        return row["alias"] if row else None

    def set_alias(self, sid: int, word: str, now_ms: int) -> str:
        """仮名「word + 通し番号」を付けて返す。番号はこれまでに付けた仮名の最大 + 1 (特徴の言葉が違っても通し)。"""
        nums = [int(m.group()) for r in self._rows.values() if r["alias"] and (m := re.search(r"\d+$", r["alias"]))]
        alias = f"{word}{max(nums, default=0) + 1}"
        self._rows[sid]["alias"] = alias
        self.store.execute("UPDATE speakers SET alias=?, updated_at=? WHERE id=?", (alias, now_ms, sid))
        return alias

    def named_score(self, emb: np.ndarray) -> float | None:
        """名前の分かった相手の声との類似度の最大値。名前の分かった相手がいなければ None。"""
        emb = _unit(emb)
        return max((float(r["emb"] @ emb) for r in self._rows.values() if r["name"]), default=None)

    def identify(self, emb: np.ndarray, now_ms: int, enroll: bool = True, threshold: float | None = None,
                 verbose: bool = True) -> int | None:
        """一致した (または今回登録した) 話者 id。登録前の知らない声なら None。一致したら声の平均を更新する。
        enroll が偽なら知らない声を登録しない (登録の候補にも数えない)。threshold は今回だけ使う閾値 (文字起こし)。"""
        emb = _unit(emb)
        threshold = self.threshold if threshold is None else threshold
        self.enrolled = None
        best, score = None, -1.0
        for sid, row in self._rows.items():
            s = float(row["emb"] @ emb)
            if s > score:
                best, score = sid, s
        log.log(logging.INFO if verbose else logging.DEBUG,  # 閾値の調整用 (0.5 秒ごとの窓は debug)
                "voice similarity to nearest: %.2f (speaker %s)", score, best)
        if best is not None and score >= threshold:
            row = self._rows[best]
            row["emb"] = _unit(row["emb"] * min(row["n"], MAX_WEIGHT) + emb)
            row["n"] += 1
            self.store.execute("UPDATE speakers SET embedding=?, n=?, updated_at=? WHERE id=?",
                               (row["emb"].tobytes(), row["n"], now_ms, best))
            return best
        if not enroll:
            return None
        if self._pending and float(_unit(np.mean(self._pending, axis=0)) @ emb) >= threshold:
            self._pending.append(emb)
        else:
            self._pending = [emb]
        if len(self._pending) < ENROLL_N:
            return None
        n, mean, self._pending = len(self._pending), _unit(np.mean(self._pending, axis=0)), []
        sid = self.store.execute("INSERT INTO speakers(name, embedding, n, updated_at) VALUES (NULL,?,?,?)",
                                 (mean.tobytes(), n, now_ms))
        self._rows[sid] = {"name": None, "emb": mean, "n": n, "alias": None}
        self.enrolled = sid
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
        a["alias"] = a["alias"] or b["alias"]
        self.store.execute("UPDATE speakers SET embedding=?, n=?, updated_at=? WHERE id=?",
                           (a["emb"].tobytes(), a["n"], now_ms, same))
        self.store.execute("DELETE FROM speakers WHERE id=?", (sid,))
        return same


# ---- 文字起こしの仮名 (設計書 6.10): 声の高さ (F0 の中央値) と抑揚 (F0 の散らばり) を言葉にする ----
_PITCH_FRAME, _PITCH_SHIFT, _PITCH_NFFT = 640, 160, 2048  # 40ms 窓 / 10ms シフト (70Hz の 2 周期以上)
F0_MIN, F0_MAX = 70.0, 500.0  # 話し声の基本周波数として探す範囲 (Hz)
VOICED_MIN = 0.45  # 自己相関の山がこれ以上のフレームを有声とみなす
VOICED_FRAMES_MIN = 20  # 有声フレームがこれ未満 (0.2 秒) なら特徴を出さない
# 声の高さの区切り (Hz、暫定)。成人男性の話し声は 100〜150Hz、女性は 200〜250Hz 前後が多い
PITCH_WORDS = ((125.0, "低い"), (165.0, "やや低い"), (220.0, "やや高い"), (float("inf"), "高い"))
CALM_ST, LIVELY_ST = 2.0, 3.5  # F0 の標準偏差 (半音、暫定)。これ未満なら「落ち着いた」、以上なら「抑揚のある」


def pitch_track(pcm: bytes) -> np.ndarray:
    """16kHz mono s16le PCM -> 有声フレームの F0 (Hz) の列。自己相関の山で見る (音量の小さいフレームは除く)。"""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    n = 1 + (len(x) - _PITCH_FRAME) // _PITCH_SHIFT if len(x) >= _PITCH_FRAME else 0
    if n == 0:
        return np.zeros(0)
    fr = x[np.arange(_PITCH_FRAME)[None, :] + _PITCH_SHIFT * np.arange(n)[:, None]]
    fr = fr - fr.mean(axis=1, keepdims=True)
    level = np.sqrt((fr ** 2).mean(axis=1))
    fr = fr[level >= max(100.0, 0.25 * float(np.percentile(level, 90)))]  # 無音・息継ぎを除く
    if len(fr) == 0:
        return np.zeros(0)
    ac = np.fft.irfft(np.abs(np.fft.rfft(fr * np.hanning(_PITCH_FRAME), _PITCH_NFFT)) ** 2)
    lo, hi = int(SAMPLE_RATE / F0_MAX), int(SAMPLE_RATE / F0_MIN)
    lag = lo + np.argmax(ac[:, lo:hi + 1], axis=1)
    peak = ac[np.arange(len(ac)), lag] / np.maximum(ac[:, 0], 1e-9)
    return SAMPLE_RATE / lag[peak >= VOICED_MIN]


def voice_word(pcm: bytes) -> str:
    """仮名の言葉 (番号の前)。「低く落ち着いた声」「やや高い声」等。有声の部分が短すぎれば「声」。"""
    f0 = pitch_track(pcm)
    if len(f0) < VOICED_FRAMES_MIN:
        return "声"
    pitch = next(w for limit, w in PITCH_WORDS if float(np.median(f0)) < limit)
    spread = float(np.std(12 * np.log2(f0)))  # 半音
    tone = "落ち着いた" if spread < CALM_ST else "抑揚のある" if spread >= LIVELY_ST else ""
    return f"{pitch[:-1]}く{tone}声" if tone else f"{pitch}声"


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
