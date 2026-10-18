"""メッセージスキーマ・状態値・バイナリヘッダ (設計書 12.1)。"""
from __future__ import annotations

import struct

from jsonschema import Draft7Validator

U32_MAX = 4294967295
MAX_CHUNK_PCM = 1048576  # 1MiB
HEADER_SIZE = 16
# 上りバイナリの seq の最上位ビット: 立っていれば共有したタブの音 (ドラマモード)、無ければマイク
DRAMA_FLAG = 0x80000000

MESSAGE_SCHEMA = {"type":"object","required":["t"],"definitions":{"u32":{"type":"integer","minimum":0,"maximum":4294967295}},"oneOf":[{"properties":{"t":{"const":"hello"},"version":{"type":"integer","minimum":1},"sample_rate":{"const":16000},"client_id":{"type":"string","minLength":1,"maxLength":64}},"required":["t","version","sample_rate","client_id"]},{"properties":{"t":{"const":"playback"},"epoch":{"$ref":"#/definitions/u32"},"utterance_id":{"$ref":"#/definitions/u32"},"chunk_id":{"$ref":"#/definitions/u32"},"position_ms":{"type":"integer","minimum":0,"maximum":600000},"state":{"enum":["started","finished","dropped"]}},"required":["t","epoch","utterance_id","chunk_id","position_ms","state"]},{"properties":{"t":{"const":"ui"},"mute":{"type":"boolean"},"on":{"type":"boolean"},"radio":{"type":"boolean"},"interval_s":{"type":"integer","minimum":10,"maximum":3600},"quiet_start":{"type":"string","pattern":"^[0-2][0-9]:[0-5][0-9]$"},"quiet_end":{"type":"string","pattern":"^[0-2][0-9]:[0-5][0-9]$"},"speaker":{"$ref":"#/definitions/u32"},"drama":{"type":"boolean"},"screen":{"type":"boolean"},"pc":{"type":"boolean"},"persona":{"type":"string","minLength":1,"maxLength":64}},"required":["t"]},{"properties":{"t":{"const":"snap"},"id":{"$ref":"#/definitions/u32"},"jpeg":{"type":"string","maxLength":1048576},"thumb":{"type":"string","maxLength":1024},"reason":{"type":"string","maxLength":200}},"required":["t","id"]},{"properties":{"t":{"enum":["ping","pong"]},"ts":{"type":"integer"}},"required":["t","ts"]},{"properties":{"t":{"const":"duck"},"epoch":{"$ref":"#/definitions/u32"},"gain":{"type":"number","minimum":0,"maximum":1}},"required":["t","epoch","gain"]},{"properties":{"t":{"const":"cancel"},"new_epoch":{"$ref":"#/definitions/u32"}},"required":["t","new_epoch"]},{"properties":{"t":{"const":"resume"},"epoch":{"$ref":"#/definitions/u32"}},"required":["t","epoch"]},{"properties":{"t":{"const":"state"},"state":{"enum":["idle","user_speaking","eot_check","preparing","ai_playing","interrupt_check","rest"]},"muted":{"type":"boolean"}},"required":["t","state","muted"]},{"properties":{"t":{"const":"error"},"code":{"type":"integer","minimum":4000,"maximum":5999},"message":{"type":"string","maxLength":200}},"required":["t","code","message"]}]}

_VALIDATOR = Draft7Validator(MESSAGE_SCHEMA)

# 状態値 -> 日本語名 (1 対 1)
STATE_NAMES_JA = {
    "idle": "待機",
    "user_speaking": "ユーザー発話中",
    "eot_check": "終話確認中",
    "preparing": "返答準備中",
    "ai_playing": "AI再生中",
    "interrupt_check": "割り込み確認中",
    "rest": "休止",
}


def validate_message(msg: object) -> bool:
    """スキーマに合致すれば True。"""
    return _VALIDATOR.is_valid(msg)


def _check_u32(name: str, v: int) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= U32_MAX:
        raise ValueError(f"{name} out of u32 range: {v}")


def pack_down_header(epoch: int, utterance_id: int, chunk_id: int, length: int) -> bytes:
    for n, v in (("epoch", epoch), ("utterance_id", utterance_id),
                 ("chunk_id", chunk_id), ("length", length)):
        _check_u32(n, v)
    return struct.pack("<IIII", epoch, utterance_id, chunk_id, length)


def unpack_down_header(data: bytes) -> tuple[int, int, int, int]:
    if len(data) < HEADER_SIZE:
        raise ValueError("header too short")
    return struct.unpack("<IIII", data[:HEADER_SIZE])


def pack_down_chunk(epoch: int, utterance_id: int, chunk_id: int, pcm: bytes) -> bytes:
    if len(pcm) > MAX_CHUNK_PCM:
        raise ValueError(f"chunk PCM too large: {len(pcm)}")
    return pack_down_header(epoch, utterance_id, chunk_id, len(pcm)) + pcm


def is_drama(seq: int) -> bool:
    return bool(seq & DRAMA_FLAG)


def pack_up_frame(seq: int, pcm: bytes) -> bytes:
    _check_u32("seq", seq)
    return struct.pack("<I", seq) + pcm


def unpack_up_frame(data: bytes) -> tuple[int, bytes]:
    if len(data) < 4:
        raise ValueError("frame too short")
    return struct.unpack("<I", data[:4])[0], data[4:]
