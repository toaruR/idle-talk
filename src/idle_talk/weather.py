"""話者の保存済み居住地から、たまに紹介する当日の予報を取得する。"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import random
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from .residence import residence
from .web_search import USER_AGENT

log = logging.getLogger("idle_talk.weather")
WEATHER_GAP_MS = 3 * 60 * 60 * 1000
RETRY_MS = 15 * 60 * 1000
WEATHER_CHANCE = 0.1
FETCH_TIMEOUT_S = 4.0
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_PROMPT = (
    "(利用者は黙っています。次の居住地の天気予報を自然なつぶやきとして1〜2文で短く紹介してください。"
    "返事や質問は求めません。これは記載された日付の予報であり、今の実測の天気ではありません。"
    "データにない天気・数値・警報を付け足さず、以前の会話の天気と混ぜないでください。\n{summary})"
)
_PREFECTURE = re.compile(r"^(.{2,3}?[都道府県])(.+)$")
_CODES = {
    0: "快晴", 1: "おおむね晴れ", 2: "晴れ時々曇り", 3: "曇り", 45: "霧", 48: "着氷性の霧",
    51: "弱い霧雨", 53: "霧雨", 55: "強い霧雨", 56: "弱い着氷性の霧雨", 57: "着氷性の霧雨",
    61: "弱い雨", 63: "雨", 65: "強い雨", 66: "弱い着氷性の雨", 67: "着氷性の雨",
    71: "弱い雪", 73: "雪", 75: "強い雪", 77: "雪粒", 80: "弱いにわか雨", 81: "にわか雨",
    82: "強いにわか雨", 85: "弱いにわか雪", 86: "強いにわか雪", 95: "雷雨",
    96: "ひょうを伴う雷雨", 99: "強いひょうを伴う雷雨",
}


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
        return json.loads(response.read(65536))


@dataclass(frozen=True)
class Forecast:
    summary: str
    url: str


class WeatherTopics:
    def __init__(self, fetch: Callable[[str], dict] = fetch_json,
                 rand: Callable[[], float] = random.random) -> None:
        self.fetch = fetch
        self.rand = rand
        self._after: dict[tuple[int, str], int] = {}

    def due(self, sid: int, place: str, now_ms: int) -> bool:
        key = (sid, place)
        if now_ms < self._after.get(key, 0) or self.rand() >= WEATHER_CHANCE:
            return False
        self._after[key] = now_ms + RETRY_MS
        return True

    def used(self, sid: int, place: str, now_ms: int) -> None:
        self._after[sid, place] = now_ms + WEATHER_GAP_MS

    def _location(self, place: str) -> dict | None:
        place = place.strip()
        if re.search(r"[一-龯]", place):
            place = re.sub(r"\s+", "", place)
        match = _PREFECTURE.match(place)
        prefecture, town = match.groups() if match else (None, place)
        # 「横浜」だけだと青森県の横浜町なども返るため、市の正式名を先に照合する。
        if town == "東京":
            queries = ["東京都"]
        elif re.search(r"[一-龯]", town) and not town.endswith(("都", "道", "府", "県", "市", "区", "町", "村")):
            queries = [town + "市", town]
        else:
            queries = [town]
        for query in queries:
            url = GEOCODING_URL + "?" + urllib.parse.urlencode(
                {"name": query, "count": 10, "language": "ja", "format": "json"})
            data = self.fetch(url)
            candidates = [r for r in data.get("results", [])
                          if r.get("name", "").casefold() == query.casefold()
                          and (prefecture is None or r.get("admin1") == prefecture)]
            if candidates:
                # 同名の市など、地域を一つに絞れない場合は推測しない。
                return candidates[0] if len(candidates) == 1 else None
        return None

    def forecast(self, place: str, now_ms: int) -> Forecast | None:
        """ネットワークは呼出元が別スレッドへ逃がす。失敗・不明・古い予報は使わない。"""
        try:
            location = self._location(place)
            if location is None:
                return None
            lat, lon = location["latitude"], location["longitude"]
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return None
            url = FORECAST_URL + "?" + urllib.parse.urlencode({
                "latitude": lat, "longitude": lon,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto", "forecast_days": 1, "temperature_unit": "celsius",
            })
            data = self.fetch(url)
            daily = data["daily"]
            date = daily["time"][0]
            offset = data["utc_offset_seconds"]
            if type(offset) is not int or not -86400 <= offset <= 86400:
                return None
            today = (dt.datetime.fromtimestamp(now_ms / 1000, dt.timezone.utc)
                     + dt.timedelta(seconds=offset)).date().isoformat()
            if date != today:
                return None
            code = daily["weather_code"][0]
            hi, lo, rain = (daily[k][0] for k in (
                "temperature_2m_max", "temperature_2m_min", "precipitation_probability_max"))
            if (type(code) is not int or code not in _CODES
                    or not all(type(v) in (int, float) and math.isfinite(v) for v in (hi, lo, rain))
                    or lo > hi or not 0 <= rain <= 100):
                return None
            summary = (f"{place}の{date}（現地日付）の予報（Open-Meteo）: {_CODES[code]}、"
                       f"最高{hi:g}℃、最低{lo:g}℃、一日の最大降水確率{rain:g}%。")
            return Forecast(summary, url)
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
            log.warning("weather unavailable")  # URL や居住地・記憶の本文はログに出さない
            return None
