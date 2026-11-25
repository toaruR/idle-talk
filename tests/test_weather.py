"""居住地の選別、予報の鮮度、頻度制限。HTTP は固定応答で試験する。"""
import asyncio
import copy
import datetime as dt
import urllib.parse
import threading

import pytest

from idle_talk import memory
from idle_talk.store import Store
from idle_talk.weather import RETRY_MS, WEATHER_GAP_MS, WeatherTopics, residence
from idle_talk.voice_id import VoiceBook
from idle_talk import topics
from test_spontaneous import make, until

NOW = int(dt.datetime(2026, 11, 25, tzinfo=dt.timezone.utc).timestamp() * 1000)
LOCATION = {"name": "横浜市", "admin1": "神奈川県", "latitude": 35.43, "longitude": 139.65}
DAILY = {"utc_offset_seconds": 32400, "daily": {"time": ["2026-11-25"], "weather_code": [3],
         "temperature_2m_max": [12.7], "temperature_2m_min": [8.0], "precipitation_probability_max": [82]}}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "weather.db", db_backup=False)
    for name in ("太郎", "花子"):
        s.execute("INSERT INTO speakers(name, embedding, n, updated_at) VALUES (?, ?, 1, 0)", (name, b""))
    yield s
    s.close()


@pytest.mark.parametrize("text,expected", [
    ("利用者は横浜に住んでいる。", "横浜"), ("利用者は神奈川県横浜市在住です。", "神奈川県横浜市"),
    ("利用者の居住地は横浜市です。", "横浜市"), ("利用者は今は東京都に暮らしている。", "東京都"),
    ("利用者はNew Yorkに住んでいる。", "New York"),
    ("利用者は横浜出身です。", None), ("利用者は横浜に旅行する。", None),
    ("利用者は以前横浜に住んでいた。", None), ("利用者は以前横浜に住んでいる。", None),
    ("利用者は横浜に住んでいない。", None), ("利用者は猫好きです。", None),
    ("利用者は友人が横浜に住んでいる。", None), ("利用者は横浜市1丁目に住んでいる。", None),
])
def test_only_current_residence_is_used(store, text, expected):
    memory.add_memory(store, text, 1, 1, persona="a", speaker_id=1)
    assert residence(store, "a", 1) == expected


def test_residence_is_scoped_to_speaker_and_persona_and_uses_latest(store):
    memory.add_memory(store, "利用者は大阪に住んでいる。", 1, 1, speaker_id=1)
    memory.add_memory(store, "利用者は横浜に住んでいる。", 2, 2, persona="a", speaker_id=1)
    memory.add_memory(store, "利用者は京都に住んでいる。", 3, 3, persona="b", speaker_id=1)
    memory.add_memory(store, "利用者は名古屋に住んでいる。", 4, 4, persona="a", speaker_id=2)
    assert residence(store, "a", 1) == "横浜"
    assert residence(store, "b", 1) == "京都"
    assert residence(store, "a", 2) == "名古屋"
    assert residence(store, "a", None) is None
    memory.add_memory(store, "利用者は横浜に住んでいない。", 5, 5, persona="a", speaker_id=1)
    assert residence(store, "a", 1) is None


def fetcher(calls, daily=None, locations=None):
    def fetch(url):
        calls.append(url)
        if "geocoding-api" in url:
            return {"results": copy.deepcopy([LOCATION] if locations is None else locations)}
        return copy.deepcopy(DAILY if daily is None else daily)
    return fetch


def test_forecast_contains_only_current_data_and_requests_local_date():
    calls = []
    result = WeatherTopics(fetcher(calls)).forecast("神奈川県横浜市", NOW)
    assert result is not None
    assert "曇り" in result.summary and "最高12.7℃" in result.summary and "最低8℃" in result.summary
    assert "82%" in result.summary and "2026-11-25" in result.summary
    assert "Open-Meteo" in result.summary
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[0]).query)
    assert query["name"] == ["横浜市"] and "神奈川県" not in calls[0]
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(result.url).query)
    assert query["timezone"] == ["auto"] and query["forecast_days"] == ["1"]


def test_bare_japanese_city_uses_formal_city_name_before_same_named_town():
    calls = []
    assert WeatherTopics(fetcher(calls)).forecast("横浜", NOW) is not None
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(calls[0]).query)["name"] == ["横浜市"]


@pytest.mark.parametrize("locations", [[], [dict(LOCATION, admin1="青森県")],
    [LOCATION, dict(LOCATION, latitude=34)]])
def test_unknown_wrong_prefecture_or_ambiguous_location_skips_forecast(locations):
    calls = []
    assert WeatherTopics(fetcher(calls, locations=locations)).forecast("神奈川県横浜市", NOW) is None
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [
    ("time", ["2026-11-24"]), ("time", []), ("weather_code", [999]), ("weather_code", [True]),
    ("temperature_2m_max", [None]), ("temperature_2m_min", [float("nan")]),
    ("temperature_2m_min", [30]), ("precipitation_probability_max", [101]),
])
def test_stale_missing_or_malformed_forecast_is_not_used(field, value):
    daily = copy.deepcopy(DAILY)
    daily["daily"][field] = value
    assert WeatherTopics(fetcher([], daily=daily)).forecast("横浜", NOW) is None


def test_forecast_uses_forecast_timezone_at_utc_day_boundary():
    daily = copy.deepcopy(DAILY)
    daily["utc_offset_seconds"] = -18000
    daily["daily"]["time"] = ["2026-11-24"]
    assert WeatherTopics(fetcher([], daily=daily)).forecast("横浜", NOW) is not None


def test_fetch_failure_returns_none_without_location_in_log(caplog):
    def fail(url):
        raise OSError(url)
    assert WeatherTopics(fail).forecast("神奈川県横浜市", NOW) is None
    assert "weather unavailable" in caplog.text and "横浜" not in caplog.text


def test_ten_percent_chance_and_per_speaker_retry_and_success_gaps():
    draws = iter([0.1, 0.099, 0, 0, 0])
    provider = WeatherTopics(rand=lambda: next(draws))
    assert not provider.due(1, "横浜", NOW)
    assert provider.due(1, "横浜", NOW)
    assert not provider.due(1, "横浜", NOW + RETRY_MS - 1)
    assert provider.due(1, "横浜", NOW + RETRY_MS)
    provider.used(1, "横浜", NOW + RETRY_MS)
    assert not provider.due(1, "横浜", NOW + RETRY_MS + WEATHER_GAP_MS - 1)
    assert provider.due(2, "横浜", NOW + RETRY_MS)
    assert provider.due(1, "横浜", NOW + RETRY_MS + WEATHER_GAP_MS)


def director(tmp_path):
    d, browser, store = make(tmp_path)
    sid = store.execute("INSERT INTO speakers(name, embedding, n, updated_at) VALUES ('太郎', ?, 1, 0)", (b"",))
    d.voices = VoiceBook(store)
    d._speaker = sid
    d.residence_questions.rand = lambda: 1  # 天気とニュースの選択だけを検証する。
    d.now_ms = lambda: NOW
    memory.add_memory(store, "利用者は横浜に住んでいる。", 1, 1,
                      persona=d.personas.current_id, speaker_id=sid)
    calls, seen = [], []
    d.weather = WeatherTopics(fetcher(calls), rand=lambda: 0)
    d.pipeline._llm = lambda messages: seen.append(messages) or iter(["横浜は曇りの予報だね。"])
    d.core.on_text(browser, '{"t":"ui","on":true}')
    return d, browser, store, calls, seen


def test_spontaneous_weather_does_not_consume_news_and_next_turn_returns_to_news(tmp_path):
    async def run():
        d, b, store, calls, seen = director(tmp_path)
        cid = topics.add_card(store, "新しい駅が開業", expires_at=NOW + 10000)
        try:
            d.clock_.advance(1200)
            assert d.spontaneous_tick() == "spontaneous"
            await until(lambda: not d.spontaneous.pending)
            prompt = seen[0][-1]["content"]
            assert "横浜" in prompt and "12.7℃" in prompt and "返事や質問は求めません" in prompt
            assert b.of("links")[-1]["items"][0]["url"].startswith("https://api.open-meteo.com/")
            assert store.query("SELECT used_at FROM topic_cards WHERE id=?", (cid,))[0][0] is None
            d.spontaneous.on_user_speech()
            d._last_op = d.clock_()
            d.clock_.advance(2400)
            d._last_op = d.clock_()
            assert d.spontaneous_tick() == "spontaneous"
            await until(lambda: not d.spontaneous.pending)
            assert "新しい駅が開業" in seen[-1][-1]["content"]
            assert len(calls) == 2
        finally:
            d.close()
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize("reason", ["disabled", "unknown", "no_residence", "chance", "failed"])
def test_weather_unavailable_uses_normal_spontaneous_topic(tmp_path, reason):
    async def run():
        d, _, store, calls, seen = director(tmp_path)
        topics.add_card(store, "新しい駅が開業", expires_at=NOW + 10000)
        if reason == "disabled":
            d.weather = None
        elif reason == "unknown":
            d._speaker = None
        elif reason == "no_residence":
            store.execute("DELETE FROM memories")
        elif reason == "chance":
            d.weather.rand = lambda: 0.1
        else:
            d.weather.fetch = lambda url: {}
        try:
            d.clock_.advance(1200)
            assert d.spontaneous_tick() == "spontaneous"
            await until(lambda: not d.spontaneous.pending)
            assert "新しい駅が開業" in seen[0][-1]["content"]
            assert calls == []
        finally:
            d.close()
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["cancel", "speaker", "persona", "residence"])
def test_weather_discarded_when_listener_changes_during_fetch(tmp_path, mutation):
    async def run():
        d, _, store, calls, seen = director(tmp_path)
        started, release = threading.Event(), threading.Event()

        def slow_fetch(url):
            started.set()
            assert release.wait(5)
            return fetcher(calls)(url)

        d.weather.fetch = slow_fetch
        try:
            d.clock_.advance(1200)
            assert d.spontaneous_tick() == "spontaneous"
            await until(started.is_set)
            if mutation == "cancel":
                d._cancel_spontaneous()
            elif mutation == "speaker":
                d._speaker = None
            elif mutation == "persona":
                d.personas.current_id = "other"
            else:
                store.execute("DELETE FROM memories")
            release.set()
            await until(lambda: d._spont_task.done())
            if mutation == "cancel":
                assert seen == []  # 生成・再生へ進まない。
            else:
                assert "現地日付" not in seen[0][-1]["content"]
            assert d.weather._after[1, "横浜"] == NOW + RETRY_MS
        finally:
            release.set()
            d.close()
            store.close()
    asyncio.run(run())
