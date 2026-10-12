# idle-talk

ローカル完結の音声雑談エージェント。ブラウザ(マイク/再生) ⇄ WebSocket ⇄ Director(Python) ⇄ llama-server(LLM) / AivisSpeech(TTS) / faster-whisper(ASR ワーカー)。設計: `docs/design-idle-talk.md`。

## 前提
- Python >= 3.13 (`pip install -e .[dev]`: pytest, jsonschema)
- llama-server: 既定 `http://127.0.0.1:8080`
- AivisSpeech: 既定 `http://127.0.0.1:10101`
- faster-whisper (`pip install faster-whisper`。未導入だと asr_worker は exit 2)
- 実機試験のみ psutil

## 起動
```
python -m idle_talk.asr_worker [--port 8090]   # 127.0.0.1 固定。モデルは起動時に1回ロード
python -m idle_talk.main [--port 8765] [--data-dir data] [--pid-file P] \
    [--llm-url ...] [--tts-url ...] [--asr-url http://127.0.0.1:8090] [--speaker ID]
```
ブラウザで `web/index.html` を開く (`?port=` で WS ポート指定、既定 8765)。二重起動は exit 4。DB は `data/idle_talk.db`。

環境変数 `LLAMA_URL` / `AIVIS_URL` / `AIVIS_SPEAKER`(既定 888753760) は `m0_bench` と `-m integration` の試験が参照する (`m0_bench` は未設定時に `idle_talk.toml` の値を使う)。`main` 本体は上記の引数と設定ファイルを使い環境変数は読まない。

## テスト
- 既定: `pytest` (addopts `-m 'not integration'`。実サービス不要)
- 実機: `pytest -m integration` (llama-server / AivisSpeech / GPU / `records/*` が必要。欠落・不合格は失敗)
- JS: `web/tests/*.test.js` (node)

## Director が events に記録する kind
- 段階計測: `eot_wait` / `asr_done` / `llm_first_clause` / `tts_first` / `play_start`
- `playback` (再生 started/finished 等。相槌クリップは除外)、`cancel` (割り込みによる再生取消)
- `echo_chain` (自声起因の応答連鎖1回=1行)
- `soak_sample` (RSS/VRAM、60秒ごと。計測失敗はログのみ)
- `run_meta` (`director.close()` 時に1行: host/started_at/ended_at/app_commit/sample_count。sample_count は run_meta 以外の行数)
- 他: `backchannel`, `backchannel_misfire`, `g2_sample`, `spontaneous_fire`, `spontaneous_cancel`

## 実機記録の作り方 (捏造禁止)
記録は実機運転で生成された DB/JSON をそのままコピーする。手編集・合成は不可 (`validate_run_meta` が出所を検査)。
1. `records/m1_events.db`: asr_worker と main を起動し、実マイク/スピーカーで20分以上会話 → 正常終了(Ctrl+C。close で run_meta が書かれる)→ `data/idle_talk.db` を `records/m1_events.db` へコピー。検証: `pytest -m integration tests/test_m1_live.py`
2. `records/soak_events.db`: 同様に8時間以上(480行以上の soak_sample)運転しコピー。検証: `tests/test_soak_live.py`
3. `records/m5_events.db`: スピーカー出力(AEC 実測)で20分以上運転、1回以上割り込み(cancel)を含めてコピー。検証: `tests/test_echo_chain_live.py`
4. `records/capture_check.json`: ブラウザで `web/capture_check.html` を localhost で開き start → ダウンロードされた `capture_check.json` を `records/` へ保存。C1-C4 が true で出所キーが揃うこと。検証: `node --test web/tests/capture_record_live.js`
各 DB は新しい `--data-dir` で運転するとログが混ざらない。

## 既知の設計上の穴・注意
- **echo_chain の検出則は最小限 (設計ギャップ)**: AI 再生中に始まった発話の認識結果から英数字・かな漢字以外(句読点・空白)を除いたものが、直近 AI 発話本文に部分文字列として含まれれば1件。閾値・類似度なし。設計書に検出則が無く、言い換えや部分認識は検出できない。M5 の「連鎖0回」判定はこの検出則に依存する。
- **`{"t":"text"}` はスキーマ外**: プロトコル(`protocol.py`)に `text` 型は無く、受信すると `validate_message` に失敗し範囲外エラーとして拒否される。AI 発話の文字表示用メッセージは現状存在しない。

## 運用フェーズでの確認事項
コードは変更しない方針。以下の live 項目はすべて運用フェーズで実施する(自動ループの対象外)。pytest の live は `-m integration` 指定時のみ実行され、skip 呼び出しはなく、前提未達・不合格は exit 非0 になる。JS の live は `*.test.js` ではないため既定の `node --test web/tests/*.test.js` には含まれない(記録が無い間は exit 1 が正)。

| ID | 前提 | コマンド | 合格条件 | 生成される記録 |
|---|---|---|---|---|
| T003 | 実 llama-server と AivisSpeech が起動、faster-whisper 利用可 | `python -m pytest tests/test_clients_live.py -m integration -q` | exit 0。LLM 最初のトークンが HTTP_TIMEOUT.llm_first(5s)以内、`[DONE]` まで llm_total(20s)以内、出力に `<think>` が 0 件、AivisSpeech の /audio_query→/synthesis が tts_sentence(3s)以内に RIFF ヘッダ付き WAV を返す、実 faster-whisper で silence_input_s(60s)の無音入力の出力件数が 0 | なし(pytest 結果のみ) |
| T004 | 同上(llama-server / AivisSpeech / faster-whisper) | `python -m pytest tests/test_m0_live.py -m integration -q` | exit 0。run_m0 を実機に対し実行して evaluate_m0 の結果が空リスト、run_m0 の出力に `<think>` が 0 件 | なし |
| T005 | ブラウザで `web/capture_check.html` を localhost で開き start、ダウンロードされた JSON を保存 | `node --test web/tests/capture_record_live.js` | exit 0。C1-C4 が全て true、host/started_at/ended_at/app_commit/frame_count が存在、ended_at > started_at、frame_count <= (ended_at-started_at)/20+1 | `records/capture_check.json` |
| T015 | 実 llama-server + 実 AivisSpeech(ASR は模擬)、psutil 導入済み。llama-server を kill するため**最後に実施** | `python -m pytest tests/test_integration_live.py -m integration -q` | exit 0。最初の合成可能な節の確定から TTS 最初の音声までの p50 が 550ms 以下(TEST_N=20 回)、T-E8: /synthesis を合成中に切断し、tts_sentence(3s)+tts_cpu_settle_s(2s)以内に TTS CPU が待機時 +tts_cpu_delta_pt(5pt)以内、T-E7: llama-server kill 後 llm_first(5s)以内に定型句へ縮退し、初回バックオフ(1s)内に再起動、次ターンで応答(`idle_talk.toml` の `[llm].model` が必要) | なし |
| T016 | asr_worker と main を起動し実マイク/スピーカーで 20 分以上会話、Ctrl+C で正常終了、`data/idle_talk.db` を records へコピー | `python -m pytest tests/test_m1_live.py -m integration -q` | exit 0。run_meta の 5 キー(host/started_at/ended_at/app_commit/sample_count)あり、ended_at-started_at >= 1200000ms、events.ts が単調非減少、sample_count が run_meta 以外の行数と一致、evaluate_gate('g3') 合格(旧 epoch 再生 0 件)、evaluate_gate('history') 合格(不整合 0) | `records/m1_events.db` |
| T022 | 8 時間以上運転(soak_sample 480 行以上)、正常終了でコピー | `python -m pytest tests/test_soak_live.py -m integration -q` | exit 0。run_meta の 5 キーあり、ended_at-started_at >= 28800000ms、soak_sample の ts が単調増加、sample_count が soak_sample 行数と一致、soak_sample が 480 行以上、analyze_soak 合格(RSS 回帰傾き <= 1.0 MiB/h) | `records/soak_events.db` |
| T023 | スピーカー出力(AEC 実測)で 20 分以上運転、割り込み(cancel)を 1 回以上含める | `python -m pytest tests/test_echo_chain_live.py -m integration -q` | exit 0。run_meta の 5 キーあり、ended_at-started_at >= 1200000ms、events.ts が単調非減少、sample_count 一致、evaluate_echo_chain 合格(自声起因の応答連鎖 0 回)、cancel が 1 件以上(割り込み可能) | `records/m5_events.db` |

実施順: T003 → T004 → T005 → T016 → T023 → T022 → T015(T015 は llama-server を kill するため最後)。T016/T022/T023 の記録は新しい `--data-dir` で運転するとログが混ざらない。

- **soak 間隔検査のリスク**(設計ギャップ): `validate_soak` は `interval_ms=60000` を指定し、`validate_run_meta` は run_meta 以外の**全行**(playback, cancel など他 kind も含む)の隣接 ts 間隔を検査する。soak_sample のみの間隔ではない。soak 運転中に会話イベントが混ざると行が増えて間隔は縮むだけだが、60秒超の空きがあれば(計測失敗、停止など)不合格になる。
  - 運転中に見る点: (1) 隣接 ts の最大間隔(60000ms を超えるか)、(2) soak_sample 以外の行(playback, cancel など)が混ざる頻度、(3) スレッド間での ts 逆行の有無。
  - 運転で `validate_soak` / `test_soak_live.py` がこの理由で失敗した場合の決定事項: 間隔検査を soak_sample 行のみに限定するか、許容幅を設計定数として定義するか(設計書への追記が先。数値は設計に無いため現状は未定義)。

## 計画との対応

`docs/plans/plan-idle-talk.json` のタスクと主な導入物。

- T001 雛形・設定・メッセージ定義: `pyproject.toml`, `config.py`, `errors.py`, `protocol.py`
- T002 遅延予算・合否ゲート: `budget.py`
- T003 外部サービスクライアント・モック: `clients.py`, `asr_worker.py`, `tests/mocks.py`
- T004 M0 計測ハーネス: `m0_bench.py`
- T005 AudioWorklet 取り込み確認: `web/worklet.js`, `web/capture_check.{js,html}`
- T006 M0 判定ルール: `m0_rules.py`
- T007 SQLite ストア: `store.py`
- T008 epoch 管理・状態機械: `epoch.py`, `fsm.py`
- T009 割り込み判定・履歴確定: `interrupt.py`, `history.py`
- T010 WebSocket サーバ: `server.py`, `portlock.py`
- T011 音声入力経路: `vad.py`, `ingress.py`, `asr_worker.py`
- T012 ブラウザクライアント: `web/index.html`, `web/player.js`, `web/client.js`
- T013 応答パイプライン: `pipeline.py`, `clause.py`, `llm_supervisor.py`
- T014 Director 結線・起動: `director.py`, `main.py`
- T015 M1 結合試験: `tests/test_integration_*.py`
- T016 M1 計測・合否レポート: `metrics.py`
- T017 終話判定・G4 比較: `eot.py`
- T018 AI 相槌: `backchannel.py`
- T019 話題カード・記憶: `topics.py`, `memory.py`
- T020 自発発話・無操作休止: `spontaneous.py`
- T021 RSS Topic Worker: `topic_worker.py`
- T022 常駐試験ハーネス: `resource_sampler.py`, `main.py`(soak_loop), `tests/test_soak*.py`
- T023 AEC 実測プローブ: `web/aec_probe.js`, `echo_chain.py`
- T024 events 記録結線の配線試験: `director.py`(playback/cancel/echo_chain 記録), `tests/test_wiring_records.py`, `web/tests/capture_record_live.js`

(Python は `src/idle_talk/` 配下。)

実機が必要な確認項目 T003 / T004 / T005 (`capture_record_live.js`) / T015 / T016 / T022 / T023 は運用フェーズの作業です。手順と合格条件は「運用フェーズでの確認事項」を参照してください。
