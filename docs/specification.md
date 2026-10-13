<!-- spec-doc:last-reviewed-commit=de4bec873e53ef63ddbbf6bfe3fab6f0387fe665 reviewed-at=2026-10-12 -->

# 仕様書

<!--
このファイルは /spec-doc コマンドが git 差分から自動で追記・更新する。
- 先頭のマーカー行は /spec-doc が「前回どのコミットまで確認したか」を記録するためのもの。
  手で編集しない（初回はマーカーなしのままでもよく、その場合 /spec-doc が初回モードで動く）
- 構成: 対象（機能 / モジュール / API / 設定 など）ごとに節を分ける
- 実装手順ではなく「何が・どう振る舞うか」（仕様・挙動・設計意図）を書く
-->

設計の根拠は `docs/design-idle-talk.md`、タスク分解は `docs/plans/plan-idle-talk.json`、起動手順と運用フェーズの実機確認は `README.md` を参照。本書は実装 (`src/idle_talk/`, `web/`) が現在どう振る舞うかをまとめる。

## 全体構成

- 利用者 1 人・Windows 11・ローカル完結の常駐型日本語音声雑談アプリ。
- 経路: ブラウザ (マイク取り込み / 再生) ⇄ WebSocket ⇄ Director (Python, 単一イベントループ) ⇄ llama-server (LLM) / AivisSpeech (TTS) / ASR ワーカー (faster-whisper)。
- ネットワークに出るのは RSS Topic Worker のみ。他はすべて 127.0.0.1。

## プロセスと起動

| プロセス | 起動 | 既定ポート | 備考 |
|---|---|---|---|
| Director | `python -m idle_talk.main` | WS 8765 / Web 8000 | `--web-port`、`--data-dir`(既定 `data`)、`--pid-file`、`--llm-url` / `--tts-url` / `--asr-url` |
| ASR ワーカー | `python -m idle_talk.asr_worker` | 8090 (`--port`) | 127.0.0.1 固定。faster-whisper `small` を CUDA/float16 で起動時に 1 回ロード。未導入なら exit 2 |
| llama-server | `[llm].model` 設定時は Director が起動・監督、未設定なら手動 | 8080 | 起動引数は `llm_supervisor.build_llama_argv`(`--jinja` + `enable_thinking=false` で思考無効) |
| AivisSpeech | 手動 | 10101 | 話者 ID 既定 888753760 (`idle_talk.toml` の `[tts].speaker` / `--speaker`) |

- 二重起動は pid ロックで検出し exit 4。`asr_worker` も `--pid-file` 指定時は同じ検査をする。
- `python -m idle_talk.ctl start|stop|restart|status` (Windows は `idle-talk.cmd`) が ASR ワーカーと Director を端末から切り離して起動する。PID ファイル (`data/idle_talk.pid` / `data/asr_worker.pid`) の生存確認で二重起動を拒否し (exit 4)、出力は `data/idle_talk.log` / `data/asr_worker.log` に追記する。Director のログは INFO 以上を `日時 レベル ロガー名: 本文` で出す。
- 発話文言は `data/talk.log` (UTF-8) に `日時 USER: …` / `日時 AI: …` で追記する (ロガー `idle_talk.talk`、他のログには流さない)。利用者は ASR 確定時、AI は DB の turns と同じく再生を終えたチャンクだけを確定時に書き、途中で止まった発話は `AI(中断):` とする。
- 停止: `ctl stop` が `data/idle_talk.stop` を置き、Director は `main.stop_watch` (0.5 秒周期) で検知して Ctrl+C と同じ終了処理をする。`ctl.STOP_TIMEOUT_S`=20 秒以内に終わらなければ強制終了する (このとき llama-server が残ることがある)。ASR ワーカーは即時に強制終了する。Director は起動時に残っていた停止ファイルを消す。
- `main` は CLI 引数と `idle_talk.toml` のみを使い、環境変数 `LLAMA_URL` / `AIVIS_URL` / `AIVIS_SPEAKER` は M0 計測と実機試験だけが読む。
- 終了時 (Ctrl+C) に `director.close()` が `run_meta` を 1 行書いてからストアを閉じる。
- 起動中は 60 秒ごとに RSS / VRAM を `soak_sample` として記録する (計測失敗はログのみで継続)。
- ASR ワーカーのポート 8090 は設計書に定義が無い実装上の既定値 (要確認)。

### ポートと話者の設定 (`idle_talk.toml`)

- 各ポートは共通設定ファイル `idle_talk.toml` の `[ports]` で変更する。キーは `ws` (Director の WS)、`asr`、`llm`、`tts`、`web` (`web/` の静的配信) で、既定値は上表と同じ。`config.load_ports` が読み、`main` / `asr_worker` / M0 計測が使う。
- 優先順位は CLI 引数 (`--port` / `--web-port` / `--*-url`)、設定ファイル、既定値の順。M0 計測では環境変数 `LLAMA_URL` / `AIVIS_URL` が設定ファイルより優先する。
- 設定ファイルは起動時のカレントディレクトリから読む。リポジトリ直下以外で起動する場合は `--config <path>` で場所を指定する。ファイルが無ければ既定値で動く。
- 未知のキーや、0〜65535 の整数でない値は起動時に `ValueError` になる。
- `main` は `[ports].web` で `web/` を 127.0.0.1 に配信し (`main.start_web`、標準ライブラリ `http.server`)、`/config.json` で `{"ws_port": <ws>}` を返す。`index.html` はこれを読んで接続するため、ポート番号を持たない。`?port=` 指定があればそちらを使う。
- AivisSpeech と、監督しない llama-server は手動起動のため、`llm` / `tts` を変えたら各サーバ自体の待受ポートも合わせる。Director が起動する llama-server には `[ports].llm` (または `--llm-url` のポート) が渡される。
- 話者 ID は `[tts]` の `speaker` で指定する。AivisSpeech が声 (モデル) とスタイルの組ごとに振る整数で、`/audio_query` と `/synthesis` の `speaker` パラメータに渡す。既定は 888753760 (`config.DEFAULT_SPEAKER`)。
- 話者 ID の優先順位は `--speaker`、設定ファイル、既定値の順。M0 計測では環境変数 `AIVIS_SPEAKER` が設定ファイルより優先する。負数・非整数・`[tts]` の未知キーは `ValueError`。
- 声パラメータは `[tts]` の `speed` / `intonation` / `tempo_dynamics` / `pitch` / `volume` / `pause` / `pre_silence` / `post_silence` で指定する (`config.VOICE_PARAMS`、`config.load_voice`)。それぞれ AudioQuery の `speedScale` / `intonationScale` / `tempoDynamicsScale` / `pitchScale` / `volumeScale` / `pauseLengthScale` / `prePhonemeLength` / `postPhonemeLength` に対応する。
- `TtsClient` は `/audio_query` の結果を指定キーだけ上書きして `/synthesis` に渡す。未指定のキーは AivisSpeech の既定値のまま。出力は `clients.TTS_RATE`=24000Hz モノラルに固定する (`web/player.js` の再生レートと一致させる)。
- `[tts]` は起動後も反映される。`main.TtsSettingsReloader` が合成のたびに設定ファイルの更新時刻を見て、変わっていれば話者と声パラメータを読み直す (次の発話から効く)。不正な値は警告して前の値を保つ。`--speaker` 指定時は話者を上書きしない。起動時に合成済みの相槌クリップは再起動まで前の声のまま。他のセクション (`[ports]` / `[spontaneous]` / `[llm]`) は起動時のみ読む。
- 範囲は AivisSpeech Engine 1.2.0 の openapi の記載に従う (speed 0.5〜2.0、intonation / tempo_dynamics / volume 0〜2、pitch -0.15〜0.15、pause と無音長は 0 以上)。範囲外・非数値は `ValueError`。
- 使える ID は AivisSpeech の `GET /speakers` で確認する。インストールされていない ID を指定すると、合成のたびに AivisSpeech がエラーを返す (起動時には検査しない)。

## WebSocket プロトコル (設計書 §12.1)

### 接続制御
- バインドは 127.0.0.1 のみ (それ以外の指定は起動時に例外)。
- Origin のホストが `localhost` / `127.0.0.1` / `::1` 以外なら close 4003。Origin 無しも拒否。
- 接続は常に 1 本。新しい接続が来たら旧接続を close 4001 で閉じる (後勝ち)。2 回目以降の接続では epoch を +1 して `cancel` を送り、旧音声を無効化する。
- `hello` の version が 1 以外、または hello がスキーマ違反なら close 4002。
- サーバは `PING_INTERVAL`(5s) ごとに `ping` を送り、pong 未応答が 2 回続くと切断扱い (close 1001)。

### テキストメッセージ
JSON Schema (`protocol.MESSAGE_SCHEMA`) で検証し、違反・不正 JSON は `error` 5006 を返して破棄する。

| t | 方向 | 主なフィールド |
|---|---|---|
| `hello` | B→S | `version`(=1), `sample_rate`(=16000), `client_id`(1〜64 文字) |
| `playback` | B→S | `epoch`, `utterance_id`, `chunk_id`, `position_ms`(0〜600000), `state`(started/finished/dropped) |
| `ui` | B→S | `mute`, `on`, `radio`, `interval_s`(10〜3600 秒), `quiet_start` / `quiet_end`(HH:MM) |
| `ping` / `pong` | 双方向 | `ts` |
| `duck` | S→B | `epoch`, `gain`(0〜1) |
| `cancel` | S→B | `new_epoch` |
| `resume` | S→B | `epoch` |
| `state` | S→B | `state`(状態名), `muted` |
| `error` | S→B | `code`(4000〜5999), `message`(200 字以内) |

- 現行 epoch と異なる `playback` は黙って破棄する。
- `{"t":"text"}` はスキーマに無く、受信すると 5006 で拒否される。AI 発話の文字表示用メッセージは存在しない (要確認: 設計 §12.1 への追加要否)。

### バイナリフレーム
- 上り (B→S): 先頭 u32 LE 連番 + 16kHz mono s16le PCM。フレーム長 20〜40ms。1MiB 超・4 byte 未満は破棄。
- 下り (S→B): 16 byte ヘッダ `<epoch, utterance_id, chunk_id, length>`(u32 LE ×4) + PCM。1 チャンク最大 1MiB。

## エラーコード (設計書 §12.2)

| コード | 種別 | 意味 |
|---|---|---|
| 4001 | WS close | 接続の置き換え (設計書の表では 4002 と意味欄が併記されている。要確認) |
| 4002 | WS close | version 不一致 / 不正 hello |
| 4003 | WS close | Origin 不許可 |
| 5001 | error | LLM タイムアウト・停止 |
| 5002 | error | TTS 失敗・タイムアウト |
| 5003 | error | ASR 失敗 |
| 5004 | error | DB 異常 |
| 5005 | error | マイクデバイス喪失 |
| 5006 | error | ui / playback の範囲外値、不正メッセージ |
| exit 4 | プロセス | 二重起動 |

## 状態機械 (設計書 §4)

状態: `idle` / `user_speaking` / `eot_check` / `preparing` / `ai_playing` / `interrupt_check` / `rest`。遷移は副作用の無い純関数 `fsm.transition` で、Director がキューで直列に適用し、遷移ごとに `state` メッセージを 1 件送る。

- `idle` → `user_speaking`: VAD 開始 (ミュート中は無視)。ASR を開始し、バックグラウンド処理を中断する。
- `user_speaking` → `eot_check`: 区間長が `MIN_SEG`(200ms) 以上で VAD 終了。未満なら雑音として破棄し `idle` へ。
- `eot_check` → `preparing`: 無音が `T_silence`(400ms) 以上かつ終話判定が真。ASR を確定し LLM を呼ぶ。途中で発話再開なら `user_speaking` へ戻る。
- `preparing` → `ai_playing`: 現行 epoch の最初の合成可能な節が確定。生成中に利用者が話し始めたら生成を取り消す。
- `ai_playing`: 利用者発話を検知すると、停止語なら即 cancel、それ以外は `duck`(gain `DUCK_GAIN`=0.2) して `interrupt_check` へ。再生完了で履歴を確定し `idle` へ。
- `interrupt_check`: 確認窓の判定が相槌なら `resume` して `ai_playing` へ、それ以外は cancel して `user_speaking` へ。
- 任意の状態 → `rest`: ミュート、無操作 `IDLE_TIMEOUT`(1800s)、連続失敗。生成・再生中なら epoch を +1 して cancel。`rest` からは wake で `idle` に戻る。

## epoch と取消 (設計書 §3.2)

- epoch は u32 の単調増加カウンタで、取消が確定したときだけ +1 する (U32 最大値の次は 0)。
- 旧 epoch の音声チャンク・再生通知・完了通知はすべて破棄する。取消と完了が同時に起きても、旧 epoch の完了は履歴に確定しない。
- 取消は LLM / ASR の完了を待たない。通常応答・相槌・自発発話の再生と取消は、同じ経路 (`ResponsePipeline.play_utterance` / `cancel`) だけを通る。

## 割り込み判定 (設計書 §5.1)

- 停止語 `STOP_WORDS`(待って / 違う / ちょっと / ストップ): NFKC 正規化後に一致するか、含んでいて長さが停止語 +2 文字以内なら停止語とみなす。duck を経ずに cancel する。
- 相槌語 (うん / うんうん / へえ / ふーん / ええ / はい / なるほど / そう / そっか) が確認窓 `[開始, 開始 + T_confirm)`(T_confirm=400ms) の内側なら相槌として再生を再開する。
- 疑問形 (末尾が ? / ？)、空、判断不能は相槌として扱わず、取り消したまま停止する。
- 強い発話: 確認窓の終わりまで続くか、AI の平均音量の `STRONG_UTT.volume_ratio`(2.0) 倍以上。

## 履歴 (設計書 §5.2)

- AI 発話の履歴には、ブラウザが `finished` を返したチャンクの本文だけを確定する。
- 割り込まれた発話は `completed=0` とし、停止位置 `stop_position_ms` を記録する。
- LLM に渡す履歴は直近 `N_TURNS`(8) ターンで、切り詰めたときはログを 1 件残す。

## 記憶 (設計書 §3.4)

- 応答の再生が完了して IDLE に戻ったとき、未処理の利用者発話 (直近 `MEM_BATCH`=4 件、`MIN_UTTERANCE_CHARS`=6 文字未満は対象外) を LLM に渡し、本人が明言した好み・習慣・予定・続いている出来事を「利用者は〜」の 1 文で抜き出させる。
- 「利用者」で始まる行だけを採用し、1 回 `MAX_PER_UTTERANCE`=3 件、1 件 `MAX_CHARS`=60 文字まで。句読点・空白を除いて同じ内容は更新時刻だけ進める。根拠は最後の発話の utterance_id。
- 保存は `MAX_MEMORIES`=30 件まで。超えたら更新時刻が古い `deletable=1` の行から消す。
- 抽出中に利用者が話し始めた・休止に入ったときは取り消して LLM の受信を打ち切り、未処理分は次の応答完了後に再試行する。抽出中は自発発話を発火しない (LLM を取り合わない)。抽出の失敗はログだけ残して会話を続ける。
- 通常応答と自発発話のシステムプロンプトに、更新が新しい順の記憶を箇条で添える (関係するときだけ触れ、無いことを覚えているふりはしない、と指示する)。
- `idle-talk memories` で一覧、`--delete ID` / `--clear` で削除 (`deletable=1` のみ)。

## 応答パイプライン (設計書 §3.3 / §4.3)

- LLM はストリーミングで受け、句点・読点などで最初の合成可能な節が確定した時点で TTS に渡す。TTS は文単位で直列に合成する。
- タイムアウト `HTTP_TIMEOUT`:
  - LLM: 最初のトークン 5s、全体 20s → 5001。
  - TTS: 1 文 3s → 5002。その文は飛ばして次の文へ続ける。
  - ASR: 区間長 ×2 + 2s → 5003。その発話は破棄する。ソケットのタイムアウト・接続断も `AsrTimeout` / `AsrFailed` に変換し、Ingress がどの失敗も握って発話だけ捨てる (WS の接続は保つ)。
- ASR は Director の単一ワーカースレッド (`director._offload_asr`) で区間順に実行し、結果はイベントループに戻して処理する。認識中も再生通知・音声送出・LLM/TTS は止まらない。後続区間の認識待ちがあるときは、終話 (SILENCE) は最後の区間の結果で確定する。
- 割り込みの認識が `T_confirm` を超えて返ると、先に判断不能として取り消され `user_speaking` に入る。その後に届いた認識結果は終話確認へ進めて応答し、認識が捨てられた (失敗・幻聴) ときは `idle` へ戻す。
- LLM が停止したときは定型句に縮退する。llama-server は `LLM_RETRY` に従い、1s → 2s → 4s のバックオフで 600s 窓内に最大 3 回まで再起動する。4 回目は再起動せず、Director を休止 (`rest`) に移して警告ログを出す。
- 再起動が働くのは Director が llama-server を起動した場合だけ。`idle_talk.toml` の `[llm]` (`server` 実行ファイル、`model` gguf パス、`ngl`、`ctx`) で `model` を指定すると、`main` が起動時に llama-server を起動し、`/health` が 200 になるまで最大 `LLM_READY_TIMEOUT_S` (120s) 待ってから続ける。出力は `<data-dir>/llama-server.log` に追記する。
- 監督は `llm_supervisor.watch` が `WATCH_INTERVAL_S` (0.5s) ごとにプロセス終了を調べ、終了していれば `LlmSupervisor.on_crash` (バックオフ + 再起動) を呼ぶ。Director 終了時は監督中の llama-server も止める。
- `[llm].model` が空、または起動時に既に同じポートで llama-server が応答している場合は、起動も監督もしない (二重起動を避けるため。後者は警告ログを出す)。この場合 llama-server が落ちても再起動されない。
- AivisSpeech には `/audio_query` → `/synthesis` の順で要求する。`/cancellable_synthesis` は AivisSpeech Engine 1.2.0 で 501 Not Implemented のため使わない。`/synthesis` は切断後も計算が続くので、中止した文の結果は epoch 不一致で破棄し、残る計算は合成中の 1 文ぶん (`tts_sentence` 以内) になる。
- Director は起動時に `TtsClient.warmup` で 1 回空合成してから待ち受ける。初回合成は GPU (DirectML) 初期化で約 14 秒かかり `tts_sentence` (3s) を超えるため。タイムアウトは `WARMUP_TIMEOUT_S` (60s)。失敗は警告ログのみで起動は続ける。
- LLM の SSE チャンクは `delta.content` が null でも受け付け、読み飛ばす (llama-server は最初のチャンクで null を送る)。
- ASR の幻聴除外: `no_speech_prob` > 0.6、または `compression_ratio` > 2.4 のセグメントは捨てる (境界値ちょうどは採用)。

## 終話判定・相槌・自発発話

- 終話判定: 既定は固定無音 (`FixedSilence`)。Smart Turn v3.2 相当のアダプタを差し替えられ、早切り件数で G4 を比べる。
- AI 相槌 (§6.1):
  - 内容に依存しない、事前合成したクリップを使う。
  - 無音が `SIL_BACKCHANNEL`(300ms) 続いたら、1 発話につき最大 1 回再生し、間隔を `BC_COOLDOWN`(5s) 空ける。
  - 無音が `T_silence` に達した時点で未完了なら取り消し、`backchannel_misfire` として記録する。
  - 相槌は G1 の遅延集計から除外する。
- 自発発話:
  - 既定はオフで、雑談モードのときだけ動く。`idle_talk.toml` の `[spontaneous]` (`on` / `interval_s` / `quiet_start` / `quiet_end` / `radio`、`config.load_spontaneous`) で起動時の値を指定する。画面の設定欄の初期値は `/config.json` の `ui` で同じ値が渡り、接続時にページが `ui` メッセージで送り返す。
  - IDLE の沈黙が `interval_s` 秒 (既定 1200、10〜3600) 続いたら話しかける。沈黙は IDLE になった時刻 (応答の再生終了、自発発話の再生終了) から測る。`main.spontaneous_loop` が `SPONT_TICK_S`=1 秒ごとに `director.spontaneous_tick` を呼ぶ (無操作休止 `IDLE_TIMEOUT` の判定も同じ呼び出し)。自発発話の指示には `director.SPONT_TOPICS` から話題の種を 1 つ無作為に添え、前の会話と違う話題を振らせる。応答の生成上限は `clients.LLM_MAX_TOKENS`=300 (2〜4 文程度)。
  - 休止時間帯 (既定 22:00〜07:00) は発話しない。
  - 無反応が続くたびに待ち時間を 2 倍に延ばす (30 秒なら 30, 60, 120…)。利用者が話すと基本値に戻る。沈黙を同意とはみなさない。
  - 利用者の操作 (発話・応答・設定変更) が `IDLE_TIMEOUT`=1800 秒無いと休止 (`rest`) に入り、自発発話も止まる。ミュートなしの `ui` (画面の「開始」による接続、設定変更) で待機に戻る (無操作・連続失敗の休止も同じ)。
  - 準備中に利用者が話し始めたら取り消す。
- ラジオモード (`radio`、既定オフ。画面の「ラジオ」チェック): 返事を期待しない独り語り。
  - `on` がオフでも動く。1 本話し終えてから `spontaneous.RADIO_GAP_S`=3 秒の沈黙で次を話す (`interval_s` と無反応による倍増は使わない)。
  - 指示は `director.RADIO_PROMPT` (ラジオのパーソナリティとして 3〜5 文、直前の続きか `SPONT_TOPICS` の話題、質問・呼びかけをしない)。
  - 記憶抽出中でも待たずに発火する (llama-server は 4 スロットで並行処理する)。
  - 聞き流し前提なので無操作休止 (`IDLE_TIMEOUT`) に入らない。休止時間帯は守る。利用者が話せば通常どおり取り消して応答し、応答後に再開する。
- 質問率 `RATIO_Q`: 3 応答に 1 回以下。

## データモデル (SQLite, 設計書 §3.4)

DB は `<data-dir>/idle_talk.db`。WAL / synchronous=NORMAL で、書き込みは専用スレッドで直列化する。

| テーブル | 列 |
|---|---|
| `events` | id, ts, epoch, kind, payload_json |
| `turns` | id, ts, role, text, completed(0/1), stop_position_ms, utterance_id |
| `topic_cards` | id, hook, facts, source_url, published_at, expires_at, angle, used_at, user_reaction |
| `memories` | id, content, evidence_utterance_id, updated_at, deletable |

- 起動時に `integrity_check` を行う。正常なら `.bak` に 1 世代だけバックアップする。破損時は `.bak` から復元し、`.bak` が無ければ破損ファイルを `.corrupt` に退避して新規作成する (5004 を記録)。
- 記憶は、根拠となる発話が無いと保存を拒否する。個別に削除できるのは `deletable=1` の行だけ。
- 話題カードは RSS Topic Worker だけが取得する。停止中は一切ネットワークに接続しない。

### events の kind
| kind | payload | 用途 |
|---|---|---|
| `eot_wait` / `asr_done` / `llm_first_clause` / `tts_first` / `play_start` | `{turn, gpu1_load}` | 段階計測。1 ターン 5 行。G1 = play_start − eot_wait |
| `playback` | `{epoch, utterance_id, chunk_id, state, position_ms}` | G3・履歴整合 (相槌クリップは除外) |
| `cancel` | — | 割り込みによる取消 1 回につき 1 行 (`rest` への遷移による取消は含まない) |
| `echo_chain` | — | 自声起因の応答連鎖 1 回につき 1 行 |
| `g2_sample` | `{latency_ms}` | 発話開始 → duck / 停止開始 |
| `soak_sample` | `{rss_mib, vram_mib}` | 常駐試験 |
| `backchannel` / `backchannel_misfire` | — | 相槌の再生 / 誤発火 |
| `spontaneous_fire` / `spontaneous_cancel` | — | 自発発話の発火 / 取消 |
| `run_meta` | `{host, started_at, ended_at, app_commit, sample_count}` | 終了時に 1 行。記録の出所を検証するのに使う |

- `echo_chain` の検出則: AI 再生中に始まった発話の ASR 結果から句読点と空白を除き、それが直近の AI 発話本文の部分文字列なら 1 件と数える。設計書に具体的な則が無いため最小の解釈で実装している。短い語の偶然一致による誤検出があり得る (要確認)。

## 計測と合否判定 (設計書 §1.2 / §11.1)

| ゲート | 判定 |
|---|---|
| G1 応答遅延 | 50 ターンで p50 ≤ 1200ms かつ p95 ≤ 2000ms。`gpu1_load=true` の標本は別集計 (E9: GPU1 負荷下で p50 ≤ 1200ms) |
| G2 割り込み反応 | p95 ≤ 250ms |
| G3 旧世代音声 | 旧 epoch の再生 0 件 |
| 履歴整合 | turns と playback の照合で不整合 0 件 |
| G5 常駐 | 8 時間・1 分ごと記録で RSS の回帰傾きが 1.0 MiB/h 以下 (`analyze_soak`) |
| E10 / M5 | 20 分の記録で自声起因の応答連鎖 0 回、cancel 1 件以上 (`evaluate_echo_chain`) |

- `validate_run_meta` は、run_meta の 5 キー、期間の下限、ts の単調性、`sample_count` と実際の行数の一致を検査し、不備があれば `ProvenanceError` を投げる。
- 常駐試験の間隔検査 (`interval_ms=60000`) は、run_meta 以外の全行の隣接 ts 間隔にかかる。`soak_sample` の行だけではない。運用フェーズで確認する (README「運用フェーズでの確認事項」)。

## 設定既定値 (`config.Config`, 設計書 §8.1)

時間は特記なき限り ms。

- 発話区間と割り込み: `T_silence`=400、`MIN_SEG`=200、`T_confirm`=400 (300〜500)、`OVERLAP_MAX`=300、`DUCK_GAIN`=0.2。
- 相槌と自発発話: `SIL_BACKCHANNEL`=300 (250〜350 かつ `T_silence` 未満)、`BC_COOLDOWN`=5s、`spont_interval_s` 10〜3600 (既定 1200)。
- 通信と休止: `PING_INTERVAL`=5s、`IDLE_TIMEOUT`=1800s。
- 遅延予算と試験: `VAR_SUM`=750、`G2_SUM`=130、`TTS_RATIO_MAX`=0.7、`TTS_GPU_THRESH`=300、`TEST_N`=20、`N_TURNS`=8、`DB_BACKUP`=1。
- 範囲外の値は、`Config` を生成した時点で `ValueError` になる。

## ブラウザ (web/)

- `index.html` + `client.js`: WS の接続・再接続と state / error の表示。接続先ポートは `?port=`、無ければ配信元の `/config.json` の `ws_port`。
- `worklet.js`: AudioWorklet で 16kHz mono s16le の 20ms フレームに区切り、u32 の連番を付ける。実レートが 16kHz でなければ変換する。
- `player.js`: epoch ごとの再生、`duck` / `cancel` / `resume` の処理、`playback` の通知。旧 epoch のチャンクは再生しない。
- `capture_check.html` / `capture_check.js`: 取り込みの実現性 C1〜C4 (echoCancellation 制約、フレームの配信、連番の連続性、フレーム長) を判定し、結果 JSON を保存する。
- `aec_probe.js`: `getUserMedia({echoCancellation:true})` を使う AEC の実測プローブ。
- 画面の静的配信の仕組みは無い。`index.html` はファイルとして直接開く。

## テスト

- 既定: `pytest`(`-m 'not integration'`)と `node --test web/tests/*.test.js`。実サービスは不要。
- 実機: `pytest -m integration` と `web/tests/capture_record_live.js`。運用フェーズで実施する (手順と合格条件は README)。実機テストは skip を使わず、前提が欠けていれば失敗する。
