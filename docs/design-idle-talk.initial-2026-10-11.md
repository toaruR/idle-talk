# idle-talk 設計書

作成日：2026-10-11
基礎資料：`docs/plans/2026-10-11-idle-talk-final-architecture.md`（合議結果。以下「合議書」）
対象：Windows 11、利用者1人、常駐する日本語の音声雑談アプリ
状態：設計確定版。既定値は §8.1 で決定済みで、M0/M1 の実測では値のみを更新する（設計構造は変更しない）。

---

## 1. 背景・問題・目的・非目標

### 1.1 問題
常駐する日本語雑談アプリを、既存 GPU 資源（RTX 3060 x2、GPU1 は他アプリ使用中）を壊さずに作る。既存の音声対話は割り込み・停止不能・遅延で雑談の楽しさを損なう。

### 1.2 目的（測定可能）
|ID|目的|目標値（試行数）|
|---|---|---|
|G1|発話終了から意味のある最初の音声まで|p50 ≤ 1.2s, p95 ≤ 2.0s（50 ターン）|
|G2|利用者の発話開始から AI 音声の低下（ダッキング）または停止の開始まで（VAD 経路）|p95 ≤ 250ms（50 回）|
|G3|停止（キャンセル確定）後に旧世代の音声が鳴る|0件（100 回の注入試験）|
|G4|話の途中を終話と誤判定|50 試行で 2 件以下|
|G5|8時間連続運転|メモリ増加の回帰傾き ≤ 1MiB/h、音声が止まらない|

### 1.3 非目標
本書末尾の「範囲外」行を唯一の定義とする。

### 1.4 制約・前提
- 利用者1人、ヘッドホン利用を初期前提。ブラウザは localhost のみ接続。
- 資源上限（超えたら構成見直し）：LLM は GPU0 で 9GiB 以内、ASR 追加 3GiB 以内、アプリ全体で追加 RAM 8GiB 以内。GPU1 は既存アプリ優先。
- 常駐プロセスは3つ：Python 本体（Director）、llama-server、AivisSpeech Engine。

---

## 2. アーキテクチャ

### 2.1 コンポーネント
ブラウザ（マイク/AudioWorklet・再生キュー・UI）⇄ WebSocket ⇄ Director（状態機械）。Director は VAD（CPU）、ASR（GPU0）、SQLite、llama-server（GPU0）、AivisSpeech（CPU）を呼ぶ。Topic Worker は M4 以降に SQLite へ書く。

### 2.2 技術選定
LLM は Qwen3.5 Q4_K_M（既定 4B、ctx 4k、思考無効はテンプレート設定）、ASR は faster-whisper を GPU0、TTS は AivisSpeech を CPU に置く。各配置の切替条件は §8.1/§9。

---

## 3. インターフェース契約

### 3.1 WebSocket プロトコル（ブラウザ <-> Director）
接続は `ws://127.0.0.1:<port>/ws`。バインドは 127.0.0.1 のみ。Origin が localhost 以外の接続は拒否する。同時接続は1本。2本目の接続は旧接続を閉じて置き換える（close code 4001）。

|方向|種別|ペイロード|備考|
|---|---|---|---|
|B→D|binary|16kHz mono s16le PCM、20〜40ms/フレーム|先頭に 4byte の連番 (u32 LE)|
|B→D|`hello`|§12.1|接続直後。version 不一致なら close 4002|
|B→D|`playback`|§12.1|実再生位置の通知|
| 共通 | `ping`/`pong` | §12.1 | `PING_INTERVAL` ごと。欠落 2 回で切断扱い |
|B→D|`ui`|§12.1|設定変更|
|D→B|binary|16byte ヘッダ + PCM（§12.1）|先読みは1チャンクまで|
|D→B|`duck`|§12.1|音量を下げる|
|D→B|`cancel`|§12.1|new_epoch 未満のキューを全破棄|
|D→B|`resume`|§12.1|ダッキング解除し続きから再生|
|D→B|`state`|§12.1|UI 表示用|
|共通|`error`|§12.1||


### 3.2 epoch 契約
- epoch は Director が単調増加で管理する u32。キャンセル確定時のみ +1 する。
- すべての生成物（LLM ストリーム、TTS 結果、音声チャンク）は生成開始時の epoch を持つ。epoch が現在値と異なる結果は到着時に破棄する（受信側の責務）。
- ブラウザは `cancel.new_epoch` 受信時、キュー内と再生中の旧 epoch チャンクを即時に破棄し、以後は旧 epoch のチャンクを受理しない。

### 3.3 外部サービス契約
|相手|呼び出し|タイムアウト|失敗時|
|---|---|---|---|
| llama-server | `/v1/chat/completions` stream | `HTTP_TIMEOUT`（§8.1） | 定型句で応答しログ。再試行は `LLM_RETRY` |
| AivisSpeech | `/cancellable_synthesis`（§12.1。その停止検証は T-E8） | `HTTP_TIMEOUT` | 該当文を飛ばし、テキストのみ UI に表示 |
|ASR ワーカー|内部キュー|`HTTP_TIMEOUT`|該当発話を破棄しログ。次の発話へ|

制約：AivisSpeech は `/cancellable_synthesis` を使い、計算停止は T-E8 で検証する（通常の `/synthesis` は切断後も計算が続く）。

### 3.4 SQLite スキーマ（概要）
|テーブル|主な列|
|---|---|
|`events`|id INTEGER PK, ts INTEGER, epoch INTEGER, kind TEXT, payload_json TEXT|
|`turns`|id INTEGER PK, ts INTEGER, role TEXT, text TEXT, completed INTEGER(0/1), stop_position_ms INTEGER NULL, utterance_id INTEGER|
|`topic_cards`|id INTEGER PK, hook TEXT NOT NULL, facts TEXT(JSON), source_url TEXT, published_at INTEGER(epoch ms), expires_at INTEGER, angle TEXT, used_at INTEGER NULL, user_reaction TEXT NULL|
|`memories`|id INTEGER PK, content TEXT, evidence_utterance_id INTEGER, updated_at INTEGER, deletable INTEGER|

WAL モード、書き込みは専用スレッドのキュー経由。スキーマ版は `PRAGMA user_version`。

---

## 4. 状態機械（Director）

### 4.1 状態
`待機 → ユーザー発話中 → 終話確認中 → 返答準備中 → AI再生中 → 割り込み確認中 → 休止`

### 4.2 遷移表
|現状態|イベント|ガード|次状態|アクション|
|---|---|---|---|---|
|待機|VAD 発話開始|ミュートでない|ユーザー発話中|ASR へ区間送出開始、背景生成中断|
|ユーザー発話中|VAD 発話終了|区間 ≥ `MIN_SEG`|終話確認中|終話判定タイマ開始|
|ユーザー発話中|VAD 発話終了|区間 < `MIN_SEG`|待機|区間破棄（雑音扱い）、イベント記録|
|終話確認中|無音が `T_silence` 継続|終話判定が「終話」|返答準備中|ASR 確定、LLM 呼び出し|
|終話確認中|VAD 発話再開|-|ユーザー発話中|タイマ取消（言いよどみ）|
|返答準備中|最初の合成可能な節|epoch 一致|AI再生中|TTS→再生キュー|
|返答準備中|VAD 発話開始|-|ユーザー発話中|生成を取消、epoch+1|
|AI再生中|VAD 発話開始|-|割り込み確認中|`duck`（強い発話・停止語なら即 `cancel`）|
|AI再生中|全チャンク再生完了|-|待機|履歴確定|
| 割り込み確認中 | 相槌と判定 | `T_confirm` 内 | AI再生中 | `resume` |
|割り込み確認中|割り込みと判定 / 判断不能|-|ユーザー発話中|`cancel`、epoch+1、旧生成物破棄、TTS 計算停止|
|任意|ミュート/長時間無操作/連続失敗|-|休止|背景処理停止、LLM はロード維持（省電力モード時のみ解放）|
|休止|解除操作|-|待機|解放していた場合はウォームアップ|

### 4.3 不変条件
1. 音を止める処理は LLM・ASR の完了を待たない。
2. AI 再生中もマイクと VAD を止めない。
3. 通常応答・自発発話・相槌クリップは同一の再生・キャンセル経路を通る。
4. 履歴に確定するのは再生完了チャンクのみ（送信完了は再生完了ではない）。
5. 沈黙は自発発話の同意ではない。

### 4.4 並行性の規則
- 状態遷移は Director 単一コルーチンで直列化する（イベントキュー）。外部コールバックは遷移を直接変更せずイベントをキューに積む。
- キャンセルと完了が競合した場合、epoch 比較で後勝ちを決める（旧 epoch の完了通知は無視）。

---

## 5. 割り込み・再生済み履歴

### 5.1 判定
- 相槌：短い発話で「うん」「へえ」等。割り込み：`STOP_WORDS`（§8.1）または `STRONG_UTT`（§8.1）を満たす発話。
- 「うん？」は疑問、「いや」は訂正になりうるため、短い発話を一律に相槌とみなさない。
- 判断できない場合は停止したままにする（安全側）。
- 既定はダッキング優先（強い発話・`STOP_WORDS` は即停止）。見直しは M1 で誤停止率と相槌頻度を見て行う。停止の確認窓は `T_confirm`（§8.1）。

### 5.2 履歴確定ルール
- チャンクは `epoch/utterance_id/chunk_id` で識別し、ブラウザの `playback` で実再生位置を得る。完了チャンクのみ履歴に確定し、途中停止は「中断された発話」として停止位置付きで記録する。
- 単語時刻が無い場合、再生秒数から聞こえた文章を推定しない。

---

## 6. 拡張設計

### 6.1 AI の相槌（M2〜）と自発発話（M3〜）
- 相槌：長めの発話中に無音が `SIL_BACKCHANNEL` 継続したとき、内容非依存の事前合成クリップを再生。1発話最大1回、`BC_COOLDOWN` を置く。誤発火は全記録。
- 自発発話：設定（オン/オフ、頻度、休止時間帯）は §8.1。雑談モードのみ。無反応なら連続せず待ち時間を延ばす。準備中に発話があれば取消す。

### 6.2 Pipecat への移行条件
スピーカー必須化、デバイス保守負荷の増大、別端末接続の要望のいずれか。先に自前 WebSocket の `getUserMedia({echoCancellation:true})` を実測し、合格条件は自分の声に応答が連鎖しないこと。

---

## 7. エッジケースと障害モード

E1〜E15 の検知・対処・テストは §12.3 に一本化する。

セキュリティ/プライバシー：音声と書き起こしは外部送信しない。ニュース取得（M4）以外のネットワークアクセスなし。保存データは SQLite のみで利用者が全削除できる。localhost バインドと Origin 検証で他サイトからのアクセスを遮断する。

---

## 8. 遅延予算と性能設計

応答遅延を終話待ち → ASR 確定 → LLM の最初の合成可能な節 → TTS 合成 → 再生開始に分解して記録する。目標値は §1.2 の G1、ここでは配分のみ。

|段階|p50 配分 (ms)|備考|
|---|---|---|
| 終話待ち（固定無音 `T_silence`） | `T_silence` 値（§8.1） | 既定値は §8.1 |
|ASR 確定|200|GPU0 想定|
|LLM 最初の合成可能な節|300|4B/9B で M0 比較|
|TTS 最初の文|250|閾値は `TTS_GPU_THRESH`（§8.1）|
|合計|`T_silence` + `VAR_SUM`|G1 p50 との差が余裕|

検算。
- G1 p50：合計 = `T_silence` + (ASR + LLM + TTS の配分) ≤ G1 p50 で合格。既定の余裕 = G1 p50 − (`T_silence` + `VAR_SUM`) = 1200 − 1150 = 50ms。合計 1200 は合格、1201 は不合格。
- G1 p95：固定の `T_silence` には倍率をかけず、可変段 `VAR_SUM` にのみ倍率 k を適用する。`T_silence` + `VAR_SUM`·k ≤ G1 p95 より k ≤ (G1 p95 − `T_silence`)/`VAR_SUM` = 2.1333…。実測 p95/p50 の各段比 r を k として代入し、r ≤ k なら合格（k=1.7 は合格、k=2.14 は不合格）。
- G2（VAD 経路）：VAD 判定窓 1 フレーム（最大 40ms、§3.1）+ Director の `duck`/`cancel` 送出（60ms 予算）+ ブラウザの音量変更・破棄（30ms 予算）= `G2_SUM` ≤ G2（既定 130 ≤ 250）。
- 停止語・継続発話経路：確定 `cancel` までは `T_confirm` + `G2_SUM`（範囲の下端 300 で 430ms、既定 400 で 530ms、上端 500 で 630ms）。この経路は G2 の対象外（G2 でダッキングは始まる）。STRONG_UTT の継続条件は窓終端で評価するため同経路。
- G4：許容件数ちょうどは合格、+1 件で不合格。G3：旧音声 1 件でも不合格。G5：傾き 1MiB/h なら 8 時間で +8MiB（ちょうどは合格）、+9MiB は不合格。
- STRONG_UTT：直近 1s 平均音量 0.1 に対し 0.2 は成立、0.19 は不成立。
- N_TURNS：8 ターン x 250 トークン + システム 500 = 2500 ≤ ctx 4096。
- TTS 比（`TTS_RATIO_MAX`=0.7、判定は「<」）：3.0s の音声を 2.0s で合成すれば 0.67 で合格、2.1s なら 0.70 で不合格。

計算量：VAD/ASR は発話長 n に対し O(n)。履歴は `N_TURNS` で制限し LLM 入力は O(1)。Qwen3.5 でキャッシュ再利用が効かなければ Qwen2.5-7B や Gemma 系を予備として比較する。事前合成の相槌音は遅延計測に含めない。

---

## 8.1 既定値表（本書で変数を使う箇所の唯一の定義）

|変数|既定値|根拠|調整時期|
|---|---|---|---|
|`T_silence`（終話待ち無音）|400ms|日本語の言いよどみは 200〜300ms が多い想定。短すぎると G4 違反|M2 で Smart Turn と比較|
|`TTS_GPU_THRESH`|300ms|超過（>）で GPU1 DirectML と比較。ちょうど 300ms は CPU 継続|M0|
|`PING_INTERVAL`|5s（欠落 2 回で切断扱い）|切断検知を 10s 以内に|M1|
|`LLM_RETRY`|最大 3 回 / 10 分、バックオフ 1s,2s,4s|E7 の再起動上限|M1|
| `IDLE_TIMEOUT` | 30 分無操作で `休止` | 長時間無操作の定義 | M3 |
| `STRONG_UTT` | 音量が直近 1s 平均の 2 倍以上、または発話が `T_confirm` 窓の終端まで継続 | 「強い発話」の定義。窓内で評価可能な条件のみ | M1 |
|`OVERLAP_MAX`|+300ms|E11 の許容遅延|M0|
|`HTTP_TIMEOUT`|LLM 最初のトークン 5s / 全体 20s、TTS 1 文 3s、ASR 区間長x2+2s|§3.3 の唯一の定義|M1|
|`DUCK_GAIN`|0.2|聞こえるが邪魔でない音量|M1|
|`VAR_SUM`|750（ASR 200 + LLM 300 + TTS 250 の p50 配分）|可変段の合計。§8 はこの名前で参照|M0|
|`G2_SUM`|130ms（VAD 40 + 送出 60 + ブラウザ 30 の予算上限）|G2 の内訳の唯一の定義|M1|
|`TTS_RATIO_MAX`|0.7|連続再生で途切れない上限（処理時間÷音声長 p95 <）|M0|
|`TEST_N`|同一シナリオの反復は 20 回、割り込み注入は G3 の試行数、負荷試験は 10 分・§1.2 の G1 の試行数|反復数の唯一の定義|M1|
|`E_PARAMS`|無音入力試験 60s／無音フレーム連続 3s で休止／cancel 後 2s 以内に TTS CPU が待機時 +5pt 以内|E4/E6/E8 の試験定数|M1|
| `DB_BACKUP` | 起動時に DB を 1 世代コピー（`.bak`）、破損時に復元 | E14 の復元元 | M1 |
|`MIN_SEG`（最短発話区間）|200ms|これ未満は咳・クリック雑音として破棄|M0|
|`T_confirm`（割り込み確認窓）|400ms（範囲 300〜500）|相槌と割り込みの判別に必要な最短観測時間|M1|
|`STOP_WORDS`|待って／違う／ちょっと／ストップ|即停止（ダッキング省略）対象|M1|
|`N_TURNS`（LLM に渡す履歴）|直近 8 ターン|ctx 4k に収まる概算（1ターン約 250 トークン）|M0|
| `SIL_BACKCHANNEL` | 300ms（範囲 250〜350、常に `T_silence` 未満） | AI 相槌のトリガー無音。`T_silence` 到達前に鳴らし、`T_silence` 到達（終話確認成立）時は未発火なら取り消す | M2 |
|`BC_COOLDOWN`|5s|連続相槌の煩わしさを避ける|M2|
|自発発話 初期値|既定オフ。オン時は `freq_per_hour`=3、休止時間帯 22:00〜07:00|控えめを既定にする|M3|
|`RATIO_Q`（質問終わり）|3 応答に 1 回以下|合議書の目安|M3|
|`ASR_HALLU`|no_speech_prob > 0.6 または compression_ratio > 2.4 で破棄|faster-whisper の標準的な閾値|M0|
|LLM モデル|暫定 4B（9B は §9 の決定ルールで昇格）|遅延目標を確実に満たす側|M0|
|ASR 配置|GPU0|§9|M0|
|TTS 配置|CPU|§9|M0|


境界値：G2（§1.2）ちょうどは合格（≤）。割り込み確認窓 `T_confirm` は [t, t+`T_confirm`) の半開区間で、窓の終端ちょうどに到着した音声は次窓に属する。無音が `T_silence` ちょうどで終話確認が成立する（≥）。

## 9. 実測による既定値更新ルール（M0）

- LLM：既定 4B。9B は、遅延目標（§1.2 G1）を満たし、30〜50 場面のブラインド比較で品質が明確に勝つ場合のみ昇格。判断材料は 9B の VRAM、最初の1文までの時間、20 ターンのキャッシュ効果。
- ASR 配置：既定 GPU0。GPU1 負荷変動、同居時の LLM 遅延、割り込み時の重なり遅延（`OVERLAP_MAX`）で判断。
- TTS 配置：既定 CPU。最初の1文の合成が `TTS_GPU_THRESH` を超えたら GPU1 DirectML と比較。
- 終話判定：既定は固定無音。G4 の試行での早切り数で Smart Turn v3.2 と比較。
- 上限は §1.4。思考無効はテンプレート設定で行い、実出力に思考が出ないことを確認する。

---

## 10. 段階導入と終了条件

|段階|内容|次へ進む条件|
|---|---|---|
|M0 計測|環境確認、ASR 幻聴率、LLM 4B/9B・TTS・ASR 配置の比較、VRAM/RAM|§9 の更新対象が確定|
|M1 最小MVP|音声往復、固定キャラ、2段階の割り込み停止、再生済み履歴、イベント記録|20分会話で音声欠け・旧音声再生・履歴不整合が0|
|M2 終話判定と相槌|固定無音 vs Smart Turn、AI 相槌、相槌判定と再開|早切りと誤発火が目標以内、煩わしさが増えない|
|M3 楽しさ|話題カード、自発発話、記憶、質問率制御|主観「続けたい」が上がる|
|M4 常駐|RSS Worker、8時間運転|p95 悪化なし、メモリ増加なし|
|M5 スピーカー|ブラウザ AEC 実測、不可なら Pipecat 比較|応答連鎖なし、割り込み可能|


---

## 11. 検証計画

### 11.1 指標と合格基準
|検証項目|方法|合格基準|
|---|---|---|
|応答遅延 G1|段階別タイムスタンプを `events` に記録|§1.2 の G1|
|停止遅延 G2|録音ループバックで測定|§1.2 の G2（ダッキング/停止の開始）|
|旧世代音声 G3|割り込み注入試験|§1.2 の G3|
|終話誤判定 G4|言いよどみ台本|§1.2 の G4|
|TTS 速度|処理時間 ÷ 音声長|p95 < `TTS_RATIO_MAX`、途切れなし|
|履歴整合|割り込み後の `turns` と実再生位置を照合|不整合0|
|常駐 G5|8時間運転、RSS・VRAM を 1 分ごとに記録|§1.2 の G5。RSS の回帰傾き（G5 の値）|

### 11.2 テスト層
- 単体：状態機械の遷移表を表駆動で全遷移網羅（epoch 競合を含む）。
- 結合：模擬ブラウザと模擬 LLM/TTS（遅延・失敗注入）で割り込みと E5/E7/E8/E15 を検証。
- 実機：M0〜M5 の終了条件。

---

## 12. 補足仕様

### 12.1 メッセージスキーマと型・範囲
```json
{"type":"object","required":["t"],"definitions":{"u32":{"type":"integer","minimum":0,"maximum":4294967295}},"oneOf":[{"properties":{"t":{"const":"hello"},"version":{"type":"integer","minimum":1},"sample_rate":{"const":16000},"client_id":{"type":"string","minLength":1,"maxLength":64}},"required":["t","version","sample_rate","client_id"]},{"properties":{"t":{"const":"playback"},"epoch":{"$ref":"#/definitions/u32"},"utterance_id":{"$ref":"#/definitions/u32"},"chunk_id":{"$ref":"#/definitions/u32"},"position_ms":{"type":"integer","minimum":0,"maximum":600000},"state":{"enum":["started","finished","dropped"]}},"required":["t","epoch","utterance_id","chunk_id","position_ms","state"]},{"properties":{"t":{"const":"ui"},"mute":{"type":"boolean"},"on":{"type":"boolean"},"freq_per_hour":{"type":"integer","minimum":0,"maximum":12},"quiet_start":{"type":"string","pattern":"^[0-2][0-9]:[0-5][0-9]$"},"quiet_end":{"type":"string","pattern":"^[0-2][0-9]:[0-5][0-9]$"}},"required":["t"]},{"properties":{"t":{"enum":["ping","pong"]},"ts":{"type":"integer"}},"required":["t","ts"]},{"properties":{"t":{"const":"duck"},"epoch":{"$ref":"#/definitions/u32"},"gain":{"type":"number","minimum":0,"maximum":1}},"required":["t","epoch","gain"]},{"properties":{"t":{"const":"cancel"},"new_epoch":{"$ref":"#/definitions/u32"}},"required":["t","new_epoch"]},{"properties":{"t":{"const":"resume"},"epoch":{"$ref":"#/definitions/u32"}},"required":["t","epoch"]},{"properties":{"t":{"const":"state"},"state":{"enum":["idle","user_speaking","eot_check","preparing","ai_playing","interrupt_check","rest"]},"muted":{"type":"boolean"}},"required":["t","state","muted"]},{"properties":{"t":{"const":"error"},"code":{"type":"integer","minimum":4000,"maximum":5999},"message":{"type":"string","maxLength":200}},"required":["t","code","message"]}]}
```
状態値は 待機=idle、ユーザー発話中=user_speaking、終話確認中=eot_check、返答準備中=preparing、AI再生中=ai_playing、割り込み確認中=interrupt_check、休止=rest。`ui` の `quiet_*` は HH:MM。
バイナリ：上り＝ `seq:u32LE` + PCM。下り＝ `epoch:u32LE, utterance_id:u32LE, chunk_id:u32LE, length:u32LE` の 16byte ヘッダ + s16le PCM（最大 1MiB/チャンク）。

外部サービス（JSON Schema draft-07。req=要求、res=応答。エラー応答は共通 `err`）：
- `err`（HTTP 4xx/5xx）：`{"type":"object","required":["error"],"properties":{"error":{"type":"object","required":["code","message"],"properties":{"code":{"type":"integer"},"message":{"type":"string"}}}}}`
- llama-server req：`{"type":"object","required":["model","messages","stream","max_tokens"],"additionalProperties":false,"properties":{"model":{"const":"local"},"messages":{"type":"array","items":{"type":"object","required":["role","content"],"properties":{"role":{"enum":["system","user","assistant"]},"content":{"type":"string"}}}},"stream":{"const":true},"max_tokens":{"const":120}}}`。res は SSE 各 data 行：`{"type":"object","required":["choices"],"properties":{"choices":{"type":"array","items":{"type":"object","required":["delta"],"properties":{"delta":{"type":"object","properties":{"content":{"type":"string"}}}}}}}}`、終端は `[DONE]`。エラー：`err` または `HTTP_TIMEOUT` 超過→定型句 + `LLM_RETRY`。
- AivisSpeech：`POST /audio_query?text=..&speaker=<id>` → JSON(AudioQuery)、続けて `POST /cancellable_synthesis?speaker=<id>`（body=AudioQuery）→ WAV。通常の `/synthesis` はキャンセル不可のため使わない。エラー：`err` または `HTTP_TIMEOUT` 超過→該当文を飛ばす。
- ASR ワーカー req：`{"type":"object","required":["utterance_id","pcm_ref","lang"],"additionalProperties":false,"properties":{"utterance_id":{"type":"integer","minimum":0,"maximum":4294967295},"pcm_ref":{"type":"string","minLength":1},"lang":{"const":"ja"}}}`（pcm_ref は共有メモリ名）。res：`{"type":"object","required":["utterance_id","text","no_speech_prob","compression_ratio"],"additionalProperties":false,"properties":{"utterance_id":{"type":"integer","minimum":0},"text":{"type":"string"},"no_speech_prob":{"type":"number","minimum":0,"maximum":1},"compression_ratio":{"type":"number","minimum":0}}}`。エラー：不正 req→HTTP 400 + `err`、`HTTP_TIMEOUT` 超過→発話を破棄しログ。


設定名の一覧：本書で `T_silence` のように示す全変数は §8.1 が唯一の定義で、その表が Options 型（名前・型・既定値）を兼ねる。メッセージ別のエラー条件：`playback` の epoch が現在値と異なる→破棄 + log、`ui` の値が範囲外→5006 を返し既存設定を維持、サイズ超過のバイナリ→破棄 + log。

### 12.2 エラーコード（4xxx は WS close code、5xxx はアプリ内 error メッセージ。二重起動はプロセス終了コード 4）
|code|意味|対処|
|---|---|---|
|4001|4002|version 不一致|UI に更新を促す|
|4003|Origin 不許可|接続拒否|
|5001|LLM タイムアウト/停止|定型句 + 再起動|
|5002|TTS 失敗/タイムアウト|文を飛ばす|
|5003|ASR 失敗|発話破棄|
|5004|DB 異常|`DB_BACKUP` から復元|
|5006|ui/playback の範囲外値|設定維持|
|5005|マイクデバイス喪失（E6）|休止|

### 12.3 障害モードの対応表（E-ID → 機構 → テスト）
|E-ID|機構|テスト（合否）|
|---|---|---|
|E1|終話確認中の VAD 再開で取消|T-E1：「今日は〜けど、」+ `T_silence` 未満の無音 + 続き。応答 0。無音が `T_silence` ちょうどで終話確認成立（境界）|
| E2 | ダッキング + `T_confirm` 判定 + resume | T-E2：再生中に「うん」`TEST_N`。`cancel` 0 件かつ再生位置が `T_confirm` 後に進行 |
| E3 | `STOP_WORDS` で即 cancel | T-E3：「待って」`TEST_N`。ダッキング開始が G2 以内、確定停止が `T_confirm` + `G2_SUM` 以内 |
| E4 | `ASR_HALLU` フィルタ | T-E4：無音 `E_PARAMS` の入力で出力 0 件 |
| E5 | `PING_INTERVAL` + epoch+1 | T-E5：WS 強制切断。`PING_INTERVAL` x 2 以内に検知、再接続後の旧音声 0 |
| E6 | 無音フレーム連続（`E_PARAMS`）で休止 | T-E6：デバイス抜去から `E_PARAMS` の休止条件 + 2 秒以内に `休止` 表示 + 5005 |
| E7 | `LLM_RETRY` + 定型句 | T-E7：llama-server プロセスを kill。`HTTP_TIMEOUT` 内に定型句、`LLM_RETRY` の初回バックオフ内に再起動し次ターンで応答 |
|E8|キャンセル可能合成 + 直列化|T-E8：cancel 後 `E_PARAMS` 内に TTS CPU が待機水準へ|
|E9|ASR/LLM を GPU0 固定|T-E9：GPU1 で GPU 使用率 90% の負荷（`TEST_N`）をかけ、G1 を満たす|
|E10|ヘッドホン前提（防御対象外。M5）|M5：再生中の自己認識による応答連鎖 0 回 / 20 分|
|E11|M0 で重なり遅延を測定|T-E11：割り込み重なりを `TEST_N` 再現し、追加遅延 p95 が `OVERLAP_MAX` 以内。超過で ASR を GPU1 へ|
|E12|`N_TURNS` 切り詰め|T-E12：G5 の試験で RSS 回帰傾きが G5 の値以内|
| E13 | E14 | 起動時 `integrity_check` + `DB_BACKUP`（§8.1） | T-E14：DB の先頭 100 バイトを破壊して起動。復元後の `turns` 行数が破壊前と一致 |
| BC | `SIL_BACKCHANNEL` 取消規則 | T-BC：無音が `T_silence` に達した試行で相槌 0 件、未達の試行で 1 件以下 |
|E15|epoch 比較で破棄|T-E15：発話 A の ASR と LLM に各 1500ms の遅延を注入し、A の 500ms 後に発話 B（epoch+1）を入力する試行を `TEST_N` 回。合格＝旧 epoch の再生 0/`TEST_N` かつ破棄 log 件数＝`TEST_N`|

防御しないもの：スピーカー時のエコー連鎖（M5 まで）、ASR の誤認識自体、GPU1 他アプリによる VRAM 枯渇。

### 12.4 防御機構ごとのトレードオフ
|機構|想定する誤検知|代替案|判断（reject/warn/log）|
|---|---|---|---|
|Origin 検証|Origin を付けない拡張|トークン認証|reject（4003）+ log|
|停止語辞書|「ちょっと」で誤停止|長さのみで判定|即停止 + log|
|幻聴フィルタ|小声の正規発話を破棄|除外リストのみ|破棄 + warn|
|ダッキング|停止遅れ|常に即停止|判断不能は停止|
|接続置換|二重タブで切断|後着を拒否|最新優先 + log|
|ping/pong|一時遅延で切断扱い|無期限待ち|欠落 2 回で切断 + log|
|ポートロック|ロック残り|PID ファイル|起動時に生存確認、reject|
|LLM 再起動上限|一時障害後に復帰しない|無制限再試行|超過で休止 + warn|
|`MIN_SEG`|短い正規発話（「うん」）を雑音扱い|閾値なし|破棄 + log|
|`T_confirm`/`STRONG_UTT`|長い相槌を割り込みと誤認|窓なし即停止|停止 + log|
|`SIL_BACKCHANNEL` 取消|相槌が出ず間が悪い|取消なし|取消 + log（T-BC）|
|epoch 破棄|正規の遅い結果も捨てる|全結果を再生|破棄 + log|
|履歴切り詰め|古い文脈の喪失|要約|切り詰め + log|
|localhost バインド|別 PC から使えない|認証付き公開|127.0.0.1 のみ（§6.2 の移行条件を満たすまで）|

---

範囲外（一行）：本書は設計のみで、実装コード、実測の実施結果（手順は §9/§11）、スピーカー運用、複数ユーザー、外部公開、配布・CI・監視の運用手順、ベクトル DB・声クローン・14B 級モデルは扱わない。
