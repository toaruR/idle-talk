# 呼びかけで ChatGPT の音声会話を始める（Chappy の再実装）

作成日：2026-11-19
状態：実装済み（2026-11-19）。設計書 `docs/design-idle-talk.md` §6.11 に反映。
元：`D:\vagrant\chappy`（Chappy 0.6.3。C#/WPF + WebView2）。Chappy は使わず、必要な部分を idle-talk に作り直した。

## 0. 経緯

- Chappy は Windows 標準の SAPI（`System.Speech` の文法認識 + 確信度）で呼びかけを待っていたが、検出の質が悪かった。
- idle-talk は常時 VAD + ReazonSpeech で全発話を文字にしている。実機ログで「チャッピー」も正しく書けていた（`data/talk.log` 2026-10-19：`PC: チャッピーです`）。
- 最初は「idle-talk が検出して名前付きパイプで Chappy を起こす」案だったが、利用者の判断で Chappy は生かさず idle-talk 側で再実装した。

## 1. Chappy から移したもの・変えたもの

| Chappy | idle-talk |
|---|---|
| SAPI の文法認識 + 確信度 | ReazonSpeech の認識結果を `wake_match`（正規化して完全一致、前後 2 字まで） |
| WebView2（アプリ内のブラウザ、専用プロファイル） | 専用プロファイル（`data/chatgpt-chrome`）の Chrome を `--remote-debugging-port` + `--app` で起動し CDP で操作 |
| `voice-probe.js`（ページの状態の判定） | `chatgpt_voice.PROBE_JS` にそのまま移した |
| `Input.dispatchMouseEvent` のクリック（WebView2 の CDP） | 同じ（Chrome の CDP）。信頼された入力なのでマイクの開始が許される |
| 会話中は SAPI の検出を休止 | 休止（`Ev.HANDOFF`）+ 上りの音を VAD・ASR に入れずに捨てる |
| `VoiceIdlePolicy`（WASAPI で応答の音を測る） | `IdleWatch`（マイクの音量 + ChatGPT の画面の生成中・応答の更新）。応答の音は測らない |
| 終了ボタン → 開始ボタンに戻ったのを確認、だめなら空白ページに隔離 | 終了ボタン → 確認、だめならページを開き直す |
| 開始音（`SystemSounds.Asterisk`） | ブラウザ（idle-talk の画面）が WebAudio で 2 音を鳴らす |
| 設定画面・トレイ・ログインの補助・自動起動 | 移さない（`idle_talk.toml` の `[chatgpt]`、ログインは専用 Chrome のウィンドウで） |

## 2. 呼びかけの判定

- 正規化：NFKC、カタカナ→ひらがな、英字は小文字、空白・句読点・長音（ー〜~・）を除く。
- 一致：正規化した発話が呼びかけを含み、残りが 2 字以下。
- 試算（`data/talk.log` の利用者・PC・ドラマ・文字起こしの 125,283 発話）：完全一致は 2 件（どちらも本当に「チャッピーです」と言った PC の声）。長音を除くと「ちゃっぴ」の 4 字になり、編集距離 1 を許すと「〜ちゃった」「迷っちゃって」等 24 件が当たったので採らない。ASR の書き違いは `phrases` に足す。

## 3. 実機で確かめたこと（2026-11-19）

- 専用プロファイルの Chrome を起動して CDP でタブにつながり、`PROBE_JS` が動く。
- 未ログインの chatgpt.com で、読み込み途中は `NeedsAction`、その後 `Login` になる（開始の判定は 2 秒続いたときだけ確定）。
- 音声会話のボタンの名前は「音声を開始」で `PROBE_JS` の開始ボタンの正規表現に一致する（「音声入力を開始」はディクテーションで一致しない）。

- 実機の呼びかけ 4 回（2026-11-19 12:06〜12:07）で ReazonSpeech は「チャピー?」「ジャッピー」「チャッピー」と書いた（もう 1 回は「きれい」）。`phrases` に「チャピー」「ジャッピー」を足した。過去ログ 12.5 万発話でこの 3 語の一致は 5 件で、すべて本当の呼びかけ。

## 4. 未確認（運用で確かめる）

- ログインした状態で、呼びかけ → 音声会話の開始 → 無操作の終了 → idle-talk に戻る、の一巡。
- 専用 Chrome の Google ログイン（remote debugging つきの Chrome を Google が拒まないか）。拒まれたらメールアドレスでログインする。
- 検出率：「チャッピー」を 30cm / 1m / 3m で各 20 回。
- 最小化・裏のウィンドウのままで音声会話が続くか（`--disable-backgrounding-occluded-windows` 等を付けている）。
- ヘッドホンのとき、ChatGPT が 30 秒を超えて話し続けると途中で終えてしまわないか（応答の音は測っていない）。
