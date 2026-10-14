# idle-talk

PC に常駐して、日本語で気軽に雑談できる音声アプリです。音声認識・会話・音声合成はすべて手元の PC で動きます。

- 話しかければ答え、AI が話している途中でも割り込めます（「うん」などの相槌では止まりません）
- 黙っていると AI から話しかけます（ニュースの見出しを話題にすることもあります）
- 調べないと答えられない質問は Web で検索してから答えます

構成: ブラウザ（マイク・再生）⇄ Director（Python）⇄ llama-server（LLM）/ AivisSpeech（声）/ faster-whisper（音声認識）。設計は [docs/design-idle-talk.md](docs/design-idle-talk.md) にあります。

## 必要なもの

| もの | 用途 | 備考 |
|---|---|---|
| Windows 11 + NVIDIA GPU | 音声認識・LLM | 動作確認は RTX 3060 |
| Python 3.13 以上 | 本体 | |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) の `llama-server` | 会話 | モデルは Qwen3.5-4B Q4_K_M（gguf） |
| [AivisSpeech](https://aivis-project.com/) | 声 | 起動しておくと `127.0.0.1:10101` で待ち受けます |
| faster-whisper | 音声認識 | 初回起動時に `small` モデルを自動で取得します |
| onnxruntime + [WeSpeaker ResNet34](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) | 声で誰か見分ける（任意） | `voxceleb_resnet34_LM.onnx`（約 26MB、CC-BY-4.0）を `data/models/` に置きます。無ければ見分けずに動きます |
| Chrome などのブラウザ | マイク・再生 | |

## セットアップ

1. Python パッケージを入れる
   ```
   pip install -e .[dev]
   pip install faster-whisper
   ```
2. [idle_talk.toml](idle_talk.toml) の `[llm]` を自分の環境に合わせる
   ```toml
   [llm]
   server = "D:/tools/llama.cpp/llama-server.exe"
   model = "D:/models/Qwen3.5-4B-Q4_K_M.gguf"
   ```
   `model` を書いておくと、llama-server は idle-talk が起動し、落ちたら再起動します。空にすると自分で起動した llama-server（ポート `[ports].llm`）を使います。
3. AivisSpeech を起動しておく
4. （任意）声で誰か見分けるなら、話者識別モデルを置く
   ```
   pip install onnxruntime
   curl -L -o data/models/voxceleb_resnet34_LM.onnx https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx
   ```

## 起動と停止

リポジトリ直下で:

```
idle-talk.cmd start     # 起動（音声認識ワーカーと本体をバックグラウンドで起動）
idle-talk.cmd stop      # 停止
idle-talk.cmd restart   # 再起動
idle-talk.cmd status    # 動作状況
```

起動したらブラウザで **http://127.0.0.1:8000/** を開きます（`file://` で HTML を直接開くと接続できません）。

うまく動かないときは `data/idle_talk.log` と `data/asr_worker.log` を見てください。

## 画面の使い方

1. **開始** を押し、マイクの使用を許可します。これで会話が始まります。
2. 「状態」に今の様子が出ます。

   | 表示 | 意味 |
   |---|---|
   | 待機 | 話しかけを待っています |
   | ユーザー発話中 / 終話確認中 | あなたの声を聞いています |
   | 返答準備中 | 答えを考えています |
   | AI再生中 | AI が話しています |
   | 割り込み確認中 | AI の発話中にあなたの声を拾い、音量を下げて内容を確認しています |
   | 休止 | ミュート中、または長く操作がなかったため休んでいます |

   Web 検索中は「(Web で「〜」を調べています…)」と表示されます。

3. 設定欄（変更はすぐ反映されます。ただし保存はされず、次回はまた `idle_talk.toml` の値から始まります）

   | 項目 | 内容 |
   |---|---|
   | ミュート | マイクを無視して休止します |
   | 自発発話 ON | 黙っていると AI から話しかけます |
   | ラジオ | 返事を待たずに数秒おきに話し続けます |
   | 沈黙(秒) | この秒数黙っていると AI が話しかけます（10〜3600） |
   | 静音開始 / 静音終了 | この時間帯は AI から話しかけません |
   | 声 | AivisSpeech の話者を選びます |

### 会話のコツ

- AI の話を止めたいときは「待って」「ストップ」などと言うと、すぐに止まります。
- 「うん」「へえ」などの相槌なら、AI は少し音量を下げるだけで話し続けます。
- スピーカーで使うと AI 自身の声を拾うことがあります。気になる場合はヘッドホンを使ってください。
- 初めての声だと、2 回ほど話したあとで AI が名前を聞いてきます。答えると以後は名前で呼びます（声の登録は不要）。別人扱いされたら同じ名前を答えれば 1 人にまとまります。

## 設定ファイル

[idle_talk.toml](idle_talk.toml) の主な項目です。各項目の説明はファイル内のコメントにあります。

| セクション | 内容 |
|---|---|
| `[ports]` | 各サービスのポート（画面は `web`=8000） |
| `[tts]` | 声の話者 ID・話速・音量など |
| `[vad]` | 声の検出感度。物音に反応しすぎるなら `ratio` を上げ、声を拾わないなら下げます |
| `[spontaneous]` | 自発発話・ラジオモードの初期値 |
| `[topics]` | 話題に使うニュースの RSS |
| `[voice_id]` | 声で話し相手を見分ける。取り違えるなら `threshold` を上げ、同じ人を別人扱いするなら下げます |
| `[search]` | Web 検索の有無。on のときは、話題カード（ニュース）が尽きたら Bing ニュースで補充もします（`on = false` にすると検索も補充もせず、ネットワークにも出ません） |

## データとプライバシー

- 音声と書き起こしは外部に送りません。外部に出るのはニュースの取得と、Web 検索（検索語と読むページの URL）だけです。
- 会話の履歴は `data/idle_talk.db`、発話の文字は `data/talk.log` に保存されます。消すにはこれらのファイルを削除してください。
- AI が覚えた記憶は次のコマンドで確認・削除できます（起動中でも可）。
  ```
  idle-talk.cmd memories              # 一覧
  idle-talk.cmd memories --delete ID  # 1 件削除
  idle-talk.cmd memories --clear      # 全削除
  ```
- 覚えた声と名前も同様に確認・削除できます。声の特徴は `data/idle_talk.db` にだけ保存し、外部に送りません（起動中に消した場合は再起動後に効きます）。
  ```
  idle-talk.cmd speakers              # 一覧
  idle-talk.cmd speakers --delete ID  # 1 件削除
  idle-talk.cmd speakers --clear      # 全削除
  ```

## 開発者向け

- テスト: `pytest`（外部サービス不要）、`node --test web/tests/*.test.js`
- ソースは `src/idle_talk/`（Python）と `web/`（ブラウザ）にあります。

## ライセンス

このリポジトリのコードは [MIT License](LICENSE) です。

次のものはこのリポジトリに含まれず、それぞれのライセンス・利用規約に従います。

- AivisSpeech と音声モデル（話者ごとに規約が異なり、合成音声の商用利用やクレジット表記に条件があるものがあります）
- LLM のモデル（Qwen3.5 など）と llama.cpp、faster-whisper
- 外部サービス（DuckDuckGo、Bing、r.jina.ai、ニュースの RSS 配信元）
