# idle-talk アーキテクチャ案（2026-10 時点）

ローカルLLMで動く日本語の全二重音声チャット。ネットからニュース・話題を自動収集し、取り留めなく喋りつつ、ユーザーと会話もできる。

## 1. 方式選定

| 方式 | 例 | 採否 |
|---|---|---|
| E2E 全二重 speech-to-speech | LLM-jp-Moshi-v1（NII, 2026-02, 7B, Apache 2.0, 日本語・相槌対応）/ PersonaPlex-7B（英語のみ） | 不採用（主系）。知識処理・ツール呼び出し・ニュース要約が弱い。Phase 4 で実験 |
| **カスケード型 疑似全二重**（常時聴取＋割り込み＋相槌） | ASR → テキストLLM → TTS を並行ストリーミング（DuplexCascade 論文と同系統） | **採用**。テキストLLMの賢さを維持したまま全二重的に振る舞える |

## 2. 全体構成

```
[ブラウザ (localhost)]  ← WebRTC。ブラウザ内蔵のエコーキャンセル(AEC)を利用
     │ mic / speaker
[Pipecat パイプライン (Python)]
  Silero VAD → Smart Turn v3.2（日本語対応・CPU 10〜65ms）
     → ASR → ★Director（会話の司令塔） → LLM (Ollama) → TTS（文単位で逐次再生）
                  ▲                       │
                  │ 話題カード             └ 割り込み時は即停止し、
[Topic Worker（非同期・定期実行）]             実際に再生した所までで履歴を切る
  RSS/HN/arXiv取得 → 本文抽出 → LLM要約 → SQLite + 埋め込み（重複排除・興味スコア）
[Memory] 既存のエピソード記憶MCPを流用（ユーザーの反応・興味の学習）
```

- 音声I/Oはブラウザ側に置く → スピーカー使用時の全二重に必須な AEC が無料で得られる。バックエンドは Windows ネイティブでも WSL2 でも可（WSL2 の場合はモデルを ext4 側に置く）。

## 3. コンポーネント

| 層 | 推奨 | 備考 |
|---|---|---|
| フレームワーク | **Pipecat** | VAD・ターン判定・割り込み・Ollama連携・SmallWebRTCTransport が揃う |
| VAD / ターン判定 | Silero VAD（`stop_secs=0.2`）＋ Smart Turn v3.2 | 「言い終わり／途中」判定を相槌トリガーにも使う |
| ASR | **Qwen3-ASR-1.7B**（CER 0.140）/ whisper-large-v3-turbo（CER 0.184, 高速） | kotoba-whisper-v2.0 は自然会話で CER 0.495 と弱い |
| LLM（VRAM 16GB） | Gemma 4 12B (QAT) / gpt-oss-20b | 会話時は thinking オフで TTFT 優先 |
| LLM（VRAM 24GB） | **Gemma 4 26B-A4B**（MoE で高速）/ Qwen3.6-27B | Ollama。Gemma 4 はチャットテンプレート/ツール呼び出し設定に注意 |
| TTS | **Irodori-TTS**（声クローン、RTX3090で実時間5〜10倍、VRAM 3.7〜6GB）/ Style-Bert-VITS2・AivisSpeech（軽量・CPU可）/ VOICEVOX | 1〜2文単位で合成し順次再生 |

## 4. Director（自作の中核）

状態: `IDLE_MONOLOGUE`（独り言）/ `LISTENING` / `RESPONDING` / `BACKCHANNEL`（相槌）

- **独り言**: 無音が N 秒続いたら話題カードを選び、1〜2文ずつ区切って話す。区切りごとに短い間を入れ、口を挟みやすくする。
- **割り込み (barge-in)**: ユーザー発話開始で 200ms 以内に TTS 停止。LLM 履歴は**実際に再生した所まで**に切り詰める（でないと会話が噛み合わない）。
- **相槌**: Smart Turn が「途中」と判定したら事前合成済みの「うん」「へぇ」を再生。
- **興味学習**: 食いついた話題／すぐ遮られた話題をスコアに反映し、記憶に残す。

### 話題カード（Topic Worker 出力）

```
{ id, source_url, fetched_at, title,
  hook,            # つかみの一言
  facts[3],        # 要点3つ
  opinion_seed,    # AI側の意見のタネ
  question,        # ユーザーへの問いかけ
  embedding, interest_score, used_at }
```

- Topic Worker は音声ループと完全に分離（ネット待ちで会話を止めない）。
- 情報源候補: NHK / ITmedia / GIGAZINE / Hacker News / arXiv などの RSS。本文抽出は trafilatura 等。
- 埋め込み: BGE-M3 等で重複排除・類似検索。

## 5. 遅延予算

| 段 | 目安 |
|---|---|
| VAD 判定待ち | 200ms |
| Smart Turn | 50ms |
| ASR 確定 | 150ms |
| LLM 最初のトークン | 300ms |
| TTS 最初の音声 | 400ms |
| **合計** | **約1.1秒** |

短縮策: ASR の途中結果で LLM を投機的に先行実行／応答冒頭を相槌・フィラー（「あ、それね」）で埋める。

## 6. 進め方

1. **Phase 1**: 交互に話す形＋割り込み対応（Pipecat + Ollama + TTS をブラウザWebRTCで接続）
2. **Phase 2**: Topic Worker と Director、独り言モード
3. **Phase 3**: 相槌、興味学習、記憶連携
4. **Phase 4（実験）**: LLM-jp-Moshi に相槌・間合いを任せ、テキストLLMを頭脳にする二層構成

## 7. 本機構成（確定: Win11 / RTX 3060 12GB ×2 / RAM 64GB / 常用・軽量優先）

方針: **GPUごとに役割を固定**し、2枚に分割ロードしない（PCIe越しの分割は遅く、常用時に両GPUが塞がる）。モデルを増やさず、LLMは会話と話題要約で1つを共用する。

| GPU | 役割 | モデル | VRAM目安 |
|---|---|---|---|
| GPU0 | LLM | **Gemma 4 12B (QAT, Q4)**、ctx 8k、thinking オフ | 約9〜10GB |
| GPU1 | ASR | **whisper-large-v3-turbo**（faster-whisper, int8） | 約1.5〜2GB |
| GPU1 | TTS | **AivisSpeech Engine**（Style-Bert-VITS2系、VOICEVOX互換API） | 約1〜2GB |
| CPU | VAD / ターン判定 | Silero VAD + Smart Turn v3.2 (int8) | — |
| CPU | 埋め込み | 小型多言語埋め込み（話題の重複排除のみ、バッチ処理） | — |

- GPU1 は約8GB空く → 普段使い（ブラウザ・動画等）や、後で Irodori-TTS（声クローン、3.7〜6GB）への差し替えに回せる。
- TTS を AivisSpeech にする理由: 軽い／Windowsアプリで常駐が楽／VOICEVOX互換APIなので既存のVOICEVOX連携資産を流用できる。

### 軽量化ティア（必要に応じて切替）

| ティア | 構成 | 使い所 |
|---|---|---|
| 最軽量 | LLM を Gemma 4 E4B / Qwen3.5-4B にし、全部 GPU1 に載せる（計 約7〜8GB） | GPU0 を完全に空けたい時 |
| **標準（推奨）** | 上表 | 常用 |
| 高品質 | Gemma 4 26B-A4B を GPU0 + `--n-cpu-moe` で一部エキスパートを RAM へ（llama.cpp） | 会話の質を上げたい時。CPU負荷増 |

### 常用のための運用設計

- **Windows ネイティブで動かす**（音声・GPU・常駐の相性が良い。WSL2 は使わない）。Python は uv で管理。
- GPU固定: Ollama は `CUDA_VISIBLE_DEVICES=0`、ASR は `device_index=1`、AivisSpeech は GPU1 を指定。
- **待機時は解放**: 会話していない時は Ollama の `keep_alive` で LLM をアンロード（VRAM・電力を返す）。呼びかけ／ブラウザを開いた時に再ロード（数秒）。
- Topic Worker は 30〜60分おき、**会話していない時かつGPU0が空いている時だけ**実行。取得はCPU、要約だけLLM。
- タスクトレイ常駐（起動／一時停止／話題収集の停止）。

## 8. 未決事項

- TTS の声（AivisSpeech の既存モデルで良いか／クローンしたいか）
- 話題ソースの好み（技術系・一般ニュース・その他）

## 参考

- NII: LLM-jp-Moshi-v1 公開 (2026/2/25) https://www.nii.ac.jp/news/upload/nii_newsrelease_20260225.pdf
- DuplexCascade (arXiv 2603.09180) https://arxiv.org/abs/2603.09180
- NVIDIA PersonaPlex-7B https://huggingface.co/nvidia/personaplex-7b-v1
- Pipecat Smart Turn https://github.com/pipecat-ai/smart-turn
- 日本語ASRベンチマーク 2026 https://neosophie.com/ja/blog/20260226-japanese-asr-benchmark
- 日本語ローカルTTSを12個試して Irodori TTS に出会うまで https://zenn.dev/shuzan/articles/15b9aeffa4d6e0
- ローカルLLMガイド 2026夏 https://dev.classmethod.jp/en/articles/local-llm-guide-2026-summer/
