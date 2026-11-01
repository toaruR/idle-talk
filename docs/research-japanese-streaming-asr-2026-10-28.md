# 日本語ストリーミングASR候補（2026-10-28）

## 確認したPCと現行構成

- CPU: AMD Ryzen 7 5700X、16論理プロセッサ（レジストリと .NET で確認）。
- RAM: 約64 GiB（.NET GCのTotalAvailableMemoryBytesは63.9 GiB。物理メモリのCIM照会はアクセス不可）。
- GPU: NVIDIA GeForce RTX 3060 12 GiB ×2（nvidia-smiで確認）。調査時の空きは約7.5 GiBと3.4 GiBで、変動する。
- `idle_talk.toml` は `engine = "reazon"`、CPU 2スレッド。
- `src/idle_talk/asr_worker.py` の `load_reazon_model` は `sherpa_onnx.OfflineRecognizer.from_transducer` を使用。モデルの差し替えだけでは音声入力からの逐次認識にならず、入力経路・認識状態・途中結果と確定結果の処理の変更が必要。

## 候補

| 候補 | 公式情報 | このPCでの評価・留保 |
| --- | --- | --- |
| NVIDIA Nemotron 3.5 ASR Streaming 0.6B | 日本語ja-JP対応。キャッシュを保持するFastConformer/RNN-T。80/160/320/560/1120 msのチャンクに対応 | 本命候補。RTX 3060で1ストリームを動かせる見込みは高いが、実測していない。チャンク幅は実際の表示遅延・発話確定遅延とは異なる。NeMoの公式OSはLinux。公式に案内されるNeMo-Speech.cppにはWindows向け手順と量子化GGUFの経路もある |
| Vosk small-ja-0.22 / ja-0.22 | 日本語モデル、48 MB / 約1 GBの配布サイズ。ストリーミングAPIあり。smallモデル一般の実行メモリ目安は約300 MB | CPUだけで軽く試すならsmall。GPUをLLM/TTS用に残せる。現行ReazonSpeechとの日本語精度の優劣は未比較 |
| Qwen3-ASR-0.6B | 日本語対応、公式にストリーミング対応。ただし確認したqwen-asr公式手順のストリーミングはvLLMバックエンド限定 | GPUでの比較候補。Windows上の既存sherpa-onnx構成に比べ導入経路が複雑になるため、最初の候補はNemotronまたはVoskとする |

## 判断

GPUを使うならNemotron 3.5 ASR Streaming 0.6Bを日本語固定、320 ms程度の設定から比較するのが妥当。CPU負荷と導入の軽さを優先するならVosk small-ja-0.22。推奨はハードウェアと公式の構成からの推定であり、このPCでの速度・日本語精度・LLM/TTS同時稼働時の測定結果ではない。インストール・モデル取得・設定変更は行っていない。

### CPUでの常時動作を優先した再評価

上記候補選定では日本語版Moonshine Streamingを見落としていた。公式の現行モデル一覧とCHANGELOGS.mdで、v0.1.5に日本語Small Streaming（123M）とTiny Streaming（34M）が追加されていることを確認した。いずれもMIT。旧日本語Tiny（26M）/Base（58M）は非ストリーミングで、これらと区別する必要がある。

配布されるストリーミングモデルは8bit量子化のORT形式で、Windows向け利用手順もある。CPUで軽く常時動かすという今回の優先条件では、まずMoonshine Japanese Tiny Streamingを試し、精度不足ならSmall Streamingを比較するのが妥当。Tinyのパラメータ数はNemotron 600Mの約1/18だが、CPU使用率や遅延が1/18になるという意味ではない。Windows標準ライブキャプションとの負荷・精度比較は未実施。

- [公式モデル一覧](https://github.com/moonshine-ai/moonshine/blob/main/docs/models/available-models.md)
- [量子化・評価条件](https://github.com/moonshine-ai/moonshine/blob/main/docs/models/accuracy.md)
- [Windowsを含むモデル取得手順](https://github.com/moonshine-ai/moonshine/blob/main/docs/using/downloading-models.md)
- [変更履歴](https://github.com/moonshine-ai/moonshine/blob/main/CHANGELOGS.md)

## 一次資料

- [Nemotron公式モデルカード](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [NVIDIA NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp)
- [Voskの機能](https://alphacephei.com/vosk/)
- [Vosk公式モデル一覧](https://alphacephei.com/vosk/models)
- [Qwen3-ASR-0.6B公式モデルカード](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)

## Windows標準ライブキャプション（追加調査）

- Microsoft Supportは、Azure AI Speechモデルのコンパクト版を端末内で使うと説明している。日本語対応。音声・認識結果はクラウドへ送信しない。
- 確認した公式説明では、具体的なモデル識別名・パラメータ数・アーキテクチャは示されていない。Whisper等の既知モデルと断定できない。
- 別の開発者向け機能としてWindows AI APIsの `Microsoft.Windows.AI.Speech` があり、CPU/NPUでオンデバイスのストリーミング認識が可能。確認時はexperimental、Windows 11 24H2以降とWindows App SDK 2.2.2-experimental9以降が前提。CPU推奨条件は4コア以上、ベースクロック3GHz以上、L3 32MB以上。
- 同APIの第三者通知にはWhisperのコンポーネント使用の記載がある。ただし、この事実から既存ライブキャプションがWhisperまたは同APIと同一モデルだとは推定しない。日本語対応範囲とこのPCの実際のOS/SDKでの利用可否は未確認。
- [ライブキャプションの公式説明](https://support.microsoft.com/en-us/accessibility/windows/use-live-captions-to-better-understand-audio)
- [Windows AI Speech Recognition API](https://learn.microsoft.com/en-us/windows/ai/apis/speech-recognition)
