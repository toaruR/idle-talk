# Haas-ja-v1.0 と現行ASRの比較（2026-10-31）

## 判断

Haasは日本語認識精度を改善するための比較候補。ただし現行モデルとの同条件の実測はなく、置換による改善率・応答時間は未確定。CPUでの常時動作の軽さを優先するなら現行の利点が残る。真のストリーミング対応を目的とする置換にはならない。

Windowsでは公式Python実装よりspeech.cppの移植版が試しやすい見込み。導入・モデル取得・実行・設定変更は行っていない。

## 現行構成を確認した根拠

- `idle_talk.toml:10`：`engine = "reazon"`、`threads = 2`、`model_dir = "data/models/reazonspeech"`。
- `src/idle_talk/asr_worker.py:117`：encoder / decoder / joiner はint8版を優先。`load_reazon_model` はsherpa-onnxの `OfflineRecognizer.from_transducer`、greedy decodingを使用。
- ローカルモデルのREADMEと公式カード：**reazonspeech-k2-v2、159.34M、Zipformer / RNN-T、日本語、約30秒まで**。配布ディレクトリ名は `sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01`。
- 実際に選ばれるint8版3ファイルの合計は160,326,446 bytes（約160 MB）。README記載の配布サイズ約680 MBや、ディレクトリ内のFP32版を含めたサイズとは区別する。実行時RAMの測定値ではない。
- `data/asr_bench_tv/results.csv` を読み取り集計：80クリップ、CER **10.7579%**、認識呼び出しp50 **164.5 ms**、p95 **295 ms**、RTF **0.03058**。今回は再認識せず既存CSVを集計しただけ。`docs/design-idle-talk.md` の以前のp50 212 msとは計測回が異なる。
- ローカルCERはNFKC、句読点・空白除去、漢数字の正規化後の文字編集距離。発話終了待ち・音声取得・画面表示を含まない認識呼び出し時間。

## 比較

| 項目 | 現行 ReazonSpeech k2-v2 | Haas-ja-v1.0 |
| --- | --- | --- |
| 構造・規模 | Zipformer / RNN-T、159.34M | FastConformer / TDT、Kimi Delta Attention、約616M（移植版資料） |
| 実行経路 | Windows、sherpa-onnx、CPU 2スレッド | 公式はLinux + NVIDIA CUDA GPU、Python 3.11+ / NeMo 2.7 / Triton。speech.cpp移植版はWindows / Vulkanの配布あり |
| モデルファイル | 使用するint8重みは約160 MB | speech.cpp版Q8_0は約0.66 GB、Q4_Kは約0.35 GB。公式NeMo配布サイズとは区別 |
| 精度の根拠 | 手元のTV 80クリップでCER 10.8% | 開発元のasr-ja-bench v1.1（5データセット）で時間加重CER strict 6.12%。手元の10.8%とはデータ・集計・正規化が違い、直接比較不可 |
| 途中結果 | 文字起こしモードで1秒ごとに蓄積区間を再認識 | 非ストリーミング。公式デモも250 msごとの再認識。0.7秒無音か14秒で確定 |
| 長い音声 | 現行ラッパーは25秒ごとに分割。文字起こし側にも区切り処理あり | 30秒までの発話で学習。公式longform処理は静かな所で分割してバッチ認識 |
| 語彙指定 | 現行ラッパーは語彙ブースト未接続 | 公式・speech.cpp版に固有名詞等のカスタム語彙機能あり |
| 物音・短い返事 | 咳・食器等を短い語と誤認。コードで除外と短文フィルタを使用 | 同じ物音での評価なし。語彙ブーストにも誤挿入の注意あり。既存対策を不要とは判断できない |
| ライセンス | Apache-2.0 | 重み・公式コードはApache-2.0。移植版コードや依存は各ライセンス |

## 公開比較を読む際の注意

speech.cpp作者のFLEURS日本語650発話のQ8_0評価ではHaasがCER 4.39%、ReazonSpeechが7.95%。しかし後者は **reazonspeech-nemo-v2（619M、FastConformer、beam search）** であり、idle-talkの **k2-v2（159.34M、Zipformer、greedy）とは別モデル**。現行比約45%削減という結論には使えない。

同移植版の参考値はRTX 2080 / VulkanでHaas Q8_0の認識p50 97 ms、VRAM約0.78 GiB、RTFx 98。別PC・別音声なので、手元のp50 164.5 msに対する高速化率は計算しない。公式のRTFx 3,236も短い発話1件の体感遅延を表す数字ではない。

## idle-talkでの次の比較条件

1. 常駐するHaasワーカーを用意し、既存のTV 80クリップに同じ正規化を適用してCER・欠落・固有名詞を比較する。毎回プロセス起動・モデル読込をすると会話用途の測定にならない。
2. 同じ発話分割のまま認識呼び出しp50/p95、最初の途中文字、発話終了から確定まで、CPU/RAM/VRAMを測る。LLM・TTS同時稼働時も確認する。
3. 無音・咳・食器・BGM・AI音声の回り込み・「うん」等の短い返事を別途評価する。TV 80クリップだけで誤割り込みの改善とは判断しない。
4. 続けて話す動画は既存の区切り方を固定して比べ、その後に区切り処理を調整する。現行でも正しい区切りとVAD区切りでCERが大きく違うため、モデルの差と区切りの差を混同しない。

## 参照（2026-10-31確認）

- [ReazonSpeech k2-v2公式カード](https://huggingface.co/reazon-research/reazonspeech-k2-v2)
- [Haas公式推論コード・精度・実行条件・語彙機能](https://github.com/third-intelligence/Haas-ASR)
- [Haas公式デモの非ストリーミング方式](https://github.com/third-intelligence/Haas-ASR/blob/main/demos/README.md)
- [speech.cppのHaas対応・精度・速度・モデルサイズ](https://github.com/nyosegawa/speech.cpp/blob/main/docs/models/fastconformer.md)
- [speech.cppのWindows配布・Vulkan要件](https://github.com/nyosegawa/speech.cpp/blob/main/docs/install.md)
- ユーザー指定の[Haasモデルカード](https://huggingface.co/third-intelligence/Haas-ja-v1.0)は今回のWebツールでは取得不可。Haasの仕様は開発元の推論コードREADMEとデモ資料、移植版仕様はその作者の資料で確認した。
