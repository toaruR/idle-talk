# 非QAT Gemma 4 12B と DFlash の比較

> 2026-10-29 清掃: このテストで追加したモデル・変換元重み・専用トークナイザーは削除済み。以下のモデル保存先は実験時点の記録。測定データ・レポートと既存モデルは保持。

2026-10-28（JST）。QAT ではない Gemma 4 12B を現在と同じ量子化構成に揃えて試した。

**非QAT版ではドラフトの採用率が回復した。一方、日本語会話では今回試した DFlash 設定のどれも先読みなしを上回らなかった。英語の短い計算・コード生成と JSON 形式出力の対照では高速化を観測した。**

## 比較用の非QATモデル

配布リポジトリ `ggml-org/gemma-4-12B-it-GGUF` の通常名 `gemma-4-12B-it-Q4_0.gguf` は、変換ログによれば QAT の `QAT_Q4_0_UNQUANTIZED` 由来だった。このファイルは非QATの対照として使用していない。

同じリポジトリの BF16 は、通常の `google/gemma-4-12B-it`（PRIMARY）由来であることを変換ログと `.src_sha` で確認して取得した。

- 配布 revision: `202658970530b31541cd0b645a9c198bed2a615e`。
- PRIMARY の元 HF revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`。自前 BF16 ドラフトの変換時に使ったターゲット revision と一致。
- 取得した BF16: 23,832,066,176 bytes。
- BF16 SHA-256: `2851bc5b1a24e766e861652c6e5f72105b3b1efd55be35642456f1171714c292`。LFS メタデータと照合。
- インストール済み `llama-quantize` で `--pure --token-embedding-type q6_k ... Q4_0 8` として自前量子化。
- 非QAT出力: `D:/vagrant/idle-talk/data/models/gemma4-nonqat/gemma4-12B-it-nonqat-Q4_0.gguf`。
- 出力 SHA-256: `017919f0c0640dfb60cc224582502fb2568bb67885a42b455eab45af00e008ad`。

比較する QAT / 非QAT ファイルはいずれも **6,975,879,296 bytes、667 テンソル**で、F32 が 338、Q4_0 が 328、Q6_K の embedding が 1。語彙、チャットテンプレート、Gemma の推論ハイパーパラメーターも一致した。単なる「非QATだが量子化精度が異なるモデル」との比較を避けている。

QAT 側は既存の `D:/models/gemma-4-12b-it-qat-q4_0.gguf`、SHA-256 は `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`。

## 実行条件

- llama-server: `0.6.0-dev`、build 11564、commit `08ffba21e`。
- RTX 3060 12 GiB の GPU 0 単独。本体もドラフトも CUDA0。
- context 4096、parallel 1、batch 256、ubatch 128、Flash Attention on、thinking 無効。
- 日本語主比較は同じ system 指示・会話 3 種類・JSON 課題を各 2 回。各起動にウォームアップを置き、prompt cache は無効。
- temperature 1.0、top-k 64、top-p 約 0.95、min-p 約 0.05。QAT / 非QAT / DFlash 有無で sampler 設定が一致することを props で確認。
- BF16 ドラフトは前回正規手順で自前変換したものをそのまま使用。追加の軽量ドラフトは、それを同じ `llama-quantize` で Q4_K_M にしたもの。
- 非QATの先読みなし基準は前後に計 12 会話を測定。QAT 基準は今回改めて 6 会話を測定。
- 速度はサーバーの `timings.predicted_per_second` のリクエストごとの中央値。採用率は候補数と採用数の合計から計算。

## 日本語会話の結果

| 本体 | ドラフト | 候補上限 | 会話数 | 生成速度 | 採用率 | 最初の一文 |
|---|---|---:|---:|---:|---:|---:|
| QAT | なし | — | 6 | **38.19 tokens/s** | — | 0.768 秒 |
| QAT | BF16 | 4 | 6 | 20.43 tokens/s | 0 / 2,378 = **0%** | 1.350 秒 |
| 非QAT | なし | — | 12 | **38.33 tokens/s** | — | **0.699 秒** |
| 非QAT | BF16 | 4 | 6 | 27.31 tokens/s | 167 / 1,764 = **9.47%** | 1.007 秒 |
| 非QAT | BF16 | 15 | 6 | 16.85 tokens/s | 138 / 7,005 = **1.97%** | 1.614 秒 |
| 非QAT | Q4_K_M | 2 | 6 | 35.02 tokens/s | 159 / 979 = **16.24%** | 0.939 秒 |
| 非QAT | Q4_K_M | 1 | 6 | **35.77 tokens/s** | 129 / 464 = **27.80%** | 0.842 秒 |

非QAT基準の前後の中央値は 38.325 / 38.314 tokens/s。今回の会話では、DFlash の最良値でも先読みなしより約 6.7% 遅かった。

候補数が異なる採用率をそのままモデル精度として比較しない。長い候補列は後半の不一致を多く含むため、候補 1 と 15 の率だけからドラフトの品質差は判断できない。実用上は速度と採用で増えた出力数を合わせて見る必要がある。

「最初の一文」は句点などの到着までの時間で、音声合成・実際の読み上げは実行していない。生成内容や文の長さも条件間で異なるため、時間差にはその影響が含まれる。

## 対照: JSON 形式出力

同じ日本語の JSON 課題を各 2 回。これは形式を指定した生成の速度測定であり、JSON の厳密な構文検証や品質評価ではない。

| 非QATでの条件 | 生成速度 | 採用率 |
|---|---:|---:|
| 先読みなし（前） | 38.48 tokens/s | — |
| 先読みなし（後） | 38.48 tokens/s | — |
| BF16、候補 4 | 41.91 tokens/s | 27.36% |
| BF16、候補 15 | 26.97 tokens/s | 7.35% |
| Q4_K_M、候補 2 | **49.21 tokens/s** | 43.75% |
| Q4_K_M、候補 1 | 46.69 tokens/s | 64.52% |

予測しやすい構造を含むこの課題では、軽量ドラフト・候補 2 で約 1.28 倍になった。日本語会話の結果とは分けて扱う。

## 対照: 英語の計算と短いコード

前回 QAT 本体で採用率ゼロだった同じ 2 件を、temperature 0、BF16 ドラフト、候補 15、最大 80 トークンで再測定。非QATの先読みなし対照も新たに取得した。

| 課題（各 1 件） | 先読みなし | BF16 DFlash | 速度比 | 採用数 / 候補数 | 応答全体の時間 |
|---|---:|---:|---:|---:|---:|
| 17 × 23 の計算 | 38.25 tokens/s | **56.84 tokens/s** | **1.49倍** | 50 / 210 = 23.81% | 1.860 → 1.460 秒 |
| sum_of_squares 関数 | 37.61 tokens/s | **79.32 tokens/s** | **2.11倍** | 23 / 60 = 38.33% | 0.916 → 0.573 秒 |

非QATの先読みなしと DFlash の出力全文は、両件とも文字列一致した。出力長は計算が 65、コードが 28 トークン。初期入力処理を含む応答全体の短縮率と、生成部分の tokens/s の倍率は異なる。

さらに、コード生成の QAT / 非QAT の出力は同じ文字列だったが、前回 QAT は 0 / 405、今回非QATは 23 / 60 の採用だった。見える最終出力が同じでも、ドラフトの適合性は同じではないことを示す一例。英語の計算については QAT / 非QAT の全文は一致しなかった。

各 1 件の短い補足であり、英語やコード生成全般の倍率としては一般化しない。

## 今回の解釈

1. 同じ BF16 ドラフト・候補 4 で、QAT の会話採用率 0% に対し非QATは 9.47%。現在の QAT 本体との組み合わせが、極端な低採用率の大きな要因だったことを支持する。
2. 採用率が回復しても、日本語会話では追加計算を回収できるほどではなかった。今回の範囲では軽量化・候補削減でも先読みなしを上回らない。
3. JSON 課題と短い英語の対照では高速化しており、この PC で DFlash が一律に機能しないという結果ではない。対象モデルとの適合性に加え、入力・出力の予測しやすさも効いている。
4. 比較対象はこの QAT GGUF と、自前量子化したこの非QAT GGUF。元チェックポイントと変換経路の差もあるため、QAT という手法一般の因果を証明したものではない。

QAT版を使う現在の idle-talk を、そのまま DFlash 有効に変更する根拠は得られなかった。今回の日本語会話という用途では、DFlash なしが最も速い測定結果だった。

## 保存物と終了確認

主比較・対照を合わせて **68 リクエスト**。モデルの取得・量子化後に性能測定を実施し、各測定サーバーは自身の起動プロセスだけを停止した。ドラフトの追加量子化は非QAT・候補15サーバーの起動時と一部重なっており、同条件の微小な性能差の厳密比較にはその影響も残る。

試験用ポートは 18081。終了時に llama-server の残存と同ポートの待受がないことを確認した。既存の設定、本体、インストール済み llama.cpp は変更していない。

原記録: `data/bench/dflash-nonqat-20261028/`。

- `candidate-convert.log` / `candidate-source-revisions.txt`: 配布ファイルの変換元の確認。
- `model-source.json` / `quantization.json`: 固定 revision、変換コマンド、ハッシュ。
- `header-comparison.json` / `sampler-controls.json`: 語彙・テンプレート・設定の一致。
- `aggregate.json` / `model-baselines.json`: 全条件とモデル別基準。
- `english-comparison.json` / `cross-model-english.json`: 英語対照の速度と出力一致。
- 各測定ディレクトリ: 生成全文、SSE、timings、GPUサンプル、起動引数、サーバーログ。
- `cleanup-evidence.json`: プロセス・待受の終了。

利用可能なモデル:

- 非QAT本体: `D:/vagrant/idle-talk/data/models/gemma4-nonqat/gemma4-12B-it-nonqat-Q4_0.gguf`。
- 非QAT BF16元データ: 同ディレクトリの `gemma4-12B-it-nonqat-BF16.gguf`。再検証・別精度への変換用に保存。
- 自前軽量ドラフト: `D:/vagrant/idle-talk/data/models/dflash/gemma4-12B-it-DFlash-self-Q4_K_M.gguf`、442,399,712 bytes、SHA-256 `6f492f8c6a748f8538f84756f2bc918596ddad04c37615a9dbd4e77b8929c08b`。

再現用は同ディレクトリの `quantize.py`、`inspect_headers.py`、`english_control.py`、`verify_results.py`、`summarize.py` と [bench_dflash.py](../tools/bench_dflash.py)。

公開元: [BF16とQ4_0の変換ログ（固定 revision）](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/blob/202658970530b31541cd0b645a9c198bed2a615e/convert.log)、[元モデル revision](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/blob/202658970530b31541cd0b645a9c198bed2a615e/.src_sha)。
