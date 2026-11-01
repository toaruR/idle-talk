# Gemma 4 12B QAT + DFlash 実機比較

> 2026-10-29 清掃: このテストで追加したモデル・変換元重み・専用トークナイザーは削除済み。以下のモデル保存先は実験時点の記録。測定データ・レポートと既存モデルは保持。

同日追試: 以下は公開 Q4 ドラフトを使った初回試験の記録。後に `--target-model-dir` を指定して BF16 を自前変換し、再測定した。候補 15 では少数の採用があったが、高速化しなかった。以下の「BF16 未実施」は初回時点を指す。追加の条件・結果は [BF16 自前変換と再測定](research-dflash-bf16-2026-10-28.md) を参照。

2026-10-28（JST）。結論: **今回の PC・本体モデル・サーバービルドの組み合わせでは、高速化せず、生成速度は約 20〜33% 低下した。先読み候補の採用率は全測定で 0%。**

公開されている Gemma 全般の成功例を否定する結果ではない。今回の組み合わせでドラフト予測が本体の生成に利用されていないことを示す。

## 主な結果

日本語会話 3 種類を各 2 回。速度、最初の本文出力、最初の句点までの時間はリクエストごとの中央値。基準は試験の前後に各 6 件、計 12 件の会話を測定した値。

| 条件 | 会話の測定数 | 生成速度 | 最初の本文出力 | 最初の一文 | 候補の採用率 |
|---|---:|---:|---:|---:|---:|
| DFlash なし（前後の基準を統合） | 12 | **36.44 tokens/s** | **0.241 秒** | **0.775 秒** | — |
| DFlash 候補 2、williamliao 変換 | 6 | 29.09 tokens/s | 0.266 秒 | 0.937 秒 | **0%** |
| DFlash 候補 4、Anbeeld 変換 | 6 | 24.44 tokens/s | 0.267 秒 | 1.107 秒 | **0%** |

速度比は候補 2 が基準の約 0.80 倍、候補 4 が約 0.67 倍。異なる変換元を用いているため、この 2 行だけから候補数変更の純粋な効果は分離できない。両方とも基準より遅いことと、採用率ゼロは確認できた。

基準測定を個別に見ると、前が 36.17、後が 36.44 tokens/s。測定途中だけ基準モデルが大幅に遅くなったという状態ではなかった。

## 条件と測定方法

- GPU: NVIDIA GeForce RTX 3060 12 GiB × 2。
- サーバー: `D:/tools/llama.cpp/llama-server.exe`、`0.6.0-dev`、build 11564、commit `08ffba21e`。
- 本体: `D:/models/gemma-4-12b-it-qat-q4_0.gguf`（6,975,879,296 bytes）。
- 試験ポート: `127.0.0.1:18081`。既存サービスのポート 8091 は継続稼働。
- 主比較: context 4096、parallel 1、GPU layer split 3:1、Flash Attention on、batch 256、ubatch 128。
- ドラフト: Q4_K_M、CUDA1。試験用本体の共有 token embedding が CUDA1 に配置されるため。
- thinking 無効。実アプリのリクエストと同様、temperature 等はサーバー既定値を使用。取得した設定は temperature 1.0、top-k 64、top-p 約 0.95、min-p 約 0.05。
- prompt cache はリクエストで無効化。測定記録の cache_n は 0。
- 固定の system 指示と、雨の日の雑談、仕事のミスへの応答、短い会話履歴からの週末提案。日本語の JSON 出力を別枠の対照として追加。
- 主比較は最大 160 出力トークン。各条件で同じ prompt・seed 列を使用し、最初にウォームアップ。
- 生成速度はサーバーが返す `timings.predicted_per_second`。TTFT は HTTP リクエスト送信から最初の非空 `delta.content` までを計測。
- 最初の一文は最初の `。！？!?` を受信するまでの時間。音声合成・実際の読み上げは実行していない。

sampling を伴うため、同じ seed でも先読みあり／なしで文章や出力長が異なる。応答全体の時間だけで速度比を出さず、生成速度と出力開始の測定値を併記した。

稼働中の既存サービスやデスクトップアプリを残した試験のため、GPU 専有時の絶対速度とは区別する。GPU メモリは他プロセスを含む総使用量を 1 秒間隔で保存しており、試験プロセス単体の VRAM としては扱っていない。

## 採用率ゼロの切り分け

主比較と補足を合わせて **40 リクエスト**を測定。うち DFlash ありのリクエストで生成した候補は **7,813 トークン、採用は 0**。ウォームアップはこの集計から除外。

| 補足試験 | 件数 | 結果 |
|---|---:|---|
| williamliao、候補 2、temperature 0 | 会話・JSON 各 1 | 採用 0 |
| 同 GGUF の不足メタデータを補ったコピー、候補 2、temperature 0 | 会話・JSON 各 1 | 採用 0 |
| Anbeeld、候補 8、既定 sampling | 会話・JSON 各 1 | 採用 0 |
| Anbeeld、候補 4、1GPU、temperature 0 | 会話・JSON 各 1 | 採用 0 |

1GPU 試験のみ context 1024、batch/ubatch 64、最大 48 トークンに縮小。GPU0 の総メモリ使用量は最大 12,084 MiB に達したため、この試験の速度を主比較には混ぜていない。

本体と最初のドラフトの tokenizer 語彙は、ともに 262,144 要素で同一ハッシュ。層数・埋め込み次元も、本体 48 層・3840 次元に対してドラフトの取り出し位置と次元が対応する値だった。これだけで完全な互換性を保証するものではない。

最初の williamliao GGUF には、現行変換処理が Gemma 向けに付与する `dflash.embedding_scale` と `dflash.final_logit_softcapping` がなかった。別コピーにそれぞれ sqrt(3840)、30 を追加し、テンソルデータがバイト単位で同一であることを SHA-256 で確認して試したが、改善しなかった。これらを元から持つ Anbeeld GGUF でも採用はなかったため、**この欠落だけが原因とは断定できない**。

初回はドラフトを CUDA0 に置いたところ、CUDA1 の共有 embedding を使えず起動エラーになった。以後は CUDA1 に配置して正常に推論・測定できた。この起動失敗は上記 40 リクエストには含めていない。

現在確認できたのは「この組み合わせでは予測が採用されていない」まで。QAT 本体とドラフトの学習時モデルの差、GGUF 変換、当該サーバーの処理を完全に切り分けるには追加調査が必要で、特定の原因が確定したとは扱わない。

変換手順についての制限: 使用したのは公開済み Q4_K_M GGUF であり、この試験ではターゲットの元 Hugging Face ディレクトリを指定した `--target-model-dir ... --outtype bf16` による自前変換も、BF16 ドラフトとの比較も実施していない。語彙のハッシュ一致、メタデータ検査、実行時の共有 embedding の利用は確認したが、それらは公開 GGUF が正しい対象モデルを指定して変換されたことや、量子化前のドラフトが正常に機能することの実証にはならない。起動時の `-md`、`--spec-type draft-dflash`、`--spec-draft-n-max`、`-fa on`、`--jinja` は指定している。したがって、正規の BF16 変換経路まで検証し尽くした結果としては扱わない。

## 終了確認と成果物

- 試験用サーバーはすべて停止。最後の確認で llama-server は元の PID 90568 のみ。
- ポート 18081 の待受なし。元の 8091 の `/health` は `{"status":"ok"}`。
- `idle_talk.toml`、既存モデル、インストール済み llama.cpp は変更していない。
- メタデータ補正用の一時モデルコピーは削除。元の公開ドラフト 2 種類は `data/models/dflash/` に保存。
- 終了済みサーバーの returncode 1 は Windows の `terminate()` による意図した停止。起動失敗の `draft2/` だけは異常終了と error.json を記録。

再実行スクリプト: [tools/bench_dflash.py](../tools/bench_dflash.py)。原記録は `data/bench/dflash-20261028/`。以下を保存している。

- `aggregate.json`、`baseline-pooled.json`: 全条件の集計。
- 各条件ディレクトリ: 起動引数、サーバー設定、生成全文、SSE イベント、timings、GPU サンプル、サーバーログ。
- `model-provenance.json`: 公開モデルの revision と LFS SHA-256。ダウンロード後に照合済み。
- `cleanup-evidence.json`: 既存サービスの正常応答と試験用ポートの終了確認。
- `inspect_gguf.py`、`repair_gguf.py`、`summarize.py`: メタデータ検査、補正診断、集計の再現用。

公開モデル: [williamliao](https://huggingface.co/williamliao/gemma-4-12B-it-DFlash-GGUF/tree/05cc859a6ef67e83e415a6094b9e0b26ec7d8156)、[Anbeeld](https://huggingface.co/Anbeeld/gemma4-12B-it-DFlash-GGUF/tree/586b6ba31beba0d9dff7927b49d684728c4dfd69)。

ソース確認: [Gemma 用 scale と softcap の変換](https://github.com/ggml-org/llama.cpp/blob/08ffba21e/conversion/qwen.py#L806)、[ドラフトの埋め込み倍率適用](https://github.com/ggml-org/llama.cpp/blob/08ffba21e/src/models/dflash.cpp#L675)、[DFlash の候補位置](https://github.com/ggml-org/llama.cpp/blob/08ffba21e/common/speculative.cpp#L1354)。
