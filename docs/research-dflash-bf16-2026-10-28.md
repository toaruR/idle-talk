# DFlash の BF16 自前変換と再測定

> 2026-10-29 清掃: このテストで追加したモデル・変換元重み・専用トークナイザーは削除済み。以下のモデル保存先は実験時点の記録。測定データ・レポートと既存モデルは保持。

後続実験: 非QAT版を現在と同じ量子化構成に揃えて比較したところ、採用率は回復したが日本語会話では高速化に届かなかった。短い英語の対照では高速化した。詳細は [非QAT版との比較](research-dflash-nonqat-2026-10-28.md) を参照。以下の「非QAT本体は未実施」は本記録作成時点を指す。

2026-10-28（JST）。前回未実施だった `--target-model-dir` を指定した自前 BF16 変換を行い、現在の Gemma 4 12B QAT 本体と再比較した。

**結果: 自前 BF16 ドラフトでも、この本体・サーバーの組み合わせでは高速化しなかった。候補数 4 では採用率 0%、15 でも日本語会話の採用率は約 0.023%。**

## 指摘された手順への対応

| 項目 | 実施内容 |
|---|---|
| ドラフト変換元 | `z-lab/gemma4-12B-it-DFlash` の公式 `model.safetensors` |
| ターゲット指定 | `google/gemma-4-12B-it` の公式設定・トークナイザーを保存したローカルディレクトリを `--target-model-dir` に指定 |
| 出力形式 | `--outtype bf16`、ドラフトを再量子化していない |
| 変換コード | 手元のサーバーと同じ `08ffba21e5c2f1728b6b405c05ff3913f65706d4` の公式コード |
| 実行時 | `-md`、`--spec-type draft-dflash`、`-fa on`、`--jinja` を指定 |
| 候補数 | 4 と、提示された例と同じ 15 |
| 学習済みブロック | 16、anchor を除く最大 15 候補の範囲内 |
| 埋め込み | 元モデルの設定から倍率 sqrt(3840) を付与、実行時は本体の token embedding を共有 |

元ターゲットの約 23.9 GB の本体重みは取得していない。このドラフトの配布には `mask_embedding.pt` がなく、公式変換コードの当該経路は設定・トークナイザーで変換できる。実行時の embedding はロード済み本体から共有する。手作業で重みや埋め込みを変更した試験ではない。

## 変換の証跡

- ドラフト revision: `541d351de6dc9cd698fc348612bd5cdf03c636d0`。
- ターゲット revision: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`。
- 元ドラフト重み SHA-256: `8b32507dcff3ca4c1bbbb6c1cca7d17749bbf394480024e505228b6ed40afa02`。Hugging Face LFS メタデータと照合。
- ターゲット tokenizer.json SHA-256: `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`。同じく照合。
- 出力: `data/models/dflash/gemma4-12B-it-DFlash-self-BF16.gguf`、1,470,874,592 bytes。
- 出力 SHA-256: `2d35a14e5f755787f573b637d2d0a43a2e406dc5299bd6ec7c791f86112e9297`。
- `general.file_type=32`、BF16 が 36 テンソル、F32 が 22 テンソル。量子化テンソルなし。
- 本体とドラフトの語彙 262,144 要素をハッシュ照合して一致。
- `dflash.embedding_scale=61.967735290527344`、`dflash.final_logit_softcapping=30`、取り出し層 `[2,11,20,28,37,46]` を確認。
- Python 3.13、torch 2.12.0+cpu、transformers 5.9.0。CPU とローカルファイルのみで変換。

AutoConfig が `gemma4_unified` を認識しない警告は出たが、公式 `load_hparams` が元の config.json を直接読むフォールバックで続行し、exit 0 で完了した。語彙・設定・型は上記の通り別途検査した。警告を消すための変換コード改変は行っていない。

## 速度比較

今回は GPU 0 が空いていたため、RTX 3060 12 GiB の GPU 0 単独に本体とドラフトを載せた。前回の 2GPU 分割試験とは速度を直接比較せず、同条件の基準を前後に測り直している。

本体は `D:/models/gemma-4-12b-it-qat-q4_0.gguf`、サーバーは build 11564（`08ffba21e`）。context 4096、parallel 1、batch 256、ubatch 128、Flash Attention on、thinking 無効、prompt cache 無効。temperature 1.0、top-k 64、top-p 約 0.95、min-p 約 0.05 の既定値を使用。

同じ日本語会話 3 種類を各 2 回。基準は前後の計 12 会話、DFlash は各 6 会話。速度と時間はリクエストごとの中央値、採用率は該当する会話の候補・採用数の合計から計算。

| 条件 | 会話数 | 生成速度 | 基準比 | 最初の本文 | 最初の一文 | 採用率 |
|---|---:|---:|---:|---:|---:|---:|
| 先読みなし | 12 | **38.35 tokens/s** | 1.00 | 0.231 秒 | 0.752 秒 | — |
| 自前 BF16、候補 4 | 6 | 20.43 tokens/s | 0.53 | 0.333 秒 | 1.347 秒 | 0 / 2,378 = **0%** |
| 自前 BF16、候補 15 | 6 | 13.08 tokens/s | 0.34 | 0.328 秒 | 1.593 秒 | 2 / 8,670 = **約 0.023%** |

先読みなしの前後の中央値は 38.35 / 38.43 tokens/s。候補 4 は約 47%、15 は約 66% 遅くなった。

JSON 対照も各 2 回実施。既定 sampling の DFlash 16 リクエスト全体では候補 16,710 トークン中 6 トークンを採用。ウォームアップは除外した。

GPU 0 の総メモリ使用量の最大サンプルは、基準 7,253 MiB、候補 4 は 9,005 MiB、候補 15 は 9,007 MiB。他プロセスを含む 1 秒間隔の観測であり、プロセス専有量や瞬間的な厳密最大値ではない。

最初の本文は非空 delta.content の到着、最初の一文は最初の `。！？!?` 到着までの時間。音声合成・実際の読み上げは実行していない。

## 温度 0 の補足

候補 15 の BF16 と先読みなしで、同じ会話・JSON を各 1 件、最大 96 トークン、同じ seed で比較。会話の採用は 0 / 1,295、JSON は 1 / 1,293。温度を下げても採用率は改善しなかった。

出力全文は両件とも一致しなかった。会話は先頭 90 文字、JSON は先頭 51 文字まで一致した後に表現差があり、JSON では「聞きながら／聴きながら」などの差があった。バッチ演算による数値差なども考えられるが、原因は特定していない。この確認を一般的な出力品質や分布同一性の保証には使わない。

合計 36 リクエストを測定。DFlash 18 リクエストでは候補 19,298 トークン中 7 トークンが採用された。

## 判断と範囲

- 前回の「公開 Q4 ドラフトのみ」「target-model-dir を使った自前変換が未実施」という不足は解消した。
- BF16 でも採用率が極端に低く、現在の構成での高速化につながらなかった。
- QAT 本体とドラフトの学習時ターゲットの差や、当該サーバーのモデル処理・数値差は未切り分け。低採用率の原因が確定したわけではない。
- BF16 にしたのはドラフト。非 QAT の元モデル本体との比較は行っていない。
- 限られた日本語 synthetic prompt の結果であり、Gemma 全般や別実装の DFlash の有効性は否定しない。

## 追加の原因切り分け

英語との相性も確認するため、自前 BF16・候補 15・temperature 0・同じ QAT 本体・同じ GPU0 で短い 2 件を追加した。英語の計算は 0 / 960 候補、Python の sum_of_squares 生成は 0 / 405 候補で、両方とも採用率 0%。したがって、日本語会話だけで発生する症状ではなかった。実行記録は `english-greedy/`、再現用は `english_control.py`。

直接の減速要因は、ほぼ 1 トークンしか確定しないラウンドにドラフト生成・一括検証の追加コストが乗ること。候補を増やすほど有効な採用が増えずに遅くなる、という測定結果に対応する。

根本原因の最有力仮説は QAT 本体とドラフトが学習したターゲットの内部表現の不整合。ただし、非 QAT の本体に替えた対照実験は未実施であり、原因確定ではない。専用ドラフトは本体の中間表現を利用するので、語彙一致や正しい BF16 変換だけで、別チェックポイントとの互換性を保証できるわけではない。llama.cpp のモデル処理・量子化ターゲットの検証経路も候補に残る。

独立した [RTX 3060 での Gemma/DFlash 実測](https://lp-soroban.highreso.jp/compute-cluster/blog/dflash-diffusion-token-generation.html)にも、QAT 構成で 2 / 7,844 候補しか採用されない類似例がある。生ログからの率は約 0.0255%（記事本文の 0.25% とは桁が異なる）。設定・ビルドが本試験と同一ではなく、これだけで QAT を原因と確定はできない。

次に原因を絞るなら、非 QAT の同じ Gemma 4 12B 本体を、可能なら同じ Q4_0 形式に揃え、現在の BF16 ドラフト・サーバー・入力で比較する。これで採用率が回復するかを先に見てから、サーバービルドやカーネルの差を調べる。

ここまでの初回 BF16 測定は 36 件、英語補足を含めると 38 件。DFlash 候補は初回 19,298 に英語の 1,365 を加えた 20,663 トークン、採用は計 7 トークン。

## 成果物と終了確認（保存先）

原記録は `data/bench/dflash-bf16-20261028/`。

- `conversion.json` / `conversion.log`: 実変換コマンド、固定 revision、ハッシュ、終了コード。
- `gguf-verification.json`: 型、語彙一致、Gemma 向け設定。
- `aggregate.json` / `baseline-pooled.json`: 全条件と基準の集計。
- `greedy-comparison.json`: 温度 0 の出力比較。
- 各測定ディレクトリ: 起動引数、生成全文、SSE、timings、GPU サンプル、サーバーログ。
- `cleanup-evidence.json`: 試験プロセス・待受の終了と実サーバーバージョン。

試験用サーバーはすべて停止。終了確認時に llama-server とポート 18081 の待受はなかった。既存設定、本体モデル、インストール済み llama.cpp は変更していない。自前 BF16 と変換元ファイルは再検証用に保存した。

再現用: 同ディレクトリの `convert.py`、`summarize.py`、`compare_greedy.py` と [tools/bench_dflash.py](../tools/bench_dflash.py)。計測ツールのバージョン記録は標準エラーに出る情報も捕捉するよう修正した。

公式入力: [z-lab 固定 revision](https://huggingface.co/z-lab/gemma4-12B-it-DFlash/tree/541d351de6dc9cd698fc348612bd5cdf03c636d0)、[Google 固定 revision](https://huggingface.co/google/gemma-4-12B-it/tree/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7)。

処理の根拠: [target-model-dir と tokenizer](https://github.com/ggml-org/llama.cpp/blob/08ffba21e/conversion/qwen.py#L748)、[Gemma 向け倍率・softcap](https://github.com/ggml-org/llama.cpp/blob/08ffba21e/conversion/qwen.py#L806)、[実行時の共有 embedding](https://github.com/ggml-org/llama.cpp/blob/08ffba21e/src/models/dflash.cpp#L675)、[AutoConfig のフォールバック](https://github.com/ggml-org/llama.cpp/blob/08ffba21e/conversion/base.py#L1450)。
