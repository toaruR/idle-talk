# 音声認識の比較: ReazonSpeech (現行) と Haas-ja-v1.0

2026-10-31 測定。idle-talk の既定の音声認識 ReazonSpeech と、
[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) (2026-10 公開) を同じ条件で比べた。

## 結論

- **精度は Haas-ja が大きく上。** 3 データの重み付き CER strict は 7.36% → 4.79%。
  感情のこもった話し方 (JVNV) では 9.51% → 4.52% と半分以下になった。
  ReazonSpeech は文の一部を丸ごと落とすことがあり (JVNV で 27 件)、Haas-ja ではほぼ起きない (1 件)。
- **速さは ReazonSpeech が上 (CPU 同士)。** 1 発話の認識時間の中央値は 0.38 秒 → 1.29 秒 (2 スレッド)。
  Haas-ja は本来 GPU で動かすモデルで、CPU で動かすのは今回の検証のための回避策。
- **そのままは入れ替えられない。** Haas-ja の公式対応は Linux + NVIDIA GPU (NeMo + Triton)。
  idle-talk は Windows 11 で、GPU は LLM が使っている。採用するなら、動かし方を先に決める必要がある (下の「導入するなら」)。

## 測り方

| 項目 | 内容 |
|---|---|
| データ | [asr-ja-bench v1.1](https://huggingface.co/datasets/third-intelligence/asr-ja-bench) のうち JSUT basic5000・JVNV・FLEURS (7,025 発話、12.5 時間)。同データの rebuild.py で作り、正解文はハッシュで照合済み。TEDxJP-10K (YouTube から再構築が要る) と SpeechBSD (Hub のログインが要る) は使っていない |
| 指標 | [jacer](https://github.com/third-intelligence/jacer) v1.2 の CER strict / flex / kana。Haas-ja のモデルカードと同じ。サブセット内はマイクロ平均、全体は音声の長さで重み付け |
| ReazonSpeech | `sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01`。idle-talk の asr_worker と同じ読み込み (int8・greedy・2 スレッド) で 1 発話ずつ |
| Haas-ja | NeMo 2.7.3 + flash-linear-attention 0.5.2。GPU が無いので CPU・float32 で動かした (下記) |
| マシン | クラウドの Linux コンテナ、CPU 4 コア、GPU なし。idle-talk の実機 (Windows + RTX 3060) とは別 |

### Haas-ja を CPU で動かした方法

Haas-ja で GPU が要るのは予測ネットワーク (KDA, 3.9M パラメータ) の Triton カーネルだけ。
エンコーダ (609M) は普通の PyTorch なので CPU でも動く。

- Triton カーネルは `TRITON_INTERPRET=1` で CPU で解釈実行できる。ただし 1 発話に 80〜180 秒かかった。
- そこで、デコード中に毎ステップ呼ばれる 3 つのカーネルを、同じ式の PyTorch 実装に置き換えた ([tools/haas_cpu_patch.py](../tools/haas_cpu_patch.py))。
  - KDA の再帰更新
  - 短い畳み込み
  - ゲート付き RMSNorm
- 置き換えの確かめ方は次の 3 つ。
  - 乱数入力で元のカーネルと値が一致する (`python tools/haas_cpu_patch.py`)。
  - テスト音声 5 本で、解釈実行と認識結果が一字一句同じ。
  - JSUT の CER がモデルカードの公表値と一致する (下表)。

## 結果

### 精度 (CER %、低いほどよい)

| データ | 内容 | ReazonSpeech strict / flex / kana | Haas-ja strict / flex / kana | Haas-ja 公表値 (GPU, bf16) |
|---|---|---|---|---|
| JSUT (5,000) | 読み上げ | 6.67 / 2.99 / 1.74 | **5.32 / 2.43 / 1.36** | 5.32 / 2.43 / 1.36 |
| JVNV (1,375) | 感情音声 (怒り・喜び・驚きなど) | 9.51 / 5.91 / 4.39 | **4.52 / 3.06 / 2.25** | 4.53 / 3.08 / 2.26 |
| FLEURS (650) | 翻訳文の読み上げ | 6.28 / 5.07 / 4.40 | **3.69 / 3.02 / 2.38** | 3.67 / 2.98 / 2.34 |
| 重み付き (3 データ) | | 7.36 / 4.17 / 2.95 | **4.79 / 2.71 / 1.79** | |

- strict は漢字とかなの書き分けの違いも誤りに数える。flex はそれを数えない。kana は読みだけで比べる。
- 公表値との差は 0.04 ポイント以内だった。この差は float32 (CPU) と bfloat16 (GPU) の違いによる。

### 文の取りこぼし (JVNV)

| | ReazonSpeech | Haas-ja |
|---|---:|---:|
| 出力が正解の 70% 未満の長さ (文の一部を落とした) | 27 件 | 1 件 |
| 読みの誤り (CER kana) が 15% を超えた | 65 件 | 2 件 |

例 (正解 / ReazonSpeech / Haas-ja):

- ああ？最低な態度だよ。もう少しマナーを守れよ。 / もう少しマナーを守れよ / あん?最低な態度だよ。もう少しマナーを守れよ
- ええ、伝染病にかかったら本当に大変ですよ。私は…… / え私はそのような危険を冒すのは嫌です / ええっ、伝染病にかかったら本当に大変ですよ。私は……
- はぁ？またお説教ですか？ / あまたお説教ですか / はぁ?またお説教ですか?

ReazonSpeech は感嘆詞の直後の文を落としやすい。雑談では「えー、」「はぁ？」で始まる発話が多いので、この差は効く。

### 1 発話の認識時間 (CPU 2 スレッド、他の処理なし、120 発話)

| | 中央値 | p90 | RTF |
|---|---:|---:|---:|
| ReazonSpeech (int8) | 0.38 秒 | 0.66 秒 | 0.042 |
| Haas-ja (float32, CPU) | 1.29 秒 | 2.15 秒 | 0.152 |

- Haas-ja を GPU で動かした場合の速さは、ここでは測れていない。
- モデルカードの値は H200・バッチ 512 での RTFx 3,236 で、1 発話ずつの遅延とは比べられない。

### 音声でない音 (合成した 2 秒の音)

| 音 | ReazonSpeech | Haas-ja |
|---|---|---|
| 無音 | うん | (なし) |
| 白色雑音 | あ | うん。 |
| ピンク雑音 | あ | うん。 |
| クリック音 (食器のつもり) | ピッ | てくき |
| 120Hz のハム | はい | (なし) |

- 合成音を 1 本ずつ試しただけなので、傾向の参考程度に見てほしい。
- どちらも物音から短い語を作ることがあるので、idle-talk の `[asr].short_chars` フィルタは引き続き要る。
- このフィルタは句読点を除いて字数を数える (`_alnum`)。そのため Haas-ja が付ける「。」の影響は受けない。

### 録画したテレビ番組の音声 (asr_bench_tv, 80 クリップ・7.7 分)

`tools/asr_from_captions.py` で録画の放送字幕から作った手元のデータ (世界遺産の紹介番組のナレーション)。
同じ jacer で採点した。10/14 と 10/28 に測った結果 (results.csv / results_moonshine.csv) も採点し直して並べた。

| | 置換 | 脱落 | 挿入 | CER strict | CER flex |
|---|---:|---:|---:|---:|---:|
| ReazonSpeech (今回) | 30 | 98 | 26 | **9.47** | **8.36** |
| ReazonSpeech (10/14 の結果) | 30 | 98 | 23 | 9.29 | 8.18 |
| Haas-ja | 33 | **58** | 76 | 10.27 | 8.98 |
| Moonshine (ストリーミング) | 78 | 54 | 117 | 15.43 | 13.57 |
| Moonshine (一括) | 79 | 54 | 124 | 15.92 | 14.06 |

正解の文字数は 1,626 字 (Moonshine の採点時は 1,614 字)。

このデータでは、CER は ReazonSpeech のほうがわずかに低い。ただし、正解の作り方による偏りがある。

- 正解は放送字幕で、クリップの範囲も字幕の時刻で切っている。字幕に無い語が、クリップの端の音声に入っていることが多い。
- Haas-ja はそれを書き起こすので、「挿入」と数えられる。例: 正解「9位に選ばれました」→ Haas-ja「景観が9位に選ばれました」。
- ReazonSpeech は端の語を落としがちで、このデータではそれが有利に働く。
  - 例: 正解「北の異民族に備え 1つにつながりました」→ ReazonSpeech「一つにつながりました」
  - 例: 正解「…装着しました窓を開けて撮影するためです」→ ReazonSpeech「…装着しました」
- 発話の途中の誤り (置換) はほぼ同じ。
- 正解にある語を落とした数 (脱落) は、Haas-ja が 4 割少ない (98 → 58)。
- 挿入を除いて数えると、Haas-ja は 5.6%、ReazonSpeech は 7.9% になる。ただし、挿入のすべてが字幕の欠けとは限らない。

クリップ別に比べると、Haas-ja が良いのは 13 件、同じが 48 件、ReazonSpeech が良いのは 19 件。

## 導入 (2026-10-31)

idle-talk には 3 つ目の案 (ONNX) で入れた。`idle-talk.cmd asr haas` / `idle-talk.cmd asr reazon` で切り換える (README「音声認識の切り換え」)。

- NeMo・PyTorch・Triton は使わない。実行時に要るのは onnxruntime-gpu と numpy だけで、Windows でもそのまま動く。
- 重いエンコーダ (609M) だけを ONNX にして GPU で動かす。
- 予測ネットワーク (KDA)・joint・TDT 貪欲デコード・特徴量は numpy で書き直した ([src/idle_talk/haas_onnx.py](../src/idle_talk/haas_onnx.py))。どれも小さく、CPU の負荷はわずか。
- 書き出しは [tools/haas_export_onnx.py](../tools/haas_export_onnx.py) で行う (Linux / WSL2 で 1 回だけ、GPU は不要)。
- 配布用には [tools/haas_pack.py](../tools/haas_pack.py) で約 1.2GB に詰められる。重みは元々 bf16 なので、復元するとバイト単位で元に戻る。

NeMo との一致は次のとおり確かめた。

- 同じ特徴量を入れたとき、ONNX のエンコーダ出力は NeMo と完全に一致した。
- 特徴量は NeMo の前処理と最大 7e-5 の差だった。
- 認識結果は、asr-ja-bench の 7,025 件 (JSUT・JVNV・FLEURS) すべてで NeMo 版と一字一句同じだった (`tools/asr_bench_ja.py --engine haas_onnx`、CPU で確認)。

| | ReazonSpeech | Haas-ja |
|---|---|---|
| 動く環境 | CPU | GPU (onnxruntime-gpu の CUDA、または onnxruntime-directml)。`[asr].gpu = -1` で CPU |
| モデルの大きさ | 約 160MB (int8 ONNX) | 約 2.4GB (fp32 ONNX、配布時は 1.2GB) |
| 出力 | 句読点なし | 句読点あり (、。?!)。フィラーは省きがち |
| ライセンス | Apache-2.0 | Apache-2.0 |

GPU での遅延と VRAM は、実機 (RTX 3060) ではまだ測っていない。

## 再現

```
python tools/asr_bench_ja.py --bench <asr-ja-bench-v1.1> --engine reazon --reazon-dir <model dir> --threads 2
python tools/asr_bench_ja.py --bench <asr-ja-bench-v1.1> --engine haas --device cpu --threads 2 --batch-size 8
python tools/asr_latency_ja.py --bench <asr-ja-bench-v1.1> --n 40 --threads 2 --reazon-dir <model dir>
```

Haas-ja 側の準備:

```
pip install "haas-asr @ git+https://github.com/third-intelligence/Haas-ASR.git" "triton==3.8.0"
pip install "jacer @ git+https://github.com/third-intelligence/jacer.git@b861959"
```

- CPU 版の torch では、`triton==3.8.0` を別に入れる。
- `<asr-ja-bench-v1.1>` は、データセットの rebuild.py に JSUT・JVNV の zip を渡して作る。
