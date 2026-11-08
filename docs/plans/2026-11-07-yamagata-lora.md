# 山形弁 (村山) LoRA 計画

作成日：2026-11-07
対象：Gemma 4 12B (`D:/models/gemma-4-12b-it-qat-q4_0.gguf`) に村山弁で話す LoRA を足す。

## 1. 方針

| 項目 | 決定 |
|---|---|
| 方言 | 村山 (山形市周辺)。標準語話者にも 8 割通じる強さ。規則は `tools/dialect/murayama_rules.md` |
| データ | `data/idle_talk.db` の実際の会話 (利用者の発話 → AI の返答、完了した返答 約 6100 件) から約 1500 件を選び、AI の返答だけを Claude (`claude -p`) で村山弁に書き換える。利用者が一部を目で見て直す |
| ベース | `unsloth/gemma-4-12B-it-qat-q4_0-unquantized` (今の GGUF と同じ QAT の重み。LoRA と推論の土台を揃える) |
| 学習 | WSL Ubuntu + Unsloth、GPU0 (RTX 3060 12GB) で 4bit QLoRA。r=16、2 epoch、seq 1024。学習中は idle-talk の LLM を止める |
| 書き出し | Unsloth で LoRA を保存 → llama.cpp の `convert_lora_to_gguf.py` で GGUF にする |
| 推論 | llama-server に `--lora` で渡す (`[llm].lora`)。ペルソナごとに効かせる切り替えは第 2 段 (§4) |

## 2. 手順

1. `python tools/dialect/extract_pairs.py` → `data/dialect/pairs.jsonl` (返答 30〜220 字、直前が利用者の発話、重複除去、最大 1500 件)
2. `python tools/dialect/convert.py --limit 40` で試し書き換え → 利用者が `data/dialect/converted.jsonl` を確認し、規則表を直す
3. `python tools/dialect/convert.py` で残りを書き換え (再開可能。済んだ id は飛ばす)
4. `python tools/dialect/review.py export` → `data/dialect/review.tsv` を直す → `review.py apply`
5. `python tools/dialect/build_dataset.py` → `data/dialect/train.jsonl` / `valid.jsonl` (system は `yamagata` ペルソナと同じ組み立て)
6. WSL で `bash tools/dialect/setup_wsl.sh` (初回のみ)、`tools/dialect/train.sh` で学習・GGUF 化 → `D:/models/lora-yamagata-murayama.gguf`
7. `idle_talk.toml` の `[llm].lora` に指定して再起動、`personas.toml` の `yamagata` で話す

## 3. 評価

- valid 5% の loss。
- 手で 30 の問いかけ (`tools/dialect/eval_prompts.txt`) を LoRA あり / なし (同じ system) で答えさせ、方言らしさ・意味の正しさ・不自然な方言の混ざり (関西・博多) を見る。
- 方言でない他のペルソナが崩れないか (§4 までは LoRA が全員に効く)。

## 4. 第 2 段：ペルソナごとの LoRA

llama-server は `--lora-init-without-apply` で読み込み、要求の `"lora":[{"id":0,"scale":1.0}]` で効かせられる。`personas.toml` に `lora = true` を足し、`clients.build_llm_request` に scale を渡す (要求スキーマの更新が要る)。第 1 段の出来を見てから行う。

2026-11-08 実装：`personas.toml` の `lora = true`（`uzu`・`yamagata`）、`clients.build_llm_request(lora=)`、`PersonaBook.lora_scale`。実機の llama-server で scale 0 は「〜とる」「お前」の標準語寄り、scale 1 は「ほだな」「〜だべね」「〜ず」になるのを確かめた。

## 4.5 第 1 版の結果 (2026-11-07)

- データ：抽出 1252 件 (返答 652、自発発話 600) → 書き換え後、別の人物設定の残り 44 件を除き train 1148 / valid 60。
- 学習：287 ステップ、約 2.5 時間。valid loss 4.64 → 1.869 (epoch 1.4 で 1.89 になり、その後はほぼ横ばい)。LoRA の GGUF は 131MB。
- `data/dialect/eval.md` (30 問、LoRA なしとありを同じ system で比べた)：
  - 良くなった点：濁音化 (「すごぐ」「わがる」)、「〜がら」「ごど」「〜だが？」「けろ」「ほんてん」が自然に出る。LoRA なしで混ざっていた他の方言 (「〜とる」「お前さん」) と文中の改行が消えた。
  - 問題 1：「(利用者は黙っています…)」に、ドラマを見ているかのような一言を返した。自発発話の学習データの多くは、話題 (ドラマの台詞・ニュース) が保存されていない一言だったため。
  - 問題 2：利用者の名前「リキ」を覚え、名前が system に無いときも呼んだ。
  - 問題 3：返答が長めになり、5〜6 文のものがある。
- 第 2 版でやること：話題の無い自発発話を学習から外す。返答の中の利用者の名前を「あんた」に置き換える。4 文を超える返答を外す。利用者による確認 (`review.tsv`) を反映する。

## 4.6 第 2 版 (2026-11-08)

- 利用者の確認 `data/dialect/review_v2.tsv` (200 件中 142 件を直した、外した行なし) を `review.py apply --tsv` で反映。反映前の `converted.jsonl` は `converted.v1.jsonl`、第 1 版の train/valid/log/eval は `data/dialect/v1/`、LoRA は `D:/models/lora-yamagata-murayama.v1.gguf` に退避。
- `build_dataset.v2_filter`：自発発話を全部外す (559 件)、5 文以上を外す (132 件)、利用者が名前を含む発話・名前そのものの話を外す (4 件)、返答の「リキ(さん/ちゃん)」を「あんた」に置き換える。
- 結果 train 396 / valid 20 (第 1 版の約 1/3)。学習は既定 (r=16、2 epoch)。
- 学習：100 ステップ、約 27 分。valid loss 2.042 (epoch 0.8) → 1.928 (1.6) → 1.917 (2.0)。valid の中身が第 1 版と違うので 1.869 とは比べられない。GGUF は `D:/models/lora-yamagata-murayama.gguf` (131MB)。
- 評価 `data/dialect/eval_v2.md` (30 問、なし / 第 1 版 / 第 2 版。`eval_lora.py --prev`)：
  - 名前：「リキ」は第 1 版 5 問 → 第 2 版 0 問。代わりに「あんた」が 13 問。
  - 長さ：平均 4.4 → 4.0 文、5 文以上は 13 → 7 問 (LoRA なしは 3.6 文、4 問)。
  - 話題の無い自発発話 (2 問)：第 1 版は 2 問ともドラマの一言だった。第 2 版は 1 問が架空のニュース、1 問が「さっきのドラマ」で、まだ直っていない (学習から外しても素のモデルより話題を作りがち)。

## 5. 限界

- 正解データは LLM の書き換えなので、本物の村山弁ではない言い回しが混ざる。利用者の確認がどれだけ入るかで決まる。
- 元の返答は博多弁・関西弁のペルソナが多い。書き換えで意味は保つが、話題の偏り (ドラマ・食べ物) はそのまま。
- TTS (AivisSpeech) の読みとアクセントは標準語のまま。
