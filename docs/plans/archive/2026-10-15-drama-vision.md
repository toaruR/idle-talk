---
type: plan
status: done
owner: ai
---

# ドラマモードで画面も見る（画像入力）

作成日：2026-10-15
対象：設計書 §6.5 ドラマモード、§12.1 LLM 要求スキーマ、`[llm]` 設定

## 目的
ドラマの一言・総合的な感想を、台詞（音）だけでなくその時点の画面 1 枚も見て言わせる。Gemma 4 12B は画像入力を持つ（mmproj）。

## 実測（2026-10-15、b11564、Gemma 4 12B QAT + mmproj）
- 画像トークンは約 48px 角で 1 トークン。640×360 で 106、480×272 で 86、1280×720 で 407。
- 1 枚の上限は `n_ubatch`（既定 512）で頭打ち。
- 640×360 なら ctx 4096 に十分収まる（台詞 `ctx_chars` 1200 字 + system と合わせても余る）。

## 方式
1. **起動**：`[llm].mmproj`（既定 ""）を足す。空でなければ `--mmproj <path>` を渡す。`[llm].image_max_tokens`（既定 0 = 指定しない）が正なら `--image-max-tokens` を渡す。
2. **有効条件**：`[llm].mmproj` が空でなければ画像を使う（`vision`）。`/config.json` に `vision` を載せ、ブラウザはそのときだけ映像トラックを残す。
3. **ブラウザ**：
   - ドラマの共有で、`vision` なら映像トラックを止めずに `<video>`（DOM に付けない）で受ける。
   - サーバーから `{"t":"snap","id":n}` を受けたら、映像を 640×360（縦横比を保って収める）の canvas に描き、JPEG（品質 0.7）を `{"t":"snap","id":n,"jpeg":"<base64>"}` で返す。映像が無い・描けないときは `jpeg` を省いて返す。
   - 共有を止めたら映像トラックも止める。
4. **サーバー**：
   - 一言・感想を作る直前に `snap` を送り、`SNAP_TIMEOUT_S`（2 秒）だけ返事を待つ。来なければ画像なしで進む（一言は遅れてよいが、待ちすぎると場面がずれる）。
   - 届いた JPEG を `data:image/jpeg;base64,...` にして、user の `content` を `[{type:text}, {type:image_url}]` にする。プロンプトに「画面の画像も添えます。映っている様子も踏まえて」を足す。
   - 画像つきの呼び出しが失敗したら（mmproj 無しの llama-server が先に動いていた等）、画像なしで 1 回だけ呼び直す。
   - 画像はメモリにだけ置き、使ったら捨てる。ログには大きさ（KB）だけ残す。
   - あらすじメモの要約・通常応答には画像を使わない（応答遅延 G1 に足さない）。
5. **スキーマ**：`LLM_REQ_SCHEMA` の `content` を「string、または text / image_url の部品の配列」に広げる。WS の `snap` を `MESSAGE_SCHEMA` に足す（B→D は `jpeg` を base64 文字列で最大 `SNAP_MAX_B64`=1MiB）。

## 変更ファイル
- `src/idle_talk/config.py`：`LlmLaunch.mmproj` / `image_max_tokens`
- `src/idle_talk/llm_supervisor.py`：起動引数
- `src/idle_talk/clients.py`：`LLM_REQ_SCHEMA`
- `src/idle_talk/protocol.py`：`snap` スキーマ
- `src/idle_talk/server.py`：`snap` を Director へ渡す（`on_snap`）
- `src/idle_talk/director.py`：`_snapshot()`（要求と待ち）、一言・感想での画像添付、失敗時の画像なし再試行
- `src/idle_talk/main.py`：`vision` を Director と `/config.json` へ
- `web/client.js`：`startDramaCapture` の映像保持、`snap` への応答
- `web/index.html`：`vision` を渡す
- テスト：上記それぞれ
- `idle_talk.toml`：`mmproj` を設定
- `docs/design-idle-talk.md`：§6.5、§7、§12.1、改訂履歴

## 運用で確かめること（自動テストの範囲外）
- 画像を使うときの GPU0 の VRAM（mmproj なしで 8.0GiB）。
- 実際の画面での一言の遅れと、画像を踏まえた一言になっているか。
