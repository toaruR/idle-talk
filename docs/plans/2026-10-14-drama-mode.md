# ドラマモード（一緒に見て、たまに一言）実装計画

作成日：2026-10-14
対象設計：`docs/design-idle-talk.md`（§6.5 として追記する）

## 1. 目的
利用者がブラウザで見ているドラマの音を AI も聞き、場面の切れ目にたまに一言リアクションする。利用者が話しかければ、直近の台詞を踏まえて普段どおり答える。

非目標：映像の理解（音のみ）、台詞の保存、ドラマ音声の外部送信、Pipecat 化。

## 2. 方式の要点
- **取り込み**：ブラウザの `getDisplayMedia` でタブ（または画面全体）を共有し、その音声だけを使う（映像トラックは即 stop）。マイクとは別の AudioWorklet で 16kHz に落とし、同じ WS で送る。
- **チャネル識別**：上りバイナリの `seq:u32LE` の最上位ビット（`DRAMA_FLAG = 0x80000000`）を立てたフレームをドラマ音声とする。既存フォーマットは変えない（マイクの seq は 2^31 未満のまま。到達には 20ms フレームで約 1.4 年かかる）。
- **ドラマ側の処理**：専用の `EnergyVad`（adaptive、マイクとは別インスタンス）→ 区間化 → 既存 ASR → 台詞バッファ（メモリのみ、直近 `DRAMA_WINDOW_S`）。FSM には一切イベントを送らない（ドラマの台詞は利用者の発話ではない）。
- **リアクション**：自発発話の経路（`spontaneous.pending`・保留・取消・再生完了）を再利用し、`spontaneous_tick` の発火条件と生成部分だけをドラマ用に差し替える。LLM の出力は全文を受け取ってから判断し、「なし」なら黙る。
- **会話**：ドラマモード中の通常応答は system に直近の台詞（`DRAMA_CTX_CHARS` 字まで）を添える。

## 3. 既定値（`idle_talk.toml` の `[drama]`）
|名前|既定|意味|
|---|---|---|
|`interval_s`|90（実機で 240 では 1 回しか判定されず、それも「なし」だったため短縮）|前のリアクション（または開始）から最低これだけ空ける（30〜3600）|
|`gap_ms`|2500|ドラマの音声区間が終わってからこれだけ台詞が無ければ「場面の切れ目」|
|`window_s`|180|LLM に見せる台詞の範囲（`DRAMA_WINDOW_S`）|
|`min_chars`|20|前回リアクション以降に溜まった台詞がこれ未満なら反応しない（`DRAMA_MIN_CHARS`）|
|`ctx_chars`|1200|LLM に渡す台詞の上限字数（`DRAMA_CTX_CHARS`。ctx 4k に収める。古い方から捨てる）|
|`max_seg_s`|15|ドラマの 1 区間の上限。超えたら切って ASR に回す（BGM で区間が終わらない対策）|

ドラマモード自体のオン/オフは画面のボタン（共有開始/停止）で、設定ファイルでは持たない。

## 4. 変更内容

### T1. プロトコルと設定
- `protocol.py`：`DRAMA_FLAG`、`is_drama(seq)`。`ui` スキーマに `drama: boolean` を追加（`UI_FIELDS` にも）。
- `config.py`：`[drama]` の読み込みと範囲検査（上表）。`idle_talk.toml` に `[drama]` 節とコメントを追加。
- テスト：`test_config_protocol.py` に `drama` の ui 検証、`[drama]` 範囲外の拒否。

### T2. `src/idle_talk/drama.py`（新規）`DramaListener`
- `feed(frame)`：VAD → 区間化（`max_seg_s` で強制分割）→ `asr_submit(pcm, seg_s, done)`。
- 台詞バッファ：`deque[(t, text)]`、`window_s` 超過分を捨てる。`ASR_HALLU` と語なしは捨てる。
- `quiet_ms(now)`：最後の音声区間終了からの経過（発話中は 0）。
- `new_chars()`／`mark_reacted(now)`：前回リアクション以降の台詞字数。
- `should_react(now, idle)`：`idle` かつ `quiet_ms ≥ gap_ms` かつ `now − last ≥ interval_s` かつ `new_chars ≥ min_chars`。
- `context_text()`：直近の台詞を時系列で連結し、`ctx_chars` に収まるよう古い方を落とす。
- `contains(text)`：マイクの認識結果が直近の台詞に含まれるか（句読点・空白除去後、マイク側の 6 割以上かつ 4 字以上が連続一致。両方とも認識結果で完全一致しないため）。
- `stop()`：バッファと VAD 状態を消す（共有停止時。台詞は残さない）。
- テスト `tests/test_drama.py`：窓の切り捨て、`max_seg_s` 分割、発火条件の各境界（ちょうどは発火）、`contains`、stop で空になること。

### T3. Director への組み込み
- `_on_up_audio`：`is_drama(seq)` なら `self.drama.feed` へ回し、マイク側の VAD・相槌・E6 判定には入れない。ドラマモードでなければ捨てる。
- ASR：ドラマ区間は別の単一ワーカー（`_drama_pool`）で投げ、利用者の区間の認識を待たせない。（当初は IDLE 以外で捨てる案だったが、スピーカーの回り込みでマイクが発話中になると台詞が欠けて回り込み判定もできなくなるため、常に認識する。）
- `_on_ui`：`drama` で `DramaListener` を on/off。on の間は通常の自発発話（話題カード・ラジオ）と `IDLE_TIMEOUT` 休止を止める（off で元の設定に戻る）。
- `spontaneous_tick`：ドラマモード中は `drama.should_react` で発火し、`_drama_respond` を走らせる（`spontaneous.fire/pending/done/cancel` はそのまま使う）。
- `_drama_respond`：`DRAMA_PROMPT`（直近の台詞を渡し「一緒に見ている友人として一言（25 字以内）。言うことが無ければ『なし』とだけ」）で LLM を全文収集（`asyncio.to_thread(self._collect, ...)`）。「なし」・空・ドラマの台詞が再開している（`quiet_ms < gap_ms`）なら黙って `spontaneous.cancel()`。それ以外は `pipeline.play_utterance(iter([text]))` で再生。発火時に `drama.mark_reacted`（黙った場合も間隔は空ける）。
- `_chat_messages`：ドラマモード中は system に `「利用者と一緒にドラマを見ている。直近の台詞: …」` を足す。
- `_on_transcript`：ドラマモード中、マイクの認識結果が `drama.contains(text)` ならスピーカーからの回り込みとして捨てる（`_is_echo` と同じ扱い、events に `drama_echo_suppressed`）。
- テスト（`tests/test_drama_director.py`、模擬 LLM/TTS）：
  - ドラマフレームで FSM 状態が変わらない（応答 0）。
  - 条件成立で 1 回リアクション → 再生完了で `spontaneous.done`。
  - LLM が「なし」→ 音声 0、間隔は進む。
  - 生成中にドラマの台詞が再開 → 音声 0。
  - リアクション再生中に利用者が話す → 既存の自発発話と同じく保留→取消。
  - 通常応答の system に台詞が入る。
  - マイクの認識結果が台詞と一致 → 応答 0、`drama_echo_suppressed` 1 件。
  - ドラマモード中は話題カードの自発発話・`IDLE_TIMEOUT` が発火しない。off で戻る。

### T4. ブラウザ
- `worklet.js`：`processorOptions.seqBase`（既定 0）を追加し、`packFrame(seqBase | seq)`。ドラマ側は `seqBase = 0x80000000`（`>>> 0` で u32 に収める）。
- `client.js`：`displayConstraints()`（`video: true, audio: { echoCancellation:false, noiseSuppression:false, autoGainControl:false }, systemAudio: 'include'`）、`startDramaCapture({getDisplayMedia, AudioContextCtor, workletUrl, client, onEnded})`：音声トラックが無ければエラー（「タブの音声を共有」のチェックを促す文言）、映像トラックは即 stop、音声トラックの `ended` で `onEnded`。`makeUi` に `drama` を追加。
- `index.html`：「ドラマを一緒に見る」ボタン（開始後に有効）。開始で共有→`sendUi({drama:true})`、停止ボタン/共有終了で capture を止めて `sendUi({drama:false})`。状態表示に「ドラマ視聴中」。
- テスト：`worklet.test.js` に seqBase のフレーム、`client` の constraints/makeUi、`browser_globals.test.js` は既存のまま通ること。

### T5. ドキュメント
- 設計書：§6.5「ドラマモード」を追加、§3.1 に上りバイナリのチャネルビットと `ui.drama`、§7 に「ドラマの音声・台詞はメモリのみで保存も外部送信もしない」、改訂履歴に 1 行。
- README：使い方（Chrome でタブ共有時に「タブの音声も共有」をオン、スピーカーなら回り込みに注意しヘッドホン推奨）。

## 5. 実機で確認すること（運用フェーズ。自動テストに入れない）
- Chrome のタブ共有で音声が取れ、ドラマ再生音が利用者の発話と誤検知されないか（スピーカー時）。誤検知が多ければ `[vad].ratio` を上げるかヘッドホンで。
- ドラマ音声の ASR 常時実行で、利用者への応答遅延（G1）が悪化しないか。
- リアクションの頻度・内容が邪魔でないか（`interval_s`・`min_chars` の調整）。

## 6. リスク
- BGM・効果音で VAD が鳴りっぱなし → `max_seg_s` で切り、幻聴フィルタで捨てる。
- ASR の GPU0 負荷：区間は IDLE 時のみ処理するので応答とは競合しにくいが、LLM と同時に走る時間は出る。
- 4B が「なし」を守らず毎回何か言う → 間隔（`interval_s`）で頻度は上限がかかる。必要ならプロンプトで頻度を下げる。
