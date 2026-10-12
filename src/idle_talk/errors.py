"""エラーコード定義 (設計書 12.2)。4xxx は WS close code、5xxx はアプリ内 error メッセージ。"""

# WS close code
CLOSE_4001 = 4001  # 設計書 12.2 の表では意味欄が 4002 と併記されている
CLOSE_VERSION_MISMATCH = 4002  # version 不一致
CLOSE_ORIGIN_DENIED = 4003  # Origin 不許可

# アプリ内 error メッセージ
ERR_LLM_TIMEOUT = 5001  # LLM タイムアウト/停止
ERR_TTS_FAILED = 5002  # TTS 失敗/タイムアウト
ERR_ASR_FAILED = 5003  # ASR 失敗
ERR_DB = 5004  # DB 異常
ERR_MIC_LOST = 5005  # マイクデバイス喪失 (E6)
ERR_OUT_OF_RANGE = 5006  # ui/playback の範囲外値

# プロセス終了コード
EXIT_DUPLICATE_LAUNCH = 4

WS_CLOSE_CODES = (4001, 4002, 4003)
APP_ERROR_CODES = (5001, 5002, 5003, 5004, 5005, 5006)
