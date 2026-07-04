"""测试 MiniMaxSyncTTSService。

使用 aioresponses mock 单次 POST /v1/t2a_v2 的响应，覆盖：
- 正常合成（hex 解码 + base64 编码）
- 致命错误不重试（1000/1004/1042/2013）
- 可重试错误退避重试（1001/1002/1039），最终成功或全部失败
- 文本超 10000 截断
- payload 字段名校验（sample_rate / text_normalization）
- 错误码分类
- voice_id 命名规则
"""

import base64

import pytest
from aioresponses import aioresponses

import tts_service
from tts_service import MiniMaxSyncTTSService, generate_voice_id_from_filename


SYNTH_URL = "https://api.minimaxi.com/v1/t2a_v2"


def _post_calls(m, url: str) -> list:
    """从 aioresponses 的 requests 中提取发往指定 URL 的 POST 调用列表（保持顺序）。

    aioresponses 的 key 中 URL 为 yarl.URL 对象而非纯字符串，故需逐项比较 str(url)。
    """
    for (method, req_url), calls in m.requests.items():
        if method == "POST" and str(req_url) == url:
            return calls
    return []


def _ok_synth_payload(audio_bytes: bytes) -> dict:
    """构造成功的同步 TTS 响应 payload：data.audio 为 hex 字符串。"""
    return {
        "data": {"audio": audio_bytes.hex(), "status": 2},
        "extra_info": {
            "audio_length": 1,
            "audio_sample_rate": 32000,
            "audio_size": len(audio_bytes),
        },
        "trace_id": "test-trace",
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }


def _err_synth_payload(code: int, msg: str = "err") -> dict:
    """构造失败的同步 TTS 响应 payload。"""
    return {"base_resp": {"status_code": code, "status_msg": msg}}


# ---------------------------------------------------------------------------
# synthesize 正常流程
# ---------------------------------------------------------------------------

async def test_synthesize_success(sample_mp3_bytes, mock_logger):
    """正常合成：mock 返回 hex，断言 success=True 且 audio_base64 可还原成原 bytes。"""
    service = MiniMaxSyncTTSService(api_key="test-key", logger=mock_logger)
    try:
        with aioresponses() as m:
            m.post(SYNTH_URL, payload=_ok_synth_payload(sample_mp3_bytes))
            result = await service.synthesize(
                text="你好",
                voice_id="vid",
                audio_setting={"format": "mp3", "sample_rate": 32000},
                language_boost="auto",
            )

        assert result["success"] is True
        assert result["format"] == "mp3"
        assert result["text"] == "你好"
        audio_b64 = result["audio_base64"]
        assert isinstance(audio_b64, str) and len(audio_b64) > 0
        # base64 解码后应为原始 mp3 字节
        decoded = base64.b64decode(audio_b64)
        assert decoded == sample_mp3_bytes
    finally:
        await service.close()


# ---------------------------------------------------------------------------
# 致命错误不重试
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fatal_code", [1000, 1004, 1042, 2013])
async def test_synthesize_fatal_no_retry(fatal_code, monkeypatch, mock_logger):
    """致命错误码立即返回 success=False 且 code 字段，调用次数=1。"""
    service = MiniMaxSyncTTSService(
        api_key="test-key", max_retries=3, retry_backoff_base=1.0, logger=mock_logger
    )
    try:
        async def _no_sleep(_x):
            return None
        monkeypatch.setattr(tts_service.asyncio, "sleep", _no_sleep)

        with aioresponses() as m:
            m.post(SYNTH_URL, payload=_err_synth_payload(fatal_code, "fatal err"))
            result = await service.synthesize(text="hi", voice_id="vid")
            post_calls = _post_calls(m, SYNTH_URL)

        assert result["success"] is False
        assert result["code"] == fatal_code
        assert len(post_calls) == 1
    finally:
        await service.close()


# ---------------------------------------------------------------------------
# 可重试错误退避重试：前两次失败，第三次成功
# ---------------------------------------------------------------------------

async def test_synthesize_retryable_then_success(monkeypatch, mock_logger, sample_mp3_bytes):
    """前两次返回 1002/1039（retryable），第三次成功，断言 success=True 且调用=3。"""
    service = MiniMaxSyncTTSService(
        api_key="test-key", max_retries=2, retry_backoff_base=1.0, logger=mock_logger
    )
    try:
        async def _no_sleep(_x):
            return None
        monkeypatch.setattr(tts_service.asyncio, "sleep", _no_sleep)

        with aioresponses() as m:
            m.post(SYNTH_URL, payload=_err_synth_payload(1002, "rate limit"))
            m.post(SYNTH_URL, payload=_err_synth_payload(1039, "tpm limit"))
            m.post(SYNTH_URL, payload=_ok_synth_payload(sample_mp3_bytes))

            result = await service.synthesize(text="hi", voice_id="vid")
            post_calls = _post_calls(m, SYNTH_URL)

        assert result["success"] is True
        assert "audio_base64" in result
        assert len(post_calls) == 3
    finally:
        await service.close()


async def test_synthesize_retryable_all_fail(monkeypatch, mock_logger):
    """全部返回 retryable 错误，重试耗尽后 success=False，调用次数=max_retries+1。"""
    service = MiniMaxSyncTTSService(
        api_key="test-key", max_retries=2, retry_backoff_base=1.0, logger=mock_logger
    )
    try:
        async def _no_sleep(_x):
            return None
        monkeypatch.setattr(tts_service.asyncio, "sleep", _no_sleep)

        with aioresponses() as m:
            m.post(SYNTH_URL, payload=_err_synth_payload(1001, "timeout"), repeat=True)
            result = await service.synthesize(text="hi", voice_id="vid")
            post_calls = _post_calls(m, SYNTH_URL)

        assert result["success"] is False
        assert result["code"] == 1001
        assert len(post_calls) == 3
    finally:
        await service.close()


# ---------------------------------------------------------------------------
# 文本超 10000 截断
# ---------------------------------------------------------------------------

async def test_synthesize_text_truncation(mock_logger, sample_mp3_bytes):
    """传 12000 字符，断言实际发给 API 的 payload text 长度=10000。"""
    service = MiniMaxSyncTTSService(
        api_key="test-key", max_retries=0, retry_backoff_base=1.0, logger=mock_logger
    )
    try:
        with aioresponses() as m:
            m.post(SYNTH_URL, payload=_ok_synth_payload(sample_mp3_bytes))
            long_text = "a" * 12000
            result = await service.synthesize(text=long_text, voice_id="vid")
            post_calls = _post_calls(m, SYNTH_URL)

        assert result["success"] is True
        assert len(post_calls) == 1
        sent_payload = post_calls[0].kwargs.get("json", {})
        assert len(sent_payload["text"]) == 10000
    finally:
        await service.close()


# ---------------------------------------------------------------------------
# payload 字段名校验
# ---------------------------------------------------------------------------

async def test_synthesize_payload_field_names(mock_logger, sample_mp3_bytes):
    """capture 发送的 payload，断言字段名对齐官方同步 schema。"""
    service = MiniMaxSyncTTSService(api_key="test-key", logger=mock_logger)
    try:
        with aioresponses() as m:
            m.post(SYNTH_URL, payload=_ok_synth_payload(sample_mp3_bytes))
            await service.synthesize(
                text="你好",
                voice_id="vid",
                voice_setting={"text_normalization": True, "speed": 1.0},
                audio_setting={
                    "sample_rate": 32000,
                    "bitrate": 128000,
                    "format": "mp3",
                    "channel": 1,
                },
            )
            post_calls = _post_calls(m, SYNTH_URL)

        assert len(post_calls) == 1
        payload = post_calls[0].kwargs["json"]

        # audio_setting 含 sample_rate，不含 audio_sample_rate
        assert "audio_setting" in payload
        assert "sample_rate" in payload["audio_setting"]
        assert "audio_sample_rate" not in payload["audio_setting"]

        # voice_setting 含 text_normalization，不含 english_normalization
        assert "voice_setting" in payload
        assert "text_normalization" in payload["voice_setting"]
        assert "english_normalization" not in payload["voice_setting"]

        # 顶层字段
        assert payload["model"] == "speech-2.8-hd"
        assert payload["stream"] is False
        assert payload["voice_setting"]["voice_id"] == "vid"
    finally:
        await service.close()


# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------

def test_classify_error(mock_logger):
    service = MiniMaxSyncTTSService(api_key="test-key", logger=mock_logger)
    for code in (1001, 1002, 1039):
        assert service._classify_error(code) == "retryable", f"code {code} 应为 retryable"
    for code in (1000, 1004, 1042, 2013):
        assert service._classify_error(code) == "fatal", f"code {code} 应为 fatal"
    # 其他未知
    assert service._classify_error(9999) == "unknown"
    assert service._classify_error(0) == "unknown"


# ---------------------------------------------------------------------------
# generate_voice_id_from_filename 命名规则
# ---------------------------------------------------------------------------

def test_generate_voice_id_from_filename():
    def is_valid(vid: str) -> bool:
        if not (8 <= len(vid) <= 256):
            return False
        if not (vid[0].isascii() and vid[0].isalpha()):
            return False
        for c in vid:
            if not (c.isascii() and (c.isalnum() or c in ("-", "_"))):
                return False
        if vid[-1] in ("-", "_"):
            return False
        return True

    cases = [
        "narrator.wav",
        "温柔的少女.mp3",
        "123numeric.wav",          # 数字开头 → 前缀 v
        "ab",                       # 过短 → 补 voice
        "a" * 300 + ".wav",        # 过长 → 截断
        "hello world!@#.wav",      # 特殊字符 → 替换 -
        "中文音色.wav",
        "My-Cool_Voice.wav",
        "不错的_声音文件.mp3",
    ]
    for name in cases:
        vid = generate_voice_id_from_filename(name)
        assert is_valid(vid), f"生成的 voice_id 不合规: {vid!r} (from {name!r})"

    # 数字开头必须前缀字母
    assert generate_voice_id_from_filename("123.wav")[0].isalpha()
    # 扩展名应被去除
    vid2 = generate_voice_id_from_filename("narrator.wav")
    assert "." not in vid2
