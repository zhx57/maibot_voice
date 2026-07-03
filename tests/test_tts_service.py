"""测试 MiniMaxAsyncTTSService。

使用 aioresponses mock HTTP 请求，覆盖三步流程、错误分类、重试与轮询超时。
"""

import asyncio
import base64
import time

import pytest
from aioresponses import aioresponses

import tts_service
from tts_service import MiniMaxAsyncTTSService, generate_voice_id_from_filename


CREATE_URL = "https://api.minimaxi.com/v1/t2a_async_v2"
QUERY_URL = "https://api.minimaxi.com/v1/query/t2a_async_query_v2"
RETRIEVE_URL = "https://api.minimaxi.com/v1/files/retrieve"
DOWNLOAD_URL = "https://cdn.example.com/audio.mp3"

# aioresponses 0.7.9 对 URL 做精确比较（含 query string），故 query/retrieve 需带参数注册
QUERY_URL_TASK = QUERY_URL + "?task_id=123"
RETRIEVE_URL_FILE = RETRIEVE_URL + "?file_id=456"


def _ok_create_payload(task_id=123, file_id=123, usage=10):
    return {
        "task_id": task_id,
        "file_id": file_id,
        "usage_characters": usage,
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }


def _ok_query_payload(status="Success", file_id=456):
    return {
        "task_id": 123,
        "status": status,
        "file_id": file_id,
        "base_resp": {"status_code": 0, "status_msg": "ok"},
    }


def _ok_retrieve_payload(download_url=DOWNLOAD_URL, file_id=456):
    return {
        "file": {
            "file_id": file_id,
            "download_url": download_url,
            "filename": "out.mp3",
            "bytes": 100,
        },
        "base_resp": {"status_code": 0, "status_msg": "ok"},
    }


# ---------------------------------------------------------------------------
# synthesize 完整流程
# ---------------------------------------------------------------------------

async def test_synthesize_success(sample_mp3_bytes, mock_logger):
    service = MiniMaxAsyncTTSService(api_key="test-key", logger=mock_logger)
    try:
        with aioresponses() as m:
            m.post(CREATE_URL, payload=_ok_create_payload())
            m.get(QUERY_URL_TASK, payload=_ok_query_payload(status="Success", file_id=456))
            m.get(RETRIEVE_URL_FILE, payload=_ok_retrieve_payload())
            m.get(DOWNLOAD_URL, body=sample_mp3_bytes, content_type="audio/mpeg")

            result = await service.synthesize(
                text="你好",
                voice_id="vid",
                audio_setting={"format": "mp3", "audio_sample_rate": 32000},
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


async def test_synthesize_task_failed(mock_logger):
    service = MiniMaxAsyncTTSService(api_key="test-key", logger=mock_logger)
    try:
        with aioresponses() as m:
            m.post(CREATE_URL, payload=_ok_create_payload())
            m.get(QUERY_URL_TASK, payload=_ok_query_payload(status="Failed", file_id=None))
            result = await service.synthesize(text="你好", voice_id="vid")

        assert result["success"] is False
        assert "失败" in result["error"]
    finally:
        await service.close()


async def test_synthesize_task_expired(mock_logger):
    service = MiniMaxAsyncTTSService(api_key="test-key", logger=mock_logger)
    try:
        with aioresponses() as m:
            m.post(CREATE_URL, payload=_ok_create_payload())
            m.get(QUERY_URL_TASK, payload=_ok_query_payload(status="Expired", file_id=None))
            result = await service.synthesize(text="你好", voice_id="vid")

        assert result["success"] is False
        assert "过期" in result["error"]
    finally:
        await service.close()


async def test_synthesize_polling_timeout(monkeypatch, mock_logger):
    """轮询超时：query 始终返回 Processing，poll_max_wait 很小。

    通过 monkeypatch asyncio.sleep 为 no-op 加速，并 patch time.time 控制 elapsed。
    """
    service = MiniMaxAsyncTTSService(
        api_key="test-key", poll_max_wait=0.5, logger=mock_logger
    )
    try:
        # 加速：sleep 立即返回
        async def _no_sleep(_x):
            return None
        monkeypatch.setattr(tts_service.asyncio, "sleep", _no_sleep)

        # 控制 time.time：前若干次返回基准值（保证至少一轮查询），之后跳变触发超时
        call_count = [0]
        real_base = 1000.0

        def fake_time():
            call_count[0] += 1
            # 前 5 次调用返回基准，第 6 次起 +1.0 秒（elapsed=1.0 > poll_max_wait=0.5）
            if call_count[0] <= 5:
                return real_base
            return real_base + 1.0

        monkeypatch.setattr(tts_service.time, "time", fake_time)

        with aioresponses() as m:
            m.post(CREATE_URL, payload=_ok_create_payload())
            # query 始终 Processing，可重复匹配
            m.get(QUERY_URL_TASK, payload=_ok_query_payload(status="Processing", file_id=None), repeat=True)
            result = await service.synthesize(text="你好", voice_id="vid")

        assert result["success"] is False
        err = result["error"].lower()
        assert "timeout" in err or "超时" in result["error"]
    finally:
        await service.close()


# ---------------------------------------------------------------------------
# create_task 重试与错误分类
# ---------------------------------------------------------------------------

async def test_create_task_retryable_retry(monkeypatch, mock_logger):
    """create_task 第一次返回 1002（retryable），第二次成功。"""
    service = MiniMaxAsyncTTSService(
        api_key="test-key", max_retries=2, retry_backoff_base=1.0, logger=mock_logger
    )
    try:
        async def _no_sleep(_x):
            return None
        monkeypatch.setattr(tts_service.asyncio, "sleep", _no_sleep)

        with aioresponses() as m:
            # 第一次 1002 retryable
            m.post(
                CREATE_URL,
                payload={"base_resp": {"status_code": 1002, "status_msg": "retryable err"}},
            )
            # 第二次成功
            m.post(CREATE_URL, payload=_ok_create_payload(task_id=999, file_id=888, usage=20))

            result = await service.create_task(text="hi", voice_id="vid")

        assert "task_id" in result
        assert result["task_id"] == 999
        assert result["file_id"] == 888
        assert result["usage_characters"] == 20
    finally:
        await service.close()


async def test_create_task_fatal_no_retry(monkeypatch, mock_logger):
    """create_task 返回 1004（fatal），立即失败不重试。"""
    service = MiniMaxAsyncTTSService(
        api_key="test-key", max_retries=3, retry_backoff_base=1.0, logger=mock_logger
    )
    try:
        async def _no_sleep(_x):
            return None
        monkeypatch.setattr(tts_service.asyncio, "sleep", _no_sleep)

        with aioresponses() as m:
            m.post(
                CREATE_URL,
                payload={"base_resp": {"status_code": 1004, "status_msg": "fatal err"}},
            )
            result = await service.create_task(text="hi", voice_id="vid")

            # 仅 1 次 POST 请求（无重试）
            post_calls = sum(
                len(calls)
                for (method, _u), calls in m.requests.items()
                if method == "POST"
            )

        assert result["success"] is False
        assert result["code"] == 1004
        assert post_calls == 1
    finally:
        await service.close()


# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------

def test_classify_error(mock_logger):
    service = MiniMaxAsyncTTSService(api_key="test-key", logger=mock_logger)
    for code in (1001, 1002, 1039):
        assert service._classify_error(code) == "retryable", f"code {code} 应为 retryable"
    for code in (1004, 1008, 1042, 2013, 2038):
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
