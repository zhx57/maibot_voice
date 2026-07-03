"""pytest-asyncio 配置与公共 fixture。"""

import functools
import inspect
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# 将插件根目录（tests 的父目录）加入 sys.path，便于直接 import tts_service / plugin
_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)


# ---------------------------------------------------------------------------
# 兼容性 shim：aiohttp 3.14+ 给 ClientResponse.__init__ 增加了必需的
# keyword-only 参数 stream_writer，而 aioresponses 0.7.9 未传递该参数，
# 导致 mock 响应构造失败。这里在缺失时注入 None 以恢复兼容。
# ---------------------------------------------------------------------------
def _patch_client_response_for_aioresponses() -> None:
    import aiohttp

    sig = inspect.signature(aiohttp.ClientResponse.__init__)
    sw_param = sig.parameters.get("stream_writer")
    if sw_param is None or sw_param.default is not inspect._empty:
        return  # 无需 patch

    _orig_init = aiohttp.ClientResponse.__init__

    @functools.wraps(_orig_init)
    def _patched_init(self, *args, **kwargs):
        # aioresponses 传 writer=None，此时 aiohttp 会访问 stream_writer.output_size，
        # 因此注入一个带 output_size 属性的 mock，避免 AttributeError。
        sw = kwargs.get("stream_writer")
        if sw is None:
            sw = MagicMock()
            sw.output_size = 0
            kwargs["stream_writer"] = sw
        return _orig_init(self, *args, **kwargs)

    aiohttp.ClientResponse.__init__ = _patched_init


_patch_client_response_for_aioresponses()


@pytest.fixture
def mock_logger():
    """返回一个 mock 的 logging.Logger。"""
    logger = MagicMock(spec=logging.Logger)
    return logger


@pytest.fixture
def sample_mp3_bytes():
    """返回一段合法的 mp3 字节（ID3 头 + 少量数据）。"""
    return b'ID3' + b'\x03\x00\x00\x00\x00\x00\x00' + b'\xff\xfb' + b'\x00' * 100


@pytest.fixture
def sample_wav_bytes():
    """返回一段合法的 wav 字节。"""
    return (
        b'RIFF' + b'\x00' * 4 + b'WAVE'
        + b'fmt ' + b'\x00' * 24
        + b'data' + b'\x00' * 4 + b'\x00' * 100
    )


@pytest.fixture
def voices_dir(tmp_path):
    """临时 voices 目录 fixture。"""
    vdir = tmp_path / "voices"
    vdir.mkdir(parents=True, exist_ok=True)
    return vdir
