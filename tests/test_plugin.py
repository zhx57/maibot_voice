"""测试插件层 AIVoicePlugin。

由于 plugin.py 依赖 maibot_sdk（外部包），在测试中通过 sys.modules 注入 mock 模块。
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# 在 import plugin 前 mock maibot_sdk
# ---------------------------------------------------------------------------
if "maibot_sdk" not in sys.modules:
    mock_sdk = types.ModuleType("maibot_sdk")
    mock_sdk.MaiBotPlugin = type("MaiBotPlugin", (), {"__init__": lambda self: None})
    mock_sdk.API = lambda *a, **kw: (lambda f: f)
    mock_sdk.Tool = lambda *a, **kw: (lambda f: f)
    mock_sdk.Field = lambda **kw: None
    mock_sdk.PluginConfigBase = type("PluginConfigBase", (), {})
    mock_sdk.CONFIG_RELOAD_SCOPE_SELF = "self"

    mock_types = types.ModuleType("maibot_sdk.types")
    mock_types.ActivationType = type("ActivationType", (), {"ALWAYS": "always"})
    mock_types.ToolParameterInfo = lambda **kw: None
    mock_types.ToolParamType = type("ToolParamType", (), {"STRING": "string"})

    mock_sdk.types = mock_types
    sys.modules["maibot_sdk"] = mock_sdk
    sys.modules["maibot_sdk.types"] = mock_types

from plugin import AIVoicePlugin  # noqa: E402


def _make_plugin() -> AIVoicePlugin:
    """构造一个带 mock ctx/config 的 AIVoicePlugin 实例。"""
    p = AIVoicePlugin()
    p.ctx = MagicMock()
    p.ctx.logger = MagicMock()
    p.ctx.send = MagicMock()
    p.ctx.send.custom = AsyncMock()
    p.config = MagicMock()
    return p


# ---------------------------------------------------------------------------
# _map_style
# ---------------------------------------------------------------------------

def test_map_style_emotion_mapping():
    """_map_style("温柔的语气") 应映射 emotion=calm。"""
    p = _make_plugin()
    emotion, voice_modify = p._map_style("温柔的语气")
    assert emotion == "calm"


def test_map_style_no_match():
    """_map_style("xyz") 无匹配，emotion 空且 voice_modify 空。"""
    p = _make_plugin()
    emotion, voice_modify = p._map_style("xyz")
    assert emotion == ""
    assert voice_modify == {}


# ---------------------------------------------------------------------------
# _resolve_voice (preset 模式)
# ---------------------------------------------------------------------------

def test_resolve_voice_preset():
    """preset 模式应返回 preset_voice。"""
    p = _make_plugin()
    p.default_voice = ""
    p.voices = {}
    p.config.voice.voice_mode = "preset"
    p.config.voice.preset_voice = "PresetVoice123"

    vid = p._resolve_voice("")
    assert vid == "PresetVoice123"


# ---------------------------------------------------------------------------
# _send_voice (关键：base64:// 前缀格式)
# ---------------------------------------------------------------------------

async def test_send_voice_format():
    """关键测试：验证 base64:// 前缀格式正确。

    _send_voice("dGVzdA==", "stream1") 第一次调用参数应为
    ("record", {"file": "base64://dGVzdA=="}, "stream1")。
    """
    p = _make_plugin()
    ok = await p._send_voice("dGVzdA==", "stream1")

    assert ok is True
    # custom 应只被调用一次（record 方式成功后不再尝试 voice）
    p.ctx.send.custom.assert_called_once_with(
        "record", {"file": "base64://dGVzdA=="}, "stream1"
    )


# ---------------------------------------------------------------------------
# voice_clone_tts 集成：mock 服务层 synthesize，断言 _send_voice 收到 base64:// 前缀
# ---------------------------------------------------------------------------

def _configure_voice(p: AIVoicePlugin) -> None:
    """对 mock config 注入对齐新配置模型的字段值。"""
    p.default_voice = ""
    p.voices = {}
    p.config.voice.voice_mode = "preset"
    p.config.voice.preset_voice = "PresetVoice123"
    p.config.voice.model = "speech-2.8-hd"
    p.config.voice.emotion = ""
    p.config.voice.speed = 1.0
    p.config.voice.vol = 1.0
    p.config.voice.pitch = 0
    p.config.voice.text_normalization = False
    p.config.voice.audio_format = "mp3"
    p.config.voice.sample_rate = 32000
    p.config.voice.bitrate = 128000
    p.config.voice.channel = 1
    p.config.voice.language_boost = "auto"
    p.config.voice.aigc_watermark = False
    p.config.voice.subtitle_enable = False
    p.config.voice.subtitle_type = "sentence"
    p.config.voice.latex_read = False


async def test_voice_clone_tts_sends_base64_prefix():
    """mock 服务层 synthesize 返回，断言 voice_clone_tts 调用 _send_voice 时传入 base64:// 前缀。"""
    p = _make_plugin()
    _configure_voice(p)

    p.tts_service = MagicMock()
    p.tts_service.synthesize = AsyncMock(return_value={
        "success": True,
        "audio_base64": "dGVzdA==",
        "format": "mp3",
        "text": "hi",
    })

    result = await p.voice_clone_tts(text="hi", stream_id="stream1")

    assert result["success"] is True
    # _send_voice 应以 base64:// 前缀调用 ctx.send.custom
    p.ctx.send.custom.assert_called_once_with(
        "record", {"file": "base64://dGVzdA=="}, "stream1"
    )
