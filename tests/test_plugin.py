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
