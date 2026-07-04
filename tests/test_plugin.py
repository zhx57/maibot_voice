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


# ---------------------------------------------------------------------------
# _clean_text_for_tts: 100% 过滤舞台提示 + 100% 保留 MiniMax 原生语气词标签
# ---------------------------------------------------------------------------

def test_clean_text_filter_chinese_stage_direction():
    """中文全角括号包裹的舞台提示应被移除。"""
    p = _make_plugin()
    text = "（轻声，带点困意）哎呀，怎么这么晚还没睡呀……"
    assert p._clean_text_for_tts(text) == "哎呀，怎么这么晚还没睡呀……"


def test_clean_text_filter_english_stage_direction():
    """英文半角括号包裹的非 MiniMax 标签（风格描述）应被移除。"""
    p = _make_plugin()
    text = "(softly) hello there"
    assert p._clean_text_for_tts(text) == "hello there"


def test_clean_text_preserve_minimax_english_tag():
    """英文半角括号的 MiniMax 原生标签应原样保留。"""
    p = _make_plugin()
    text = "今天天气真好(laughs)，当然了"
    assert p._clean_text_for_tts(text) == "今天天气真好(laughs)，当然了"


def test_clean_text_preserve_all_19_minimax_tags():
    """全部 19 个 MiniMax 原生语气词标签均应被 100% 保留（半角括号 + 小写）。"""
    from plugin import MINIMAX_EMOTION_TAGS
    p = _make_plugin()
    for tag in MINIMAX_EMOTION_TAGS:
        text = f"前缀({tag})后缀"
        cleaned = p._clean_text_for_tts(text)
        assert f"({tag})" in cleaned, f"标签 {tag!r} 未被保留: {cleaned!r}"


def test_clean_text_chinese_bracket_tag_to_halfwidth():
    """中文全角括号包裹的 MiniMax 标签应转成英文半角括号保留。"""
    p = _make_plugin()
    text = "今天（sighs）真累"
    assert p._clean_text_for_tts(text) == "今天(sighs)真累"


def test_clean_text_case_insensitive_preserve():
    """大写/混合大小写的 MiniMax 标签应保留并统一为小写。"""
    p = _make_plugin()
    assert p._clean_text_for_tts("(LAUGHS) 哈哈") == "(laughs) 哈哈"
    assert p._clean_text_for_tts("(ChucKle) 呵呵") == "(chuckle) 呵呵"


def test_clean_text_tag_with_inner_spaces():
    """括号内标签带前后空格应正确识别并标准化。"""
    p = _make_plugin()
    assert p._clean_text_for_tts("( sighs )好累") == "(sighs)好累"


def test_clean_text_mixed_tags():
    """舞台提示与 MiniMax 标签混合：移除舞台提示，保留 MiniMax 标签。"""
    p = _make_plugin()
    text = "（温柔）你好（laughs）真开心"
    assert p._clean_text_for_tts(text) == "你好(laughs)真开心"


def test_clean_text_multiple_stage_directions():
    """多个舞台提示连续出现应全部移除，正文保留。"""
    p = _make_plugin()
    text = "（轻声）（困意）终于到家了"
    assert p._clean_text_for_tts(text) == "终于到家了"


def test_clean_text_all_stage_directions_returns_empty():
    """全是舞台提示时返回空字符串（上层应据此报错）。"""
    p = _make_plugin()
    assert p._clean_text_for_tts("（轻声）（困意）") == ""


def test_clean_text_no_brackets_unchanged():
    """无任何括号的文本应原样返回。"""
    p = _make_plugin()
    text = "今天天气真好，我们去散步吧。"
    assert p._clean_text_for_tts(text) == text


def test_clean_text_nested_like_not_matched():
    """含括号的非标签内容（如括号内有英文但非 MiniMax 标签）应被移除。"""
    p = _make_plugin()
    # "thinking" 不在 MINIMAX_EMOTION_TAGS 中，应被移除
    assert p._clean_text_for_tts("(thinking)嗯让我想想") == "嗯让我想想"


def test_clean_text_preserve_tag_at_boundaries():
    """MiniMax 标签位于文本首尾时应保留。"""
    p = _make_plugin()
    assert p._clean_text_for_tts("(sighs)好累") == "(sighs)好累"
    assert p._clean_text_for_tts("好累(sighs)") == "好累(sighs)"


# ---------------------------------------------------------------------------
# _clean_text_for_tts: 模型兼容性（仅 speech-2.8 系列保留语气词标签）
# ---------------------------------------------------------------------------

def test_clean_text_28_hd_preserves_tags():
    """speech-2.8-hd 支持 19 个原生语气词标签，应保留。"""
    p = _make_plugin()
    assert p._clean_text_for_tts("(laughs)哈哈", model="speech-2.8-hd") == "(laughs)哈哈"
    assert p._clean_text_for_tts("好累(sighs)", model="speech-2.8-hd") == "好累(sighs)"


def test_clean_text_28_turbo_preserves_tags():
    """speech-2.8-turbo 也支持，应保留。"""
    p = _make_plugin()
    assert p._clean_text_for_tts("(humming)哼唱", model="speech-2.8-turbo") == "(humming)哼唱"


@pytest.mark.parametrize("model", [
    "speech-2.6-hd", "speech-2.6-turbo",
    "speech-02-hd", "speech-02-turbo",
    "speech-01-hd", "speech-01-turbo",
])
def test_clean_text_non_28_models_strip_tags(model):
    """非 2.8 系列模型不支持原生语气词标签，应移除（否则会被当文本念出）。"""
    p = _make_plugin()
    assert p._clean_text_for_tts("(laughs)哈哈", model=model) == "哈哈"
    assert p._clean_text_for_tts("好累(sighs)", model=model) == "好累"


def test_clean_text_non_28_model_all_19_tags_stripped():
    """非 2.8 模型下，全部 19 个语气词标签都应被移除。"""
    from plugin import MINIMAX_EMOTION_TAGS
    p = _make_plugin()
    for tag in MINIMAX_EMOTION_TAGS:
        text = f"前缀({tag})后缀"
        cleaned = p._clean_text_for_tts(text, model="speech-2.6-hd")
        assert f"({tag})" not in cleaned, f"2.6 模型下标签 {tag!r} 应被移除: {cleaned!r}"
        assert "前缀" in cleaned and "后缀" in cleaned


def test_clean_text_non_28_model_chinese_bracket_tag_stripped():
    """非 2.8 模型下，中文全角括号包裹的语气词标签也应被移除。"""
    p = _make_plugin()
    assert p._clean_text_for_tts("今天（sighs）真累", model="speech-02-hd") == "今天真累"


def test_clean_text_non_28_model_mixed_with_stage_direction():
    """非 2.8 模型下，舞台提示和语气词标签都应被移除。"""
    p = _make_plugin()
    text = "（温柔）你好（laughs）真开心"
    assert p._clean_text_for_tts(text, model="speech-2.6-turbo") == "你好真开心"


def test_clean_text_non_28_model_all_tags_only_returns_empty():
    """非 2.8 模型下，文本全是语气词标签时应返回空。"""
    p = _make_plugin()
    assert p._clean_text_for_tts("(laughs)(sighs)", model="speech-2.6-hd") == ""


def test_clean_text_empty_model_defaults_to_preserve():
    """model 参数为空时默认保留标签（向后兼容，便于无 config 上下文调用）。"""
    p = _make_plugin()
    assert p._clean_text_for_tts("(laughs)哈哈") == "(laughs)哈哈"
    assert p._clean_text_for_tts("(laughs)哈哈", model="") == "(laughs)哈哈"


# ---------------------------------------------------------------------------
# voice_clone_tts 集成：过滤流程贯穿到 synthesize 调用
# ---------------------------------------------------------------------------

async def test_voice_clone_tts_cleans_text_before_synthesize():
    """voice_clone_tts 应将清洗后的文本传给 synthesize，舞台提示不进入 TTS。"""
    p = _make_plugin()
    _configure_voice(p)

    captured: dict = {}
    p.tts_service = MagicMock()
    async def _fake_synthesize(**kwargs):
        captured["text"] = kwargs.get("text", "")
        return {"success": True, "audio_base64": "dGVzdA==", "format": "mp3", "text": kwargs.get("text", "")}
    p.tts_service.synthesize = _fake_synthesize

    result = await p.voice_clone_tts(
        text="（轻声，带点困意）哎呀(laughs)，怎么这么晚还没睡呀……",
        stream_id="stream1",
    )

    assert result["success"] is True
    # 舞台提示被移除，MiniMax 原生 (laughs) 标签被保留
    assert captured["text"] == "哎呀(laughs)，怎么这么晚还没睡呀……"
    assert "轻声" not in captured["text"]
    assert "困意" not in captured["text"]
    assert "(laughs)" in captured["text"]


async def test_voice_clone_tts_empty_after_cleaning_returns_error():
    """清洗后文本为空时应返回明确错误，且不调用 synthesize。"""
    p = _make_plugin()
    _configure_voice(p)

    p.tts_service = MagicMock()
    p.tts_service.synthesize = AsyncMock()

    result = await p.voice_clone_tts(text="（轻声）（困意）", stream_id="stream1")

    assert result["success"] is False
    assert "empty" in result["error"].lower()
    p.tts_service.synthesize.assert_not_called()
