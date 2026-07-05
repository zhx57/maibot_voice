"""AI Voice Service Plugin for MaiBot."""

import asyncio
import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from maibot_sdk import API, CONFIG_RELOAD_SCOPE_SELF, MaiBotPlugin, Tool, Field, PluginConfigBase
from maibot_sdk.types import ActivationType, ToolParameterInfo, ToolParamType

try:
    from .tts_service import MiniMaxSyncTTSService, VoiceCloneManager, generate_voice_id_from_filename
except ImportError:
    from tts_service import MiniMaxSyncTTSService, VoiceCloneManager, generate_voice_id_from_filename


class PluginSectionConfig(PluginConfigBase):
    __ui_label__ = "插件"
    enabled: bool = Field(default=True, description="是否启用")
    config_version: str = Field(default="3.0.0", description="配置版本")


class VoiceSectionConfig(PluginConfigBase):
    __ui_label__ = "语音设置"
    minimax_api_key: str = Field(default="", description="MiniMax API Key")
    api_base_url: str = Field(default="https://api.minimaxi.com", description="MiniMax API地址", json_schema_extra={"label": "API地址"})
    model: str = Field(default="speech-2.8-hd", description="TTS模型，如 speech-2.8-hd / speech-2.6-turbo / speech-02-hd")
    voice_mode: str = Field(default="clone", description="语音模式: 'clone'(音色复刻) 或 'preset'(预置音色)")
    preset_voice: str = Field(default="English_expressive_narrator", description="预置音色voice_id（仅preset模式）")
    voices_dir: str = Field(default="voices", description="音色目录路径（相对于插件目录）")
    default_voice: str = Field(default="", description="默认音色名称（clone模式下为音频文件名）")
    clone_voice: str = Field(default="", description="复刻音色文件名（clone模式下优先使用）")
    emotion: str = Field(default="", description="情绪枚举 happy/sad/angry/fearful/disgusted/surprised/calm/fluent/whisper，留空自动；fluent/whisper 仅 speech-2.6 系列支持")
    speed: float = Field(default=1.0, description="语速 [0.5, 2]")
    vol: float = Field(default=1.0, description="音量 (0, 10]")
    pitch: int = Field(default=0, description="音调 [-12, 12] 整数，0 为原音色")
    text_normalization: bool = Field(default=False, description="中英文文本归一化")
    audio_format: str = Field(default="mp3", description="音频格式: mp3/pcm/flac/wav/pcmu_raw/pcmu_wav/opus")
    sample_rate: int = Field(default=32000, description="采样率 [8000,16000,22050,24000,32000,44100]")
    bitrate: int = Field(default=128000, description="比特率 [32000,64000,128000,256000]，仅 mp3 生效")
    channel: int = Field(default=1, description="声道 [1,2]，1 单声道（默认，适合语音消息）/2 双声道")
    language_boost: str = Field(default="auto", description="语种识别增强 auto 或具体语种")
    aigc_watermark: bool = Field(default=False, description="AIGC 水印（仅非流式生效）")
    subtitle_enable: bool = Field(default=False, description="是否开启字幕服务")
    subtitle_type: str = Field(default="sentence", description="字幕粒度: sentence/word/word_streaming")
    latex_read: bool = Field(default=False, description="是否朗读 LaTeX 公式（仅中文，开启后 language_boost 强制 Chinese）")
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_backoff_base: float = Field(default=1.5, description="重试退避基数")


class VoicePluginConfig(PluginConfigBase):
    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    voice: VoiceSectionConfig = Field(default_factory=VoiceSectionConfig)


# MiniMax 同步 TTS 原生语气词标签（speech-2.8 系列）
# 来源: 官方 /v1/t2a_v2 OpenAPI -> T2aV2Req.text 字段描述
# 这些标签会被 TTS 引擎识别为语气/声音动作渲染，必须 100% 保留
MINIMAX_EMOTION_TAGS: frozenset[str] = frozenset({
    "laughs", "chuckle", "coughs", "clear-throat", "groans",
    "breath", "pant", "inhale", "exhale", "gasps", "sniffs",
    "sighs", "snorts", "burps", "lip-smacking", "humming",
    "hissing", "emm", "sneezes",
})

# 支持原生语气词标签的模型（仅 speech-2.8 系列）
# 其他模型（2.6 / 02 / 01）不支持，标签会被当作普通文本朗读，必须移除
EMOTION_TAG_SUPPORTING_MODELS: frozenset[str] = frozenset({
    "speech-2.8-hd", "speech-2.8-turbo",
})

# 匹配中文全角（）和英文半角() 括号内的非嵌套内容
_TAG_PATTERN = re.compile(r"[（(]([^()（）]*?)[)）]")

# 匹配 XML/HTML 标签字面化残留（LLM 把 Tool 调用标签泄漏到正文）
# 例: "<reply_text>" / "</reply_text>" / "小于reply text>parameter" / "小于reply_text大于"
# 要求标签名以英文字母开头，避免误伤 "3小于4" "你小于我大于他" 等正常中文
_TAG_LITERAL_PATTERN = re.compile(
    r"(?:小于|<)\s*/?\s*[a-zA-Z][\w\s_.-]*?\s*(?:>|大于)"
    r"(?:\s*parameter)?"
)


class AIVoicePlugin(MaiBotPlugin):
    config_model = VoicePluginConfig

    def __init__(self) -> None:
        super().__init__()
        self.tts_service: Optional[MiniMaxSyncTTSService] = None
        self.voice_clone_mgr: Optional[VoiceCloneManager] = None
        self.voices: dict[str, str] = {}
        self.default_voice: str = ""
        self._voice_cache: dict[str, dict] = {}

    async def on_load(self) -> None:
        self._ensure_config_exists()
        self.ctx.logger.info("AI Voice Plugin loading...")
        api_key = self.config.voice.minimax_api_key
        if not api_key:
            self.ctx.logger.warning("MiniMax API Key not configured, TTS will be disabled")
        else:
            self.tts_service = MiniMaxSyncTTSService(
                api_key=api_key,
                api_base_url=self.config.voice.api_base_url,
                model=self.config.voice.model,
                max_retries=self.config.voice.max_retries,
                retry_backoff_base=self.config.voice.retry_backoff_base,
                logger=self.ctx.logger,
            )
            self.voice_clone_mgr = VoiceCloneManager(
                api_key=api_key,
                api_base_url=self.config.voice.api_base_url,
                logger=self.ctx.logger,
            )
        self.default_voice = self.config.voice.default_voice or self.config.voice.clone_voice
        await self._load_voices()
        self.ctx.logger.info("AI Voice Plugin loaded: mode=%s, default_voice=%s, voices=%s",
            self.config.voice.voice_mode, self.default_voice, list(self.voices.keys()))

    def _ensure_config_exists(self) -> None:
        """如果用户目录下不存在 config.toml，则从 config.example.toml 复制生成。不会覆盖用户已有配置。"""
        import shutil
        plugin_dir = Path(__file__).parent
        config_path = plugin_dir / "config.toml"
        example_path = plugin_dir / "config.example.toml"
        if config_path.exists():
            return
        if example_path.exists():
            shutil.copy2(example_path, config_path)
            self.ctx.logger.info("Generated config.toml from config.example.toml")
        else:
            self.ctx.logger.warning("Neither config.toml nor config.example.toml found")

    async def on_unload(self) -> None:
        if self.tts_service:
            await self.tts_service.close()
            self.tts_service = None
        if self.voice_clone_mgr:
            await self.voice_clone_mgr.close()
            self.voice_clone_mgr = None
        self.voices.clear()
        self._voice_cache = {}

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        if scope == CONFIG_RELOAD_SCOPE_SELF:
            self.ctx.logger.info("Plugin config updated: version=%s", version)
            api_key = self.config.voice.minimax_api_key
            api_base_url = self.config.voice.api_base_url
            model = self.config.voice.model
            if not api_key:
                self.ctx.logger.warning("MiniMax API Key not configured after update, TTS disabled")
                if self.tts_service:
                    await self.tts_service.close()
                    self.tts_service = None
                if self.voice_clone_mgr:
                    await self.voice_clone_mgr.close()
                    self.voice_clone_mgr = None
            else:
                if self.tts_service:
                    self.tts_service.update_api_key(api_key)
                    self.tts_service.update_api_base_url(api_base_url)
                    self.tts_service.update_model(model)
                else:
                    self.tts_service = MiniMaxSyncTTSService(
                        api_key=api_key,
                        api_base_url=api_base_url,
                        model=model,
                        max_retries=self.config.voice.max_retries,
                        retry_backoff_base=self.config.voice.retry_backoff_base,
                        logger=self.ctx.logger,
                    )
                # VoiceCloneManager 没有 update 方法，重建
                if self.voice_clone_mgr:
                    await self.voice_clone_mgr.close()
                self.voice_clone_mgr = VoiceCloneManager(
                    api_key=api_key,
                    api_base_url=api_base_url,
                    logger=self.ctx.logger,
                )
            self.default_voice = self.config.voice.default_voice or self.config.voice.clone_voice
            await self._load_voices()
            self.ctx.logger.info("Config reloaded: mode=%s, default_voice=%s", self.config.voice.voice_mode, self.default_voice)

    async def _load_voices(self) -> None:
        voices_dir_str = self.config.voice.voices_dir
        plugin_dir = Path(__file__).parent
        voices_dir = plugin_dir / voices_dir_str
        self.ctx.logger.info("Loading voices from: %s (plugin_dir=%s)", voices_dir, plugin_dir)

        if not voices_dir.exists():
            self.ctx.logger.warning("Voices dir not found: %s, creating it", voices_dir)
            voices_dir.mkdir(parents=True, exist_ok=True)
            return

        self.ctx.logger.info("Voices dir exists: %s, contents: %s", voices_dir, [f.name for f in voices_dir.iterdir()])

        audio_files = list(voices_dir.glob("*.wav")) + list(voices_dir.glob("*.mp3"))
        if not audio_files:
            self.ctx.logger.warning("No audio files found in: %s", voices_dir)
            return

        # 读取本地缓存
        cache_path = voices_dir / ".voice_cache.json"
        self._voice_cache = self._read_voice_cache(cache_path)
        self.voices.clear()
        for audio_file in audio_files:
            voice_name = audio_file.stem
            file_mtime = audio_file.stat().st_mtime
            file_size = audio_file.stat().st_size
            cached = self._voice_cache.get(voice_name)
            # 缓存有效则复用
            if cached and cached.get("file_mtime") == file_mtime and cached.get("file_size") == file_size and cached.get("voice_id"):
                self.voices[voice_name] = cached["voice_id"]
                self.ctx.logger.info("Reusing cached voice: %s -> voice_id=%s", voice_name, cached["voice_id"])
                continue
            # 需要注册
            if not self.config.voice.minimax_api_key:
                self.ctx.logger.warning("Cannot register voice '%s': API Key not configured", voice_name)
                continue
            if not self.voice_clone_mgr:
                self.ctx.logger.warning("Cannot register voice '%s': VoiceCloneManager not initialized", voice_name)
                continue
            try:
                self.ctx.logger.info("Registering clone voice: %s", voice_name)
                file_id = await self.voice_clone_mgr.upload_audio(str(audio_file))
                voice_id = generate_voice_id_from_filename(voice_name)
                reg_result = await self.voice_clone_mgr.register_clone_voice(file_id, voice_id)
                if reg_result.get("success"):
                    self.voices[voice_name] = voice_id
                    self._voice_cache[voice_name] = {"voice_id": voice_id, "file_id": file_id, "created_at": time.time(), "file_mtime": file_mtime, "file_size": file_size}
                    self.ctx.logger.info("Clone voice registered: %s -> voice_id=%s, file_id=%s", voice_name, voice_id, file_id)
                else:
                    self.ctx.logger.error("Failed to register clone voice '%s': %s (code=%s)", voice_name, reg_result.get("error"), reg_result.get("code"))
            except Exception as e:
                self.ctx.logger.error("Exception registering clone voice '%s': %s", voice_name, e)
        # 写回缓存
        self._write_voice_cache(cache_path, self._voice_cache)
        if not self.default_voice and self.voices:
            self.default_voice = next(iter(self.voices))
            self.ctx.logger.info("Using default voice: %s", self.default_voice)
        self.ctx.logger.info("Voice loading complete, total %d voices, names=%s", len(self.voices), list(self.voices.keys()))

    def _read_voice_cache(self, path: Path) -> dict[str, dict]:
        """读取本地音色缓存 JSON。异常时返回空 dict。"""
        try:
            if not path.exists():
                return {}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            self.ctx.logger.warning("Voice cache invalid format (not dict): %s", path)
            return {}
        except Exception as e:
            self.ctx.logger.warning("Failed to read voice cache '%s': %s", path, e)
            return {}

    def _write_voice_cache(self, path: Path, cache: dict[str, dict]) -> None:
        """写回本地音色缓存 JSON。异常时记录警告。"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.ctx.logger.warning("Failed to write voice cache '%s': %s", path, e)

    async def _find_stream_id(self, kwargs: dict) -> str:
        """Find stream_id from various sources."""
        # Method 1: from kwargs (system may pass it)
        sid = kwargs.get("stream_id", "")
        if sid:
            self.ctx.logger.info("Got stream_id from kwargs: %s", sid)
            return str(sid)

        # Method 2: from kwargs message
        msg = kwargs.get("message", {})
        if isinstance(msg, dict):
            sid = msg.get("stream_id", "")
            if sid:
                self.ctx.logger.info("Got stream_id from message: %s", sid)
                return str(sid)

        # Method 3: try to find group stream
        try:
            streams = await self.ctx.chat.get_group_streams()
            if streams:
                sid = streams[0].get("stream_id", "")
                self.ctx.logger.info("Using first group stream: %s", sid)
                return str(sid)
        except Exception as e:
            self.ctx.logger.warning("chat.get_group_streams failed: %s", e)

        return ""

    async def _send_voice(self, audio_b64: str, stream_id: str) -> bool:
        """Send voice message using base64 format.
        注意: 调用后 audio_b64 会被消费，不再可用。
        """
        self.ctx.logger.info("Sending voice to stream=%s, b64_len=%d", stream_id, len(audio_b64))

        b64_url = f"base64://{audio_b64}"
        # 原始引用可以释放（如果调用方已 del 则无额外效果，但作为安全措施）
        audio_b64 = ""

        # Method 1: record type with base64 (NapCat standard for voice)
        try:
            await self.ctx.send.custom("record", {"file": b64_url}, stream_id)
            self.ctx.logger.info("Sent via record+base64")
            return True
        except Exception as e:
            self.ctx.logger.warning("record+base64 failed: %s", e)

        # Method 2: voice type with base64
        try:
            await self.ctx.send.custom("voice", {"file": b64_url}, stream_id)
            self.ctx.logger.info("Sent via voice+base64")
            return True
        except Exception as e:
            self.ctx.logger.warning("voice+base64 failed: %s", e)

        self.ctx.logger.error("All voice send methods failed")
        return False

    def _resolve_voice(self, voice_name: str) -> str:
        """解析音色名称，返回 MiniMax voice_id。返回空字符串表示失败。"""
        voice_mode = self.config.voice.voice_mode
        voice_key = voice_name if voice_name and voice_name != "default" else self.default_voice
        if voice_mode == "clone":
            if voice_key and voice_key in self.voices:
                return self.voices[voice_key]
            clone_fallback = self.config.voice.clone_voice
            if clone_fallback and clone_fallback in self.voices:
                self.ctx.logger.warning("Voice '%s' not found, using clone_voice config: '%s'", voice_key, clone_fallback)
                return self.voices[clone_fallback]
            if self.voices:
                available = list(self.voices.keys())
                self.ctx.logger.error("Voice '%s' not found in voices! Available: %s.", voice_key, available)
                return ""
            self.ctx.logger.error("No voices loaded! Please put .wav/.mp3 files in the voices/ directory.")
            return ""
        # preset 模式
        preset_id = voice_key or self.config.voice.preset_voice or "English_expressive_narrator"
        return preset_id

    def _clean_text_for_tts(self, text: str, model: str = "") -> str:
        """过滤 reply_text 中的舞台提示/风格标签，100% 保留 MiniMax 原生语气词标签。

        规则：
        - 先移除 XML/HTML 标签字面化残留（如 "<reply_text>" / "小于reply text>parameter"）
          这类是 LLM 把 Tool 调用标签泄漏到正文，必须移除避免被念出
        - 扫描所有中文全角（）和英文半角() 括号内容
        - 括号内文本 strip 后小写，若在 MINIMAX_EMOTION_TAGS 集合：
          - 当 model 属于 speech-2.8 系列 → 转成英文半角括号保留（TTS 引擎会渲染为语气/声音动作）
          - 当 model 不支持（2.6 / 02 / 01 系列）→ 移除（否则会被当普通文本朗读，如念出"laughs"）
          - model 为空时默认保留（向后兼容）
        - 否则视为舞台提示/风格标签（如"轻声""困意""东北话"）→ 移除
        - 清理移除后留下的多余空白与孤立标点
        """
        # 1. 移除标签字面化残留（LLM 泄漏的 Tool 调用标签）
        text = _TAG_LITERAL_PATTERN.sub("", text)

        supports_tags = (not model) or (model in EMOTION_TAG_SUPPORTING_MODELS)

        def _replace(m: re.Match) -> str:
            inner = m.group(1).strip()
            if inner.lower() in MINIMAX_EMOTION_TAGS:
                if not supports_tags:
                    return ""  # 当前模型不支持语气词标签，移除
                # 统一转成英文半角括号 + 小写（MiniMax 原生格式）
                return f"({inner.lower()})"
            return ""  # 舞台提示，移除

        cleaned = _TAG_PATTERN.sub(_replace, text)
        # 清理多余空白
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # 清理移除后可能留下的孤立标点（如 "，。" "， " " 、"）
        cleaned = re.sub(r"[，,、]\s*[。.]", "。", cleaned)
        cleaned = re.sub(r"^[，,、\s]+", "", cleaned)
        cleaned = re.sub(r"\s+[，,、]+", "，", cleaned)
        return cleaned

    def _map_style(self, style_instruction: str) -> tuple[str, dict]:
        """将 style_instruction 自然语言映射为 (emotion, voice_modify)。
        返回 (emotion字符串可能为空, voice_modify字典)。
        注意 emotion 兼容性：fluent/whisper 仅 speech-2.6 系列支持，
        speech-2.8 系列不支持 whisper，故对不兼容的情绪降级为空（让模型自动匹配）。"""
        emotion = ""
        voice_modify: dict[str, Any] = {}
        if not style_instruction:
            return emotion, voice_modify
        text = style_instruction.lower()
        model = self.config.voice.model
        # whisper 仅 speech-2.6-turbo / speech-2.6-hd 支持
        whisper_supported = model in ("speech-2.6-turbo", "speech-2.6-hd")
        # fluent 仅 speech-2.6-turbo / speech-2.6-hd 支持
        fluent_supported = model in ("speech-2.6-turbo", "speech-2.6-hd")
        # emotion 映射（关键词扩展，覆盖 LLM 常用同义词）
        emotion_map = [
            (["温柔", "安慰", "温暖", "平静", "柔和", "舒缓", "轻柔", "calm", "gentle", "soft", "tender", "mild", "soothing"], "calm"),
            (["俏皮", "活泼", "开心", "快乐", "高兴", "愉悦", "兴奋", "欢喜", "欣喜", "欢快", "happy", "cheerful", "lively", "joyful", "excited", "delighted"], "happy"),
            (["悲伤", "难过", "失落", "伤感", "哀伤", "忧郁", "沮丧", "心痛", "sad", "sorrow", "depressed", "melancholy", "down"], "sad"),
            (["愤怒", "生气", "恼怒", "不满", "气愤", "恼火", "发火", "angry", "mad", "furious", "irritated", "annoyed"], "angry"),
            (["害怕", "恐惧", "紧张", "焦虑", "惊慌", "不安", "畏惧", "fear", "afraid", "nervous", "anxious", "scared", "terrified"], "fearful"),
            (["厌恶", "嫌弃", "反感", "鄙视", "恶心", "disgust", "disgusted", "contempt", "repulsed"], "disgusted"),
            (["惊讶", "惊喜", "意外", "震惊", "吃惊", "错愕", "surprise", "surprised", "astonished", "shocked"], "surprised"),
            (["低语", "耳语", "悄悄话", "whisper"], "whisper" if whisper_supported else ""),
            (["流畅", "自然", "平淡", "fluent", "natural"], "fluent" if fluent_supported else ""),
        ]
        for keywords, emo in emotion_map:
            if any(k in text for k in keywords):
                emotion = emo
                break
        # voice_modify 映射
        pitch = 0
        timbre = 0
        intensity = 0
        if any(k in text for k in ["低沉", "低音", "deep", "low"]):
            pitch = -20
        if any(k in text for k in ["清甜", "清脆", "明亮", "bright", "crisp"]):
            timbre = 20
        if any(k in text for k in ["有力", "强势", "strong", "powerful"]):
            intensity = -20
        if any(k in text for k in ["柔和", "轻柔", "gentle", "tender"]):
            intensity = 20
        sound_effects = ""
        if "空旷" in text or "spacious" in text or "echo" in text:
            sound_effects = "spacious_echo"
        elif "广播" in text or "auditorium" in text:
            sound_effects = "auditorium_echo"
        elif "电话" in text or "telephone" in text:
            sound_effects = "lofi_telephone"
        elif "机械" in text or "robot" in text:
            sound_effects = "robotic"
        if pitch or timbre or intensity or sound_effects:
            voice_modify = {"pitch": pitch, "timbre": timbre, "intensity": intensity}
            if sound_effects:
                voice_modify["sound_effects"] = sound_effects
        return emotion, voice_modify

    @API("voice_clone_tts", description="TTS with specified voice", version="1", public=True)
    async def voice_clone_tts(self, text: str, style_instruction: str = "", stream_id: str = "", voice_name: str = "") -> dict[str, Any]:
        if not self.tts_service:
            return {"success": False, "error": "TTS service not initialized"}

        if not text or not text.strip():
            return {"success": False, "error": "Empty text"}

        # 过滤舞台提示/风格标签，100% 保留 MiniMax 原生语气词标签（如 (laughs)/(sighs)/(humming)）
        # 注意: 仅 speech-2.8 系列支持原生语气词标签，其他模型会移除标签避免被当文本朗读
        original_len = len(text)
        text = self._clean_text_for_tts(text, model=self.config.voice.model)
        if not text.strip():
            return {"success": False, "error": "Text is empty after cleaning stage directions"}
        if len(text) != original_len:
            self.ctx.logger.info("Cleaned stage directions: %d -> %d chars", original_len, len(text))

        voice_id = self._resolve_voice(voice_name)
        if not voice_id:
            return {"success": False, "error": "No voice configured. Check voice config."}

        self.ctx.logger.info("TTS request: voice_id=%s, text_len=%d, style_len=%d", voice_id, len(text), len(style_instruction))

        try:
            # 风格映射
            emotion, voice_modify = self._map_style(style_instruction)
            # LLM 动态指令优先于配置默认值（配置作 fallback）
            final_emotion = emotion or self.config.voice.emotion
            # emotion 模型兼容性校验：fluent/whisper 仅 speech-2.6 系列支持
            if final_emotion in ("fluent", "whisper") and self.config.voice.model not in ("speech-2.6-turbo", "speech-2.6-hd"):
                self.ctx.logger.warning("emotion '%s' not supported by model '%s', dropping (need speech-2.6 series)", final_emotion, self.config.voice.model)
                final_emotion = ""
            # 组装 voice_setting（对照官方 T2aV2VoiceSetting）
            voice_setting: dict[str, Any] = {
                "voice_id": voice_id,
                "speed": self.config.voice.speed,
                "vol": self.config.voice.vol,
                "pitch": self.config.voice.pitch,
            }
            if final_emotion:
                voice_setting["emotion"] = final_emotion
            if self.config.voice.text_normalization:
                voice_setting["text_normalization"] = True
            # 组装 audio_setting（对照官方 T2aV2AudioSetting）
            audio_setting: dict[str, Any] = {
                "sample_rate": self.config.voice.sample_rate,
                "bitrate": self.config.voice.bitrate,
                "format": self.config.voice.audio_format,
                "channel": self.config.voice.channel,
            }
            # voice_modify 兼容性校验：仅 mp3/wav/flac 支持（pcm/pcmu_*/opus 不支持）
            if voice_modify and self.config.voice.audio_format not in ("mp3", "wav", "flac"):
                self.ctx.logger.warning("voice_modify ignored: format %s not supported (need mp3/wav/flac)", self.config.voice.audio_format)
                voice_modify = {}
            result = await self.tts_service.synthesize(
                text=text,
                voice_id=voice_id,
                voice_setting=voice_setting,
                audio_setting=audio_setting,
                language_boost=self.config.voice.language_boost or None,
                voice_modify=voice_modify or None,
                aigc_watermark=self.config.voice.aigc_watermark,
                subtitle_enable=self.config.voice.subtitle_enable,
                subtitle_type=self.config.voice.subtitle_type,
                latex_read=self.config.voice.latex_read,
            )
            success = result.get("success")
            error = result.get("error", "")
            audio_b64 = result.get("audio_base64", "")
            self.ctx.logger.info("TTS result: success=%s, audio_len=%d, error=%s", success, len(audio_b64), error)
            if success and audio_b64 and stream_id:
                # 提取纯 base64（service 返回的已是纯 base64，但做防御性处理）
                if "base64," in audio_b64:
                    audio_b64 = audio_b64.split("base64,", 1)[1].strip()
                result.pop("audio_base64", None)
                await self._send_voice(audio_b64, stream_id)
                audio_b64 = ""
            elif audio_b64:
                result.pop("audio_base64", None)
            return result
        except Exception as e:
            self.ctx.logger.error("TTS failed: %s", e)
            return {"success": False, "error": str(e)}

    @Tool(
        "send_voice_reply",
        brief_description="使用语音回复用户，可选传入风格指令控制语气情绪",
        detailed_description=(
            "使用语音进行回复。当用户要求语音、用户发送了语音消息、或你认为当前场景适合语音回复时调用。\n"
            "必填参数：reply_text（回复文本）、msg_id（当前消息ID）。\n"
            "可选参数：style_instruction（语音风格指令），不传则使用默认音色自然朗读，需要更生动的表达时可填写。\n\n"
            "【风格控制】可通过 style_instruction 参数精细控制语音演绎效果，"
            "根据你当前扮演的角色人设和对话情境，主动提供完整的风格描述，让语音更拟人、更有感染力。\n\n"
            "支持三种风格控制方式，可自由组合：\n\n"
            "1. 自然语言风格指令（推荐）：用自然语言描述语气、情绪、语速等，像给演员说戏一样。\n"
            "   示例：'一位温柔的少女，声音清甜软糯，语速偏慢，用安慰的语气，带点关切'\n"
            "   示例：'语气俏皮活泼，带点小得意，语速偏快，声音明亮有活力'\n"
            "   示例：'声音低沉严肃，像在教训人，语速慢一些，带点长辈的威严'\n\n"
            "2. 导演模式（高级）：从角色、场景、指导三个维度全方位刻画表演，适合需要高度拟人化的场景。\n"
            "   示例：'角色：一位温柔的大姐姐，性格体贴温暖，声音甜美有亲和力。"
            "场景：安慰失恋的朋友。指导：语调柔和温暖，气息松弛，偶尔带叹息，语速偏慢，尾音上扬带笑意。'\n"
            "   示例：'角色：百年门阀的大大小姐，声音冷冽有威压，说话语速极慢，每个字都像在舌尖滚过。"
            "场景：在祠堂面对企图带她私奔的男人。指导：实音重且硬，尾音处加入轻微气音透出疲惫。'\n\n"
            "3. 音频标签（在 reply_text 中使用）：在文本任意位置用括号标注语气/情绪/声音动作。\n"
            "   中文全角/半角均可：（紧张）呼……冷静。（叹气）算了。（轻笑）好吧好吧。\n"
            "   英文：(sighs) I don't know. (laughs) That's funny!\n"
            "   整段风格标签：在文本开头加（温柔）你好呀~ 或（东北话）哎呀妈呀~ 或（唱歌）歌词...\n"
            "   常用风格标签：开心/悲伤/愤怒/温柔/慵懒/俏皮/磁性/沙哑/甜美/冷漠/严肃/活泼/深沉 等\n"
            "   常用动作标签：叹气/轻笑/哽咽/深呼吸/咳嗽/打哈欠/低语/提高音量 等\n\n"
            "提示：style_instruction（整体风格）+ reply_text 中的音频标签（句内细节）可同时使用，两者不冲突。\n\n"
            "【情绪控制 - 重要】style_instruction 中包含特定关键词会触发对应情绪渲染，请根据对话情境主动判断是否使用：\n"
            "- calm（温和平静）：温柔/安慰/温暖/平静/柔和/舒缓/轻柔 —— 安慰人、温柔说话时用\n"
            "- happy（高兴）：开心/快乐/高兴/愉悦/兴奋/欢喜/欣喜/欢快/俏皮/活泼 —— 高兴、兴奋、俏皮时用\n"
            "- sad（悲伤）：悲伤/难过/失落/伤感/哀伤/忧郁/沮丧/心痛 —— 难过、失落、伤感时用\n"
            "- angry（愤怒）：愤怒/生气/恼怒/不满/气愤/恼火/发火 —— 生气、训斥、不满时用\n"
            "- fearful（害怕）：害怕/恐惧/紧张/焦虑/惊慌/不安/畏惧 —— 紧张、害怕、焦虑时用\n"
            "- disgusted（厌恶）：厌恶/嫌弃/反感/鄙视/恶心 —— 嫌弃、反感时用\n"
            "- surprised（惊讶）：惊讶/惊喜/意外/震惊/吃惊/错愕 —— 惊讶、意外、惊喜时用\n\n"
            "情绪使用原则：\n"
            "1. 根据场景判断，不必每次都加情绪。日常平淡对话可不写情绪关键词，让模型自动匹配。\n"
            "2. 当对话有明显情绪色彩时（安慰、生气、惊讶、开心等），主动在 style_instruction 中包含对应关键词。\n"
            "3. 可与其他风格描述组合，如'温柔安慰的语气，语速稍慢'（calm 已隐含在'温柔/安慰'）。\n"
            "4. 情绪关键词写一次即可，无需重复。中文英文均可。"
        ),
        activation_type=ActivationType.ALWAYS,
        parameters=[
            ToolParameterInfo(name="reply_text", param_type=ToolParamType.STRING, description="回复文本。可在文本中插入音频标签控制句内语气细节，如（叹气）（轻笑）（温柔）（紧张）等。", required=True),
            ToolParameterInfo(name="msg_id", param_type=ToolParamType.STRING, description="当前消息ID", required=True),
            ToolParameterInfo(
                name="style_instruction",
                param_type=ToolParamType.STRING,
                description=(
                    "（可选）语音风格指令，用自然语言描述语气、情绪、语速等，让语音更拟人。\n"
                    "不传则默认朗读。当你觉得需要更生动的表达时再填写。\n"
                    "简单用法：'温柔安慰的语气，语速稍慢'\n"
                    "导演模式：'角色：XX，性格XX。场景：XX。指导：语调XX，语速XX，气息XX。'\n"
                    "也可配合 reply_text 中的音频标签（叹气/轻笑/停顿等）使用。"
                ),
                required=False,
                default="",
            ),
        ],
    )
    async def send_voice_reply(self, reply_text: str, msg_id: str = "", style_instruction: str = "", **kwargs: Any) -> dict[str, Any]:
        # Find the correct stream_id
        stream_id = await self._find_stream_id(kwargs)
        if not stream_id:
            self.ctx.logger.error("Cannot find stream_id for msg_id=%s, kwargs_keys=%s", msg_id, list(kwargs.keys()))
            return {"success": False, "method": "voice", "error": "Cannot find chat stream"}

        self.ctx.logger.info("Using stream_id=%s for voice reply, text_len=%d", stream_id, len(reply_text))

        # Always use the configured default voice
        asyncio.create_task(self._async_voice_reply(reply_text, style_instruction, stream_id, ""))
        return {"success": True, "method": "voice", "error": ""}

    async def _async_voice_reply(self, text: str, style_instruction: str, stream_id: str, voice_name: str) -> None:
        """Send voice reply asynchronously."""
        try:
            result = await self.voice_clone_tts(text=text, style_instruction=style_instruction, stream_id=stream_id, voice_name=voice_name)
            self.ctx.logger.info("Async voice reply result: %s", result.get("success"))
        except Exception as e:
            self.ctx.logger.error("Async voice reply failed: %s", e)


def create_plugin() -> AIVoicePlugin:
    return AIVoicePlugin()