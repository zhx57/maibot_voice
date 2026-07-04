"""MiniMax 同步语音合成服务模块。

封装 MiniMax 同步 TTS 流程：单次 POST /v1/t2a_v2（stream=false），
响应 ``data.audio`` 为 hex 编码字符串，解码为原始字节后 base64 编码返回。

同时实现 VoiceCloneManager 音色复刻子模块（上传音频 + 注册克隆音色）。
"""

import asyncio
import base64
import gc
import logging
import random
import time
from pathlib import Path
from typing import Any, Optional

import aiohttp


# ---------------------------------------------------------------------------
# 模块级辅助函数
# ---------------------------------------------------------------------------

def generate_voice_id_from_filename(file_name: str) -> str:
    """基于文件名生成符合 MiniMax voice_id 命名规则的标识符。

    规则：
    - 长度 [8, 256]
    - 首字符必须为英文字母（若文件名首字符非英文字母，前缀加 "v"）
    - 仅含字母/数字/``-``/``_``（其他字符替换为 ``-``）
    - 末位不可为 ``-`` 或 ``_``
    - 长度不足 8 则后补 "voice" 直到 >=8
    - 长度超 256 则截断
    """
    # 去扩展名
    base = file_name if "." not in file_name else file_name.rsplit(".", 1)[0]
    if not base:
        base = "voice"

    # 首字符必须为英文字母，否则前缀加 "v"
    if not (base[0].isascii() and base[0].isalpha()):
        base = "v" + base

    # 仅保留 ASCII 字母数字，其他字符替换为 "-"
    chars: list[str] = []
    for c in base:
        if c.isascii() and c.isalnum():
            chars.append(c)
        else:
            chars.append("-")
    result = "".join(chars)

    # 末位不可为 "-" 或 "_"
    while result and result[-1] in ("-", "_"):
        result = result[:-1]

    # 长度不足 8 则后补 "voice" 直到 >=8
    while len(result) < 8:
        result = result + "voice"

    # 长度超 256 则截断，并再次清理末位
    if len(result) > 256:
        result = result[:256]
        while result and result[-1] in ("-", "_"):
            result = result[:-1]

    # 极端情况下截断/清理后可能再次变短，做一次兜底
    while len(result) < 8:
        result = result + "voice"

    return result


# ---------------------------------------------------------------------------
# MiniMax 同步 TTS 主服务
# ---------------------------------------------------------------------------

class MiniMaxSyncTTSService:
    """MiniMax 同步语音合成服务。

    通过单次 POST /v1/t2a_v2（stream=false）完成语音合成，
    响应 ``data.audio`` 为 hex 编码字符串，解码为原始字节后 base64 编码返回。
    """

    DEFAULT_API_BASE_URL = "https://api.minimaxi.com"
    SUPPORTED_MODELS = {
        "speech-2.8-hd", "speech-2.8-turbo",
        "speech-2.6-hd", "speech-2.6-turbo",
        "speech-02-hd", "speech-02-turbo",
        "speech-01-hd", "speech-01-turbo",
    }
    # 官方 T2AAudioSetting.format 枚举
    SUPPORTED_FORMATS = {"mp3", "pcm", "flac", "wav", "pcmu_raw", "pcmu_wav", "opus"}
    # 官方文档 t2a_v2 错误码：
    #   1000 未知错误、1001 超时、1002 限流、1004 鉴权失败、
    #   1039 TPM限流、1042 非法字符超10%、2013 参数错误
    RETRYABLE_CODES = {1001, 1002, 1039}
    FATAL_CODES = {1000, 1004, 1042, 2013}
    # voice_modify 仅支持以下格式（官方 VoiceModify 说明）
    VOICE_MODIFY_FORMATS = {"mp3", "wav", "flac"}
    # 同步 TTS 单次请求文本长度上限
    MAX_TEXT_LENGTH = 10000

    def __init__(
        self,
        api_key: str,
        api_base_url: str = "",
        model: str = "speech-2.8-hd",
        max_retries: int = 3,
        retry_backoff_base: float = 1.5,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.api_key: str = api_key.strip() if api_key else ""
        self.api_base_url: str = (api_base_url.strip() if api_base_url else self.DEFAULT_API_BASE_URL).rstrip("/")
        self.model: str = model if model in self.SUPPORTED_MODELS else "speech-2.8-hd"
        self.max_retries: int = int(max_retries)
        self.retry_backoff_base: float = float(retry_backoff_base)
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # 配置热更新
    # ------------------------------------------------------------------

    def update_api_key(self, api_key: str) -> None:
        self.api_key = api_key.strip() if api_key else ""

    def update_api_base_url(self, api_base_url: str) -> None:
        self.api_base_url = (api_base_url.strip() if api_base_url else self.DEFAULT_API_BASE_URL).rstrip("/")

    def update_model(self, model: str) -> None:
        if model in self.SUPPORTED_MODELS:
            self.model = model
        else:
            self.logger.warning("Unsupported model: %s, keeping current: %s", model, self.model)

    # ------------------------------------------------------------------
    # HTTP 基础设施
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建复用的 ClientSession（timeout 设大一些，total=300）。"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _classify_error(self, status_code: int) -> str:
        """返回 'retryable' / 'fatal' / 'unknown'。"""
        if status_code in self.RETRYABLE_CODES:
            return "retryable"
        if status_code in self.FATAL_CODES:
            return "fatal"
        return "unknown"

    async def close(self) -> None:
        """关闭 ClientSession，释放资源。"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # 同步语音合成
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        voice_setting: Optional[dict] = None,
        audio_setting: Optional[dict] = None,
        pronunciation_dict: Optional[dict] = None,
        language_boost: Optional[str] = None,
        voice_modify: Optional[dict] = None,
        aigc_watermark: bool = False,
        subtitle_enable: bool = False,
        subtitle_type: str = "sentence",
        latex_read: bool = False,
    ) -> dict[str, Any]:
        """单次 POST /v1/t2a_v2（stream=false）完成同步语音合成。

        成功返回 ``{"success": True, "audio_base64": <str>, "format": <str>, "text": text}``，
        失败返回 ``{"success": False, "error": <msg>}`` 或
        ``{"success": False, "error": <msg>, "code": <status_code>}``。

        payload 字段名严格对齐官方同步 T2aV2Req schema：
        - 顶层：model/text/stream/voice_setting/audio_setting/pronunciation_dict/
          language_boost/voice_modify/subtitle_enable/subtitle_type/aigc_watermark
        - voice_setting：voice_id/speed/vol/pitch/emotion/text_normalization/latex_read
        - audio_setting：sample_rate/bitrate/format/channel

        错误码分类：
        - retryable（重试）：1001 超时 / 1002 限流 / 1039 TPM限流 / 网络异常
        - fatal（立即返回）：1000 未知 / 1004 鉴权失败 / 1042 非法字符超10% / 2013 参数错误
        """
        if not self.api_key:
            return {"success": False, "error": "API Key not configured"}
        if not text or not text.strip():
            return {"success": False, "error": "Empty text"}
        if not voice_id:
            return {"success": False, "error": "voice_id is required"}
        if self.model not in self.SUPPORTED_MODELS:
            return {"success": False, "error": f"Unsupported model: {self.model}"}

        # 同步限制：文本长度上限 10000 字符
        if len(text) > self.MAX_TEXT_LENGTH:
            self.logger.warning(
                "synthesize text length %d exceeds %d, truncated",
                len(text), self.MAX_TEXT_LENGTH,
            )
            text = text[:self.MAX_TEXT_LENGTH]

        # 解析输出格式（用于返回）
        fmt = "mp3"
        if audio_setting and audio_setting.get("format"):
            candidate = str(audio_setting.get("format")).lower()
            if candidate in self.SUPPORTED_FORMATS:
                fmt = candidate

        # 构造 voice_setting（voice_id 必填，latex_read 由参数注入）
        vs: dict[str, Any] = dict(voice_setting or {})
        vs["voice_id"] = voice_id
        vs["latex_read"] = bool(latex_read)

        # payload 严格对照官方同步 T2aV2Req schema
        payload: dict[str, Any] = {
            "model": self.model,
            "text": text,
            "stream": False,
            "voice_setting": vs,
            "subtitle_enable": bool(subtitle_enable),
            "subtitle_type": subtitle_type,
            "aigc_watermark": bool(aigc_watermark),
        }
        if audio_setting:
            payload["audio_setting"] = audio_setting
        if pronunciation_dict:
            payload["pronunciation_dict"] = pronunciation_dict
        if language_boost:
            payload["language_boost"] = language_boost
        if voice_modify:
            payload["voice_modify"] = voice_modify

        url = f"{self.api_base_url}/v1/t2a_v2"
        last_error = "unknown error"

        for attempt in range(self.max_retries + 1):
            try:
                session = await self._get_session()
                async with session.post(url, json=payload, headers=self._headers()) as response:
                    data: Optional[dict] = None
                    try:
                        data = await response.json(content_type=None)
                    except Exception:
                        data = None

                    if isinstance(data, dict):
                        base_resp = data.get("base_resp", {}) or {}
                        sc = base_resp.get("status_code", -1)
                        if sc == 0:
                            data_obj = data.get("data") or {}
                            status = data_obj.get("status")
                            hex_str = data_obj.get("audio")
                            # 同步非流式：data.status==2 表示合成结束
                            if status != 2:
                                return {
                                    "success": False,
                                    "error": f"data.status={status}, expected 2",
                                    "code": sc,
                                }
                            if not hex_str:
                                return {
                                    "success": False,
                                    "error": "No audio data in response",
                                    "code": sc,
                                }
                            try:
                                audio_bytes = bytes.fromhex(hex_str)
                            except (ValueError, TypeError) as e:
                                return {
                                    "success": False,
                                    "error": f"hex decode failed: {e}",
                                    "code": sc,
                                }
                            audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
                            self.logger.info(
                                "TTS synthesize success: b64_len=%d, format=%s, voice=%s",
                                len(audio_base64), fmt, voice_id,
                            )
                            return {
                                "success": True,
                                "audio_base64": audio_base64,
                                "format": fmt,
                                "text": text,
                            }

                        error_msg = base_resp.get("status_msg", "unknown error")
                        last_error = f"status_code={sc}, msg={error_msg}"
                        classification = self._classify_error(sc)
                        # retryable 才重试
                        if classification == "retryable" and attempt < self.max_retries:
                            backoff = self.retry_backoff_base ** attempt + random.uniform(0, 1)
                            self.logger.warning(
                                "synthesize retryable error (attempt=%d/%d): %s, retry in %.2fs",
                                attempt + 1, self.max_retries, last_error, backoff,
                            )
                            await asyncio.sleep(backoff)
                            continue
                        # fatal / unknown / 重试耗尽：直接返回
                        return {"success": False, "error": last_error, "code": sc}

                    # 非 JSON 响应（多为 HTTP 错误），按瞬时错误重试
                    last_error = f"HTTP {response.status}: invalid response"
                    if attempt < self.max_retries:
                        backoff = self.retry_backoff_base ** attempt + random.uniform(0, 1)
                        self.logger.warning(
                            "synthesize invalid response (attempt=%d/%d): %s, retry in %.2fs",
                            attempt + 1, self.max_retries, last_error, backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    return {"success": False, "error": last_error}

            except aiohttp.ClientError as e:
                last_error = f"Network error: {e}"
                self.logger.warning(
                    "synthesize network error (attempt=%d/%d): %s",
                    attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    backoff = self.retry_backoff_base ** attempt + random.uniform(0, 1)
                    await asyncio.sleep(backoff)
                    continue
                return {"success": False, "error": last_error}
            except Exception as e:
                last_error = str(e)
                self.logger.error(
                    "synthesize error (attempt=%d/%d): %s",
                    attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    backoff = self.retry_backoff_base ** attempt + random.uniform(0, 1)
                    await asyncio.sleep(backoff)
                    continue
                return {"success": False, "error": last_error}

        return {"success": False, "error": last_error}


# ---------------------------------------------------------------------------
# Voice Clone 音色复刻管理
# ---------------------------------------------------------------------------

class VoiceCloneManager:
    """MiniMax 音色复刻管理器：上传参考音频 + 注册克隆音色。"""

    def __init__(
        self,
        api_key: str,
        api_base_url: str = "",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.api_key: str = api_key.strip() if api_key else ""
        self.api_base_url: str = (
            api_base_url.strip() if api_base_url else MiniMaxSyncTTSService.DEFAULT_API_BASE_URL
        ).rstrip("/")
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _auth_headers(self) -> dict[str, str]:
        """仅 Authorization，multipart 上传时不要手动加 Content-Type。"""
        return {"Authorization": f"Bearer {self.api_key}"}

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # 上传音频文件
    # ------------------------------------------------------------------

    async def upload_audio(self, file_path: str) -> int:
        """multipart 上传音频到 POST /v1/files/upload，返回 file_id (int)。

        form fields: purpose=voice_clone, file=<音频文件>。
        headers 只需 Authorization Bearer（multipart 自动设置 boundary）。
        """
        if not self.api_key:
            raise RuntimeError("API Key not configured")
        if not file_path:
            raise RuntimeError("file_path is empty")

        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"Audio file not found: {file_path}")

        url = f"{self.api_base_url}/v1/files/upload"
        session = await self._get_session()

        with open(path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("purpose", "voice_clone")
            form.add_field(
                "file",
                f,
                filename=path.name,
                content_type="application/octet-stream",
            )
            async with session.post(url, data=form, headers=self._auth_headers()) as response:
                try:
                    data = await response.json(content_type=None)
                except Exception:
                    data = None

                if not isinstance(data, dict):
                    raise RuntimeError(f"upload_audio invalid response: HTTP {response.status}")

                base_resp = data.get("base_resp", {}) or {}
                sc = base_resp.get("status_code", -1)
                if sc != 0:
                    raise RuntimeError(
                        f"upload_audio failed: status_code={sc}, msg={base_resp.get('status_msg')}"
                    )

                file_info = data.get("file", {}) or {}
                file_id = file_info.get("file_id")
                if file_id is None:
                    raise RuntimeError("upload_audio: no file_id in response")

                self.logger.info(
                    "Voice clone audio uploaded: file=%s, file_id=%s", path.name, file_id,
                )
                return int(file_id)

    # ------------------------------------------------------------------
    # 注册克隆音色
    # ------------------------------------------------------------------

    async def register_clone_voice(self, file_id: int, voice_id: str) -> dict[str, Any]:
        """注册克隆音色 POST /v1/voice_clone。

        body: ``{"file_id": <int>, "voice_id": "<自定义ID>"}``。
        返回 ``{"success": True}`` 或 ``{"success": False, "error":..., "code":...}``。
        """
        if not self.api_key:
            return {"success": False, "error": "API Key not configured", "code": -1}

        url = f"{self.api_base_url}/v1/voice_clone"
        payload = {"file_id": file_id, "voice_id": voice_id}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as response:
                try:
                    data = await response.json(content_type=None)
                except Exception:
                    data = None

                if not isinstance(data, dict):
                    return {
                        "success": False,
                        "error": f"register_clone_voice invalid response: HTTP {response.status}",
                        "code": -1,
                    }

                base_resp = data.get("base_resp", {}) or {}
                sc = base_resp.get("status_code", -1)
                if sc == 0:
                    self.logger.info(
                        "Voice clone registered: voice_id=%s, file_id=%s", voice_id, file_id,
                    )
                    return {"success": True}

                error_msg = base_resp.get("status_msg", "unknown error")
                self.logger.error(
                    "register_clone_voice failed: code=%s, msg=%s, voice_id=%s",
                    sc, error_msg, voice_id,
                )
                return {"success": False, "error": error_msg, "code": sc}
        except aiohttp.ClientError as e:
            self.logger.error("register_clone_voice network error: %s", e)
            return {"success": False, "error": f"Network error: {e}", "code": -1}
        except Exception as e:
            self.logger.error("register_clone_voice error: %s", e)
            return {"success": False, "error": str(e), "code": -1}

    # ------------------------------------------------------------------
    # 生成符合规则的 voice_id
    # ------------------------------------------------------------------

    def generate_voice_id(self, file_name: str) -> str:
        """基于文件名生成符合 MiniMax voice_id 命名规则的标识符。"""
        return generate_voice_id_from_filename(file_name)
