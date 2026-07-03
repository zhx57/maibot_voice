"""MiniMax 异步语音合成服务模块。

封装 MiniMax 异步 TTS 的完整三步流程：
1. 创建异步语音合成任务 (POST /v1/t2a_async_v2)
2. 轮询任务状态 (GET /v1/query/t2a_async_query_v2)
3. 检索并下载音频 (GET /v1/files/retrieve + 预签名 URL 下载)

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
# MiniMax 异步 TTS 主服务
# ---------------------------------------------------------------------------

class MiniMaxAsyncTTSService:
    """MiniMax 异步语音合成服务。

    通过创建异步任务 + 轮询状态 + 下载音频三步流程完成语音合成。
    """

    DEFAULT_API_BASE_URL = "https://api.minimaxi.com"
    SUPPORTED_MODELS = {
        "speech-2.8-hd", "speech-2.8-turbo",
        "speech-2.6-hd", "speech-2.6-turbo",
        "speech-02-hd", "speech-02-turbo",
        "speech-01-hd", "speech-01-turbo",
    }
    # 官方 T2AAsyncV2AudioSetting.format 枚举
    SUPPORTED_FORMATS = {"mp3", "pcm", "flac", "wav", "pcmu_raw", "pcmu_wav", "opus"}
    # 官方文档 t2a_async_v2 错误码：1002 限流、1004 鉴权失败、1039 TPM限流、1042 非法字符超10%、2013 参数错误
    # 1001(超时) 为通用瞬时错误，按可重试处理
    RETRYABLE_CODES = {1001, 1002, 1039}
    FATAL_CODES = {1004, 1008, 1042, 2013, 2038}
    # voice_modify 仅支持以下格式（官方 VoiceModify 说明）
    VOICE_MODIFY_FORMATS = {"mp3", "wav", "flac"}
    QUERY_MAX_QPS = 10  # 查询接口每秒最多 10 次
    MIN_POLL_INTERVAL = 1.0  # 轮询间隔至少 1 秒

    def __init__(
        self,
        api_key: str,
        api_base_url: str = "",
        model: str = "speech-2.8-hd",
        poll_interval: float = 1.0,
        poll_max_wait: float = 120,
        max_retries: int = 3,
        retry_backoff_base: float = 1.5,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.api_key: str = api_key.strip() if api_key else ""
        self.api_base_url: str = (api_base_url.strip() if api_base_url else self.DEFAULT_API_BASE_URL).rstrip("/")
        self.model: str = model if model in self.SUPPORTED_MODELS else "speech-2.8-hd"
        self.poll_interval: float = float(poll_interval)
        self.poll_max_wait: float = float(poll_max_wait)
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
    # 第 1 步：创建异步语音合成任务
    # ------------------------------------------------------------------

    async def create_task(
        self,
        text: str,
        voice_id: str,
        voice_setting: Optional[dict] = None,
        audio_setting: Optional[dict] = None,
        pronunciation_dict: Optional[dict] = None,
        language_boost: Optional[str] = None,
        voice_modify: Optional[dict] = None,
        aigc_watermark: bool = False,
    ) -> dict[str, Any]:
        """调用 POST /v1/t2a_async_v2 创建任务。

        成功返回 ``{"task_id":..., "file_id":..., "usage_characters":...}``，
        失败返回 ``{"success": False, "error":..., "code":...}``。
        对 retryable 错误按指数退避重试 max_retries 次。

        注意：payload 仅包含官方 T2AAsyncV2Req schema 定义的字段
        (model/text/voice_setting/audio_setting/pronunciation_dict/language_boost/voice_modify/aigc_watermark)，
        不发送 schema 未定义的字段（如 subtitle_enabled），避免触发 2013 参数错误。
        """
        if not self.api_key:
            return {"success": False, "error": "API Key not configured"}
        if self.model not in self.SUPPORTED_MODELS:
            return {"success": False, "error": f"Unsupported model: {self.model}"}

        # 构造 voice_setting（voice_id 参数优先）
        vs: dict[str, Any] = dict(voice_setting or {})
        vs["voice_id"] = voice_id

        # payload 严格对照官方 T2AAsyncV2Req schema
        payload: dict[str, Any] = {
            "model": self.model,
            "text": text,
            "voice_setting": vs,
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

        url = f"{self.api_base_url}/v1/t2a_async_v2"
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
                            return {
                                "task_id": data.get("task_id"),
                                "file_id": data.get("file_id"),
                                "usage_characters": data.get("usage_characters", 0),
                            }
                        error_msg = base_resp.get("status_msg", "unknown error")
                        last_error = f"status_code={sc}, msg={error_msg}"
                        classification = self._classify_error(sc)
                        # retryable 才重试
                        if classification == "retryable" and attempt < self.max_retries:
                            backoff = self.retry_backoff_base ** attempt + random.uniform(0, 1)
                            self.logger.warning(
                                "create_task retryable error (attempt=%d/%d): %s, retry in %.2fs",
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
                            "create_task invalid response (attempt=%d/%d): %s, retry in %.2fs",
                            attempt + 1, self.max_retries, last_error, backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    return {"success": False, "error": last_error}

            except aiohttp.ClientError as e:
                last_error = f"Network error: {e}"
                self.logger.warning(
                    "create_task network error (attempt=%d/%d): %s",
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
                    "create_task error (attempt=%d/%d): %s",
                    attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    backoff = self.retry_backoff_base ** attempt + random.uniform(0, 1)
                    await asyncio.sleep(backoff)
                    continue
                return {"success": False, "error": last_error}

        return {"success": False, "error": last_error}

    # ------------------------------------------------------------------
    # 第 2 步：查询任务状态
    # ------------------------------------------------------------------

    async def query_task(self, task_id: Any) -> dict[str, Any]:
        """调用 GET /v1/query/t2a_async_query_v2 查询任务状态。

        返回 ``{"status": <小写>, "file_id":..., "raw_status":...}``。
        status 大小写不敏感比较，统一返回小写：processing/success/failed/expired。
        网络错误时返回 processing 状态（带 error 字段），由上层轮询超时兜底。
        """
        url = f"{self.api_base_url}/v1/query/t2a_async_query_v2"
        params = {"task_id": task_id}
        try:
            session = await self._get_session()
            async with session.get(url, params=params, headers=self._headers()) as response:
                try:
                    data = await response.json(content_type=None)
                except Exception:
                    data = None

                if not isinstance(data, dict):
                    self.logger.warning("query_task non-JSON response: HTTP %s", response.status)
                    return {
                        "status": "processing",
                        "file_id": None,
                        "raw_status": "Processing",
                        "error": f"HTTP {response.status}: invalid response",
                    }

                raw_status = data.get("status", "Processing")
                file_id = data.get("file_id")
                status_lower = str(raw_status).lower()
                return {
                    "status": status_lower,
                    "file_id": file_id,
                    "raw_status": raw_status,
                }
        except aiohttp.ClientError as e:
            self.logger.warning("query_task network error (will retry in loop): %s", e)
            return {
                "status": "processing",
                "file_id": None,
                "raw_status": "Processing",
                "error": f"Network error: {e}",
            }
        except Exception as e:
            self.logger.warning("query_task error (will retry in loop): %s", e)
            return {
                "status": "processing",
                "file_id": None,
                "raw_status": "Processing",
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # 第 3 步：文件检索（获取下载 URL）
    # ------------------------------------------------------------------

    async def retrieve_file(self, file_id: Any) -> dict[str, Any]:
        """调用 GET /v1/files/retrieve 获取预签名下载 URL。

        返回 ``{"download_url":..., "filename":..., "bytes":...}``。
        """
        url = f"{self.api_base_url}/v1/files/retrieve"
        params = {"file_id": file_id}
        session = await self._get_session()
        async with session.get(url, params=params, headers=self._headers()) as response:
            try:
                data = await response.json(content_type=None)
            except Exception:
                data = None

            if not isinstance(data, dict):
                raise aiohttp.ClientError(f"retrieve_file invalid response: HTTP {response.status}")

            file_info = data.get("file", {}) or {}
            return {
                "download_url": file_info.get("download_url", ""),
                "filename": file_info.get("filename", ""),
                "bytes": file_info.get("bytes", 0),
            }

    # ------------------------------------------------------------------
    # 第 4 步：下载音频（预签名 URL，不需要鉴权头）
    # ------------------------------------------------------------------

    async def download_audio(self, download_url: str) -> str:
        """流式下载音频并返回纯 base64 字符串（无 data: 前缀）。

        download_url 是预签名 URL，下载时不需要鉴权头。
        """
        session = await self._get_session()
        chunks_bytes = bytearray()
        try:
            async with session.get(download_url) as response:
                response.raise_for_status()
                async for chunk in response.content.iter_chunked(65536):
                    chunks_bytes.extend(chunk)
            audio_base64 = base64.b64encode(bytes(chunks_bytes)).decode("ascii")
        finally:
            # 立即释放原始字节
            chunks_bytes.clear()
            del chunks_bytes
            gc.collect()
        return audio_base64

    # ------------------------------------------------------------------
    # 编排：合成完整流程
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
    ) -> dict[str, Any]:
        """编排完整三步流程：创建任务 -> 轮询 -> 检索 -> 下载。

        成功返回 ``{"success": True, "audio_base64": <str>, "format": <str>, "text": text}``，
        失败返回 ``{"success": False, "error": str}``。
        """
        if not self.api_key:
            return {"success": False, "error": "API Key not configured"}
        if not text or not text.strip():
            return {"success": False, "error": "Empty text"}
        if not voice_id:
            return {"success": False, "error": "voice_id is required"}

        # 解析输出格式（用于返回）
        fmt = "mp3"
        if audio_setting and audio_setting.get("format"):
            candidate = str(audio_setting.get("format")).lower()
            if candidate in self.SUPPORTED_FORMATS:
                fmt = candidate

        try:
            # --- 第 1 步：创建任务 ---
            create_result = await self.create_task(
                text=text,
                voice_id=voice_id,
                voice_setting=voice_setting,
                audio_setting=audio_setting,
                pronunciation_dict=pronunciation_dict,
                language_boost=language_boost,
                voice_modify=voice_modify,
                aigc_watermark=aigc_watermark,
            )
            if "task_id" not in create_result:
                return {
                    "success": False,
                    "error": create_result.get("error", "create_task failed"),
                }
            task_id = create_result.get("task_id")
            self.logger.info(
                "TTS task created: task_id=%s, usage_characters=%s, model=%s, voice=%s",
                task_id, create_result.get("usage_characters"), self.model, voice_id,
            )

            # --- 第 2 步：轮询任务状态 ---
            start_time = time.time()
            file_id: Any = None
            poll_count = 0
            while True:
                elapsed = time.time() - start_time
                if elapsed > self.poll_max_wait:
                    return {
                        "success": False,
                        "error": f"Polling timeout after {self.poll_max_wait}s, task_id={task_id}",
                        "task_id": task_id,
                    }

                query_result = await self.query_task(task_id)
                status = str(query_result.get("status", "processing")).lower()

                if status == "success":
                    file_id = query_result.get("file_id")
                    if file_id is None:
                        return {
                            "success": False,
                            "error": "Success status but no file_id",
                            "task_id": task_id,
                        }
                    self.logger.info("TTS task success: task_id=%s, file_id=%s", task_id, file_id)
                    break
                elif status == "failed":
                    raw_status = query_result.get("raw_status", "Failed")
                    err_detail = query_result.get("error", raw_status)
                    return {
                        "success": False,
                        "error": f"任务失败: {err_detail}",
                        "task_id": task_id,
                    }
                elif status == "expired":
                    return {
                        "success": False,
                        "error": "任务已过期",
                        "task_id": task_id,
                    }
                else:
                    # processing（含查询瞬时错误，由超时兜底）
                    poll_count += 1
                    backoff_factor = min(1.0 + (poll_count - 1) * 0.2, 3.0)
                    sleep_time = max(
                        self.MIN_POLL_INTERVAL,
                        self.poll_interval * backoff_factor + random.uniform(0, 0.5),
                    )
                    # QUERY_MAX_QPS=10 即最小 0.1s，MIN_POLL_INTERVAL 已保证
                    await asyncio.sleep(sleep_time)
                    continue

            # --- 第 3 步：检索文件下载 URL ---
            retrieve_result = await self.retrieve_file(file_id)
            download_url = retrieve_result.get("download_url", "")
            if not download_url:
                return {
                    "success": False,
                    "error": "No download_url from retrieve_file",
                    "task_id": task_id,
                    "file_id": file_id,
                }
            self.logger.info(
                "TTS file retrieved: file_id=%s, filename=%s, bytes=%s",
                file_id, retrieve_result.get("filename"), retrieve_result.get("bytes"),
            )

            # --- 第 4 步：下载音频并 base64 编码 ---
            audio_base64 = await self.download_audio(download_url)
            self.logger.info(
                "TTS audio downloaded: b64_len=%d, format=%s", len(audio_base64), fmt,
            )

            return {
                "success": True,
                "audio_base64": audio_base64,
                "format": fmt,
                "text": text,
            }

        except aiohttp.ClientError as e:
            self.logger.error("synthesize network error: %s", e)
            return {"success": False, "error": f"Network error: {e}"}
        except Exception as e:
            self.logger.error("synthesize error: %s", e)
            return {"success": False, "error": str(e)}


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
            api_base_url.strip() if api_base_url else MiniMaxAsyncTTSService.DEFAULT_API_BASE_URL
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
