"""端到端集成测试。

需要真实 MiniMax API Key，通过环境变量 MINIMAX_API_KEY 守卫。
未设置时自动 skip。
"""

import base64
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set, skipping integration test",
)


async def test_end_to_end_synthesis():
    from tts_service import MiniMaxSyncTTSService

    api_key = os.environ["MINIMAX_API_KEY"]
    service = MiniMaxSyncTTSService(api_key=api_key, model="speech-2.8-hd")
    try:
        result = await service.synthesize(
            text="你好，这是一段测试语音。今天天气真不错(sighs)，我们一起出去走走吧。",
            voice_id="female-shaonv-jingpin",
            audio_setting={
                "format": "mp3",
                "sample_rate": 32000,
                "bitrate": 128000,
                "channel": 1,
            },
            language_boost="auto",
        )
        assert result["success"] is True
        audio_b64 = result["audio_base64"]
        assert len(audio_b64) > 0
        # 验证 base64 解码后为合法音频
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 0
        # 验证 mp3 文件头（ID3 或 0xFFFB / 0xFFF3 / 0xFFF2 帧）
        assert (
            audio_bytes[:3] == b"ID3"
            or audio_bytes[:2] == b"\xff\xfb"
            or audio_bytes[:2] == b"\xff\xf3"
            or audio_bytes[:2] == b"\xff\xf2"
        ), f"Invalid mp3 header: {audio_bytes[:4].hex()}"
        # 打印结果摘要供人工确认
        print(f"\n[OK] 合成成功，音频大小: {len(audio_bytes)} 字节, base64 长度: {len(audio_b64)}")
        print(f"[OK] 文件头: {audio_bytes[:4].hex()}")
    finally:
        await service.close()
