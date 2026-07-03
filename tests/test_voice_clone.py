"""测试 VoiceCloneManager：上传音频 + 注册克隆音色。"""

from pathlib import Path

from aioresponses import aioresponses

from tts_service import VoiceCloneManager, generate_voice_id_from_filename


UPLOAD_URL = "https://api.minimaxi.com/v1/files/upload"
CLONE_URL = "https://api.minimaxi.com/v1/voice_clone"


async def test_upload_audio(mock_logger, sample_mp3_bytes, tmp_path):
    """mock POST /v1/files/upload 返回 file_id，断言返回正确 file_id。"""
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(sample_mp3_bytes)

    mgr = VoiceCloneManager(api_key="test-key", logger=mock_logger)
    try:
        with aioresponses() as m:
            m.post(
                UPLOAD_URL,
                payload={
                    "file": {"file_id": 12345, "filename": "sample.mp3", "bytes": len(sample_mp3_bytes)},
                    "base_resp": {"status_code": 0, "status_msg": "ok"},
                },
            )
            file_id = await mgr.upload_audio(str(audio_file))

        assert file_id == 12345
        assert isinstance(file_id, int)
    finally:
        await mgr.close()


async def test_register_clone_voice_success(mock_logger):
    """mock POST /v1/voice_clone 返回 status_code=0，断言 success=True。"""
    mgr = VoiceCloneManager(api_key="test-key", logger=mock_logger)
    try:
        with aioresponses() as m:
            m.post(
                CLONE_URL,
                payload={"base_resp": {"status_code": 0, "status_msg": "success"}},
            )
            result = await mgr.register_clone_voice(file_id=12345, voice_id="myvoice123")

        assert result["success"] is True
    finally:
        await mgr.close()


async def test_register_clone_voice_1043(mock_logger):
    """mock 返回 status_code=1043，断言 success=False 且 code=1043。"""
    mgr = VoiceCloneManager(api_key="test-key", logger=mock_logger)
    try:
        with aioresponses() as m:
            m.post(
                CLONE_URL,
                payload={"base_resp": {"status_code": 1043, "status_msg": "voice id exists"}},
            )
            result = await mgr.register_clone_voice(file_id=12345, voice_id="myvoice123")

        assert result["success"] is False
        assert result["code"] == 1043
        assert "error" in result
    finally:
        await mgr.close()


async def test_register_clone_voice_2038(mock_logger):
    """mock 返回 status_code=2038，断言 success=False 且 code=2038。"""
    mgr = VoiceCloneManager(api_key="test-key", logger=mock_logger)
    try:
        with aioresponses() as m:
            m.post(
                CLONE_URL,
                payload={"base_resp": {"status_code": 2038, "status_msg": "quota exceeded"}},
            )
            result = await mgr.register_clone_voice(file_id=12345, voice_id="myvoice123")

        assert result["success"] is False
        assert result["code"] == 2038
    finally:
        await mgr.close()


def test_generate_voice_id_naming_rules():
    """测试 generate_voice_id_from_filename 生成符合规则。"""
    def is_valid(vid: str) -> bool:
        # 首字符字母
        if not (vid[0].isascii() and vid[0].isalpha()):
            return False
        # 仅字母数字 - _
        for c in vid:
            if not (c.isascii() and (c.isalnum() or c in ("-", "_"))):
                return False
        # 末位非 -_
        if vid[-1] in ("-", "_"):
            return False
        # 长度 [8, 256]
        if not (8 <= len(vid) <= 256):
            return False
        return True

    # VoiceCloneManager.generate_voice_id 委托给模块级函数
    mgr = VoiceCloneManager(api_key="test-key")

    cases = [
        "narrator.wav",
        "温柔的少女.mp3",
        "123start.wav",        # 数字开头
        "ab.wav",              # 过短
        "hello world.wav",     # 含空格
        "special!@#.wav",      # 特殊字符
        "My-Cool_Voice.wav",
    ]
    for name in cases:
        vid = mgr.generate_voice_id(name)
        assert is_valid(vid), f"generate_voice_id 生成不合规: {vid!r} (from {name!r})"
        # 与模块级函数一致
        assert vid == generate_voice_id_from_filename(name)
