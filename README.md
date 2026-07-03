# MiniMax Async TTS Voice Plugin for MaiBot

基于 MiniMax 异步语音合成 API 的 AI 语音服务插件，为 MaiBot 提供智能语音回复功能。支持**预置音色**和**音色复刻**两种模式。采用异步任务模型（创建任务→轮询→下载），专为长文本场景优化。


## 功能特性(好用的话请点个star⭐~)

- 🎤 **预置音色**：支持 MiniMax 100+ 系统音色（中英日韩等多语种）
- 🎭 **音色复刻**：通过音频样本注册克隆音色，自动缓存 voice_id 复用，无需每次传输参考音频
- 🤖 **AI 工具链暴露**：`send_voice_reply` 工具暴露在 AI 工具链中（`activation_type=ALWAYS`），AI 可自主决定何时使用语音回复
- 🎨 **AI 智能风格控制**：风格指令由 AI 自行决定，支持自然语言风格指令映射为 MiniMax emotion + voice_modify
- 🎬 **导演模式**：支持从角色、场景、指导三个维度刻画语音表演
- ⚡ **异步发送**：语音合成和发送异步执行，不阻塞 AI 主循环
- 🔧 **自动配置初始化**：首次启动自动生成 config.toml
- 🛡️ **生产级可靠性**：指数退避重试、限流感知、超时/过期处理、错误分类
- 📝 **丰富参数**：情绪枚举、语速/音量/音调、音频格式/采样率/比特率、发音字典、音色修改

## 安装

### 1. 复制插件到 MaiBot

```bash
cd /path/to/MaiBot/plugins/
git clone https://github.com/Emilia-awa/maibot-mimotts-voice.git
```

### 2. 安装依赖

```bash
pip install aiohttp
```

测试依赖（可选）：

```bash
pip install pytest pytest-asyncio aioresponses
```

### 3. 配置

编辑 `config.toml`（可从 `config.example.toml` 复制）：

```toml
[plugin]
enabled = true
config_version = "2.0.0"

[voice]
minimax_api_key = "your_api_key_here"
api_base_url = "https://api.minimaxi.com"
model = "speech-2.8-hd"
voice_mode = "clone"
preset_voice = "English_expressive_narrator"
voices_dir = "voices"
default_voice = ""
clone_voice = ""
```

### 4. 重启 MaiBot

插件会自动加载。

## 配置说明

### API 配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `minimax_api_key` | MiniMax API Key（必填） | - |
| `api_base_url` | API 地址 | `https://api.minimaxi.com` |
| `model` | TTS 模型 | `speech-2.8-hd` |

### 音色配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `voice_mode` | 音色模式：`preset` 或 `clone` | `clone` |
| `preset_voice` | 预置音色 voice_id | `English_expressive_narrator` |
| `voices_dir` | 音色复刻目录（相对于插件目录） | `voices` |
| `default_voice` | 默认音色名称 | - |
| `clone_voice` | 指定复刻音色文件名（留空使用第一个文件） | - |

### 语音参数

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `emotion` | 情绪枚举 happy/sad/angry/fearful/disgusted/surprised/calm/fluent/whisper（留空自动；fluent/whisper 仅 2.6 系列支持） | - |
| `speed` | 语速 [0.5, 2] | `1.0` |
| `vol` | 音量 (0, 10] | `1.0` |
| `pitch` | 音调 [-12, 12] 整数 | `0` |
| `english_normalization` | 英文文本归一化 | `false` |

### 音频设置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `audio_format` | 音频格式：`mp3`/`pcm`/`flac`/`wav`/`pcmu_raw`/`pcmu_wav`/`opus` | `mp3` |
| `audio_sample_rate` | 采样率 [8000,16000,22050,24000,32000,44100] | `32000` |
| `bitrate` | 比特率 [32000,64000,128000,256000]（仅 mp3） | `128000` |
| `channel` | 声道 [1,2] | `2` |

### 其他

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `language_boost` | 语种识别增强 auto 或具体语种 | `auto` |
| `aigc_watermark` | AIGC 水印（仅非流式生效） | `false` |
| `poll_interval` | 轮询间隔（秒，≥1.0） | `1.0` |
| `poll_max_wait` | 最大轮询等待（秒） | `120` |
| `max_retries` | 最大重试次数 | `3` |
| `retry_backoff_base` | 重试退避基数 | `1.5` |

## 支持模型

| 模型 | 特性 |
|------|------|
| `speech-2.8-hd` | 情绪渲染融合语气词，重塑自然听感（默认推荐） |
| `speech-2.8-turbo` | 极致生成速度，更自然逼真 |
| `speech-2.6-hd` | 超低延时，归一化升级 |
| `speech-2.6-turbo` | 极速版，适用于语音聊天和数字人 |
| `speech-02-hd` | 出色韵律、稳定性和复刻相似度 |
| `speech-02-turbo` | 出色韵律和稳定性，小语种能力加强 |

## 音色模式

### 预置音色（preset）

无需额外配置，直接在 `preset_voice` 中填入 MiniMax 系统 voice_id（如 `English_expressive_narrator`、`audiobook_male_1` 等）。MiniMax 提供 100+ 系统音色，覆盖中英日韩等多语种，详见 MiniMax 官方文档。

### 音色复刻（clone）

1. 将 `voice_mode` 改为 `clone`
2. 将参考音频文件放入 `voices/` 目录（支持 WAV 和 MP3）
3. 插件自动上传并注册为 MiniMax 克隆音色，缓存 voice_id 复用，无需每次传输参考音频
4. 文件名即为音色名称，建议录音时长 30 秒以上

```
plugins/maibot-mimotts-voice/voices/
└── salt.wav    # 音色名称: salt
```

## 工具说明

插件提供 `send_voice_reply` 工具给 AI，AI 会根据对话上下文自行判断是否调用。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `reply_text` | string | ✅ | 要合成语音的回复文本 |
| `msg_id` | string | ✅ | 当前消息 ID，用于自动查找聊天流 |
| `style_instruction` | string | ❌ | 语音风格指令 |

### 风格控制方式

风格控制由 AI 根据当前扮演的角色人设和对话情境**自行决定**，无需用户手动指定。AI 会自动判断何时需要风格指令，并选择合适的表达方式。

支持三种风格控制方式，AI 可自由组合使用：

**1. 自然语言风格指令**（AI 通过 `style_instruction` 参数传入）

AI 用自然语言描述语气、情绪、语速等，像给演员说戏一样：

- `一位温柔的少女，声音清甜软糯，语速偏慢，用安慰的语气`
- `语气俏皮活泼，带点小得意，语速偏快`
- `声音低沉严肃，像在教训人，语速慢一些`

**2. 导演模式**（AI 根据场景需要自动启用）

AI 从角色、场景、指导三个维度全方位刻画表演：

```
角色：一位温柔的大姐姐，性格体贴温暖，声音甜美有亲和力。
场景：安慰失恋的朋友。
指导：语调柔和温暖，气息松弛，偶尔带叹息，语速偏慢，尾音上扬带笑意。
```

**3. 音频标签**（AI 在 `reply_text` 中自动插入）

AI 在文本任意位置用括号标注语气/情绪/声音动作：

- 中文：`（紧张）呼……冷静。（叹气）算了。（轻笑）好吧好吧。`
- 英文：`(sighs) I don't know. (laughs) That's funny!`
- 整段风格：`（温柔）你好呀~` `（东北话）哎呀妈呀~` `（唱歌）歌词...`
- 常用标签：开心/悲伤/愤怒/温柔/慵懒/俏皮/磁性/沙哑/甜美/冷漠/叹气/轻笑/哽咽/深呼吸 等

> `style_instruction`（整体风格）+ `reply_text` 中的音频标签（句内细节）可同时使用，两者不冲突。

## 目录结构

```
maibot-mimotts-voice/
├── plugin.py           # 插件主入口
├── tts_service.py      # MiniMax 异步 TTS 服务 + 音色复刻管理
├── config.toml         # 插件配置（gitignored）
├── config.example.toml # 配置示例
├── _manifest.json      # 插件清单
├── README.md           # 本文档
├── LICENSE             # MIT 许可证
├── requirements.txt    # Python 依赖
├── tests/              # 测试套件
│   ├── conftest.py
│   ├── test_tts_service.py
│   ├── test_voice_clone.py
│   ├── test_plugin.py
│   └── integration_test.py
└── voices/             # 音色复刻文件目录（gitignored）
    └── .voice_cache.json  # 克隆音 voice_id 缓存（自动生成）
```

## 注意事项

1. **API Key**：需在 [MiniMax 开放平台](https://platform.minimaxi.com) 申请，个人/企业认证后可用
2. **音色复刻权限**：需完成认证，否则 voice_clone 返回 2038 错误
3. **克隆音色保活**：复刻音色若 7 天内未调用 TTS 接口会被自动删除，请定期使用
4. **下载有效期**：异步任务完成后，音频下载 URL 有效期 9 小时（32400 秒），插件会立即下载
5. **音频格式**：默认 mp3，NapCat 兼容性最佳；voice_modify 仅支持 mp3/wav/flac
6. **费用**：按字符计费，详见 MiniMax 定价

## 更新日志

### v2.0.0

- 🔄 **后端切换**：从小米 MiMo 同步 TTS 整体迁移至 MiniMax 异步语音合成 API
- 🎭 **音色复刻改造**：采用标准注册制（上传→注册→voice_id 持久化复用），自动缓存
- 🎨 **风格映射**：style_instruction 映射为 MiniMax emotion + voice_modify
- 🛡️ **生产级可靠性**：指数退避重试、限流感知、超时/过期处理、错误分类
- 📝 **丰富参数**：情绪、语速/音量/音调、音频格式/采样率/比特率、发音字典、音色修改
- ✅ **测试套件**：新增完整单元测试与可选集成测试

### v1.6.0

- 原小米 MiMo 版本（已弃用）

## 许可证

MIT License
