# GenMelodies

将 MIDI 文件转换为**原神乐器**可用的自动演奏乐谱，并直接分析 MP3 / WAV 的 BPM、拍点与拍号。

## 功能

- 🎹 **MIDI → 数字谱** — 解析 `.mid` / `.midi` 文件，精确提取音符和节奏
- 🔤 **MIDI → 字母谱** — 生成原神乐器键盘字母谱（QWE/ASD/ZXC 布局）
- 🎯 **和弦识别** — 自动检测同时按下的多音和弦
- **MP3 / WAV 节奏分析** — 原始 PCM 起音分析，输出速度候选、拍点、下拍和不确定性；弱证据时报告未知，不硬猜 4/4
- **人工校正** — 指定四分音符 BPM、拍号和小节下拍起点；量化保留休止时长，支持 6/8 等分母
- **音频 → 粗略单音乐谱** — 适合干净独奏；不支持完整歌曲的多声部转谱或声部分离

## 安装

```bash
cd GenMelodies
pip install -r requirements.txt
```

需要 Python 3.10+。WAV 无需额外依赖；MP3 首次解码需要本机 GCC / Clang / MSVC，自动构建项目内的 MIT-0 `dr_mp3` 桥接。也可手动运行 `python tools/build_audio_decoder.py`。不需要 ffmpeg、librosa 或 SciPy。[详细用法与限制](docs/audio.md)。

## 快速使用

```bash
# MIDI 转数字谱
python genmelodies.py song.mid

# MIDI 转字母谱
python genmelodies.py song.mid --format letter

# 指定音轨
python genmelodies.py song.mid --track 1

# 指定输出路径
python genmelodies.py song.mid -o output.txt

# MP3 仅分析节奏；JSON 包含候选和置信指标
python genmelodies.py song.mp3 --analyze-rhythm --json

# 已知节奏时校正；6/8 的 BPM 也始终是四分音符 BPM
python genmelodies.py song.mp3 --analyze-rhythm --bpm 120 --time-signature 6/8 --beat-offset 0.25

# 独奏音频生成粗略单音乐谱，另存节奏诊断
python genmelodies.py solo.wav --bpm 120 --time-signature 4/4 --rhythm-json solo.rhythm.json
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入文件（.mid/.midi/.mp3/.wav） | 必填 |
| `-o, --output` | 输出文件路径 | 同输入名，.txt |
| `-f, --format` | `number` / `letter` | `number` |
| `-t, --track` | MIDI 音轨索引 | 全部合并 |
| `--analyze-rhythm` | 仅分析 MP3/WAV 节奏 | 关闭 |
| `--json` | 分析模式向 stdout 写纯 JSON | 关闭 |
| `--rhythm-json` | 保存音频节奏诊断 JSON | 无 |
| `--bpm` | 指定四分音符 BPM（20–400，有限数） | 自动 / MIDI 元数据 |
| `--time-signature` | 指定拍号，例如 3/4、6/8 | 自动 / MIDI 元数据 |
| `--beat-offset` | 第一小节下拍时间（秒） | 音频估计 / MIDI 零点 |
| `-h, --help` | 帮助信息 | - |
| `--version` | 版本号 | - |

## 乐谱格式

兼容 [刻师傅格式](https://www.bilibili.com/video/BV1Gs4y1X7Xt?p=2)，可被 [GenLyre](https://www.bilibili.com/video/BV1Gs4y1X7Xt) 读取并自动演奏。

### 数字谱
```
0.476              ← 每小节时长（秒）
(137-1-2) /...     ← 和弦 (1 3 7 -1 -2) 同时按
                   ← / 小节分隔，空格=固定时间槽的休止
```

- `1~7` : 音级 (C=1, D=2, E=3, F=4, G=5, A=6, B=7)
- `+`/`-` : 升/降八度
- `()` : 和弦
- `/` : 小节分隔
- `[]`：装饰音

### 字母谱（符号同上）
```
0.68               ← 每小节时长（秒）
(VAW) E /T Q /...  ← (VAW)=和弦, E T Q=依次单键
```

键盘布局（C大调 3 个八度）：

| 高八度 | Q | W | E | R | T | Y | U |
| 中八度 | A | S | D | F | G | H | J |
| 低八度 | Z | X | C | V | B | N | M |

## 原神乐器支持

风物之诗琴、镜花之琴等 21 键布局乐器。

## License

GPLv3

新增的第三方音频解码源码仅为 MIT-0 授权的 `dr_mp3`，无新增 Python 传递依赖；来源和许可见 [音频文档](docs/audio.md)。
