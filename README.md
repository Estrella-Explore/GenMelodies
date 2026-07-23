# GenMelodies

将 MIDI 文件转换为**原神乐器**可用的自动演奏乐谱。

## 功能

- 🎹 **MIDI → 数字谱** — 解析 `.mid` / `.midi` 文件，精确提取音符和节奏
- 🔤 **MIDI → 字母谱** — 生成原神乐器键盘字母谱（QWE/ASD/ZXC 布局）
- 🎯 **和弦识别** — 自动检测同时按下的多音和弦

## 安装

```bash
cd GenMelodies
pip install -r requirements.txt
```

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
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_file` | 输入文件（.mid/.midi） | 必填 |
| `-o, --output` | 输出文件路径 | 同输入名，.txt |
| `-f, --format` | `number` / `letter` | `number` |
| `-t, --track` | MIDI 音轨索引 | 全部合并 |
| `-h, --help` | 帮助信息 | - |
| `--version` | 版本号 | - |

## 乐谱格式

兼容 [刻师傅格式](https://www.bilibili.com/video/BV1Gs4y1X7Xt?p=2)，可被 [GenLyre](https://www.bilibili.com/video/BV1Gs4y1X7Xt) 读取并自动演奏。

### 数字谱
```
0.476              ← 每拍时长（秒）
(137-1-2) /...     ← 和弦 (1 3 7 -1 -2) 同时按
                   ← / 节拍分隔，空格=休止
```

- `1~7` : 音级 (C=1, D=2, E=3, F=4, G=5, A=6, B=7)
- `+`/`-` : 升/降八度
- `()` : 和弦
- `/` : 节拍分隔
- `[]`：装饰音

### 字母谱（符号同上）
```
0.68               ← 每拍时长（秒）
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