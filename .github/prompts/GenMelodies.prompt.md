---
name: GenMelodies
description: Convert MIDI files into automatically playable sheet music for Genshin Impact instruments.
argument-hint: Provide a MIDI file to generate sheet music in numeric or letter format.
agent: agent
---

# GenMelodies

将 MIDI 文件转换为**原神乐器**可用的自动演奏乐谱。

## 语法

```bash
python genmelodies.py <input_file> [options]
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_file` | 输入文件（.mid/.midi） | 必填 |
| `-o, --output` | 输出文件路径 | 同输入名，.txt |
| `-f, --format` | `number` / `letter` | `number` |
| `-t, --track` | 音轨索引 | 全部合并 |
| `--no-simplify` | 禁用简化 | 关闭 |
| `-h, --help` | 帮助 | - |
| `--version` | 版本号 | - |

### 示例

```bash
python genmelodies.py song.mid
python genmelodies.py song.mid --format letter
python genmelodies.py song.mid -o output.txt
python genmelodies.py song.mid --track 1
```

## 乐谱格式

兼容 [刻师傅格式](https://www.bilibili.com/video/BV1Gs4y1X7Xt?p=2)，可被 [GenLyre](https://www.bilibili.com/video/BV1Gs4y1X7Xt) 读取并自动演奏。

### 数字谱
```
0.476              ← 每小节时长（秒）
(137-1-2) /...     ← 和弦 / 节拍分隔
```

- `1~7` : 音级 (C~B)，`+`/`-` : 八度，`()` : 和弦，`/` : 节分隔，空格：休止，`[]`：装饰音

### 字母谱
```
0.68
(VAW) E /T Q /...
```

| 高 | Q | W | E | R | T | Y | U |
| 中 | A | S | D | F | G | H | J |
| 低 | Z | X | C | V | B | N | M |

分别对应 C 大调 3 个八度。