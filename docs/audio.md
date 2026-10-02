# MP3 / WAV 节奏分析

音频入口支持 MP3 和未压缩的整数 PCM WAV（8/16/24/32 位）。Python 依赖沿用现有的 NumPy、mido；没有 librosa、SciPy、机器学习权重、ffmpeg 运行依赖或联网服务。

MP3 用项目内的 `dr_mp3` C 头文件和小型 ctypes 桥接解码。首次读取 MP3 会尝试用本机的 GCC、Clang 或 MSVC 构建动态库，随后复用 `.audio-native/` 缓存。需要 Python 3.10+；如果缺少编译器，安装编译器或在配置好编译环境的终端运行下面命令。WAV 不需要编译器。

```bash
pip install -r requirements.txt
python tools/build_audio_decoder.py
python genmelodies.py song.mp3 --analyze-rhythm
python genmelodies.py song.mp3 --analyze-rhythm --json > rhythm.json
python genmelodies.py song.mp3 --analyze-rhythm -o rhythm.json
```

`--json` 保证 stdout 只有 JSON；错误和警告写入 stderr。分析成功表示完成分析，即使 BPM 或拍号为未知也会退出 0；JSON 中未知字段是 `null`。解码失败和输入错误退出 1，参数格式错误退出 2。未知结果不会伪装成 120 BPM 或 4/4。

## 读懂结果与人工校正

自动速度候选搜索 40–240 脉冲 BPM；确认复合拍号后会换算成四分音符 BPM。人工 `--bpm` 的有效范围为 20–400 四分音符 BPM（包含端点），音频与 MIDI 入口采用相同的参数校验；非有限值或超出范围会在读取文件前报参数错误。

速度检测基于原始 PCM 的多频段起音信息、周期性候选和拍点跟踪，拍号基于持续的重音分组证据。候选与置信指标供比较使用，指标不是经过真实歌曲校准的正确率。只凭声音不能总是唯一确定记谱方式：均匀点击不能区分 3/4 和 4/4，强弱拍、切分、半拍伴奏也可能造成倍速/半速歧义。因此必须结合听感检查候选和警告；弱证据时拍号留空。

```bash
# 已知速度或拍号时人工覆盖；节奏拍点仍从音频估计
python genmelodies.py song.mp3 --analyze-rhythm --bpm 126 --time-signature 4/4 --json
# 0.37 秒是第一小节下拍，以此校正乐谱网格
python genmelodies.py song.mp3 --analyze-rhythm --bpm 126 --time-signature 4/4 --beat-offset 0.37 --json
```

所有 BPM 都以**四分音符**为单位。6/8 的 `--bpm 120` 表示一小节 1.5 秒、附点四分音符脉冲间隔 0.75 秒；若习惯把 6/8 的大拍记成 80 BPM，应输入四分音符 BPM 120。`beat_unit` 表示拍点列表的单位：`quarter` 或 `dotted-quarter`；`bpm_unit` 始终为 `quarter-note`。其余简单人工拍号返回四分音符脉冲，小节下拍间距仍根据拍号分母计算；拍点的单位以 `beat_unit` 为准。

JSON 的主要字段：

| 字段 | 含义 |
|---|---|
| `bpm` | 四分音符速度，证据不足时为 `null` |
| `time_signature` | `[分子, 分母]`，不确定时为 `null` |
| `tempo_candidates` | 倍速、半速等竞争解释及得分 |
| `meter_candidates` | 拍号候选、得分、下拍位置、重音一致性与三连细分证据；置信指标为启发式指标 |
| `tempo_confidence`, `meter_confidence` | 内部证据指标，不是概率 |
| `beat_times`, `downbeat_times` | 拍点和下拍时间（秒） |
| `grid_origin` | 小节网格原点（秒），已知拍号时优先用下拍 |
| `beat_unit`, `bpm_unit` | 拍点与 BPM 的单位 |
| `duration`, `warnings` | 解码时长与不确定性说明 |

## 从音频生成乐谱

```bash
python genmelodies.py solo.mp3 --bpm 120 --time-signature 3/4 --rhythm-json solo.rhythm.json
python genmelodies.py solo.wav --bpm 120 --time-signature 6/8 --beat-offset 0.25 --synthesize
```

这一入口的音高提取只是基于 FFT 自相关 / YIN 差分的**粗略单音**估计，适合独奏或干净单旋律。完整歌曲的人声与伴奏、和弦、鼓和混响会导致误识别，`--extract-melody` 不会对音频做声部分离。音高失败不会影响独立的 `--analyze-rhythm` 分析。生成乐谱要求速度和拍号都已知；不确定时会提示相应覆盖参数。`--rhythm-json` 会在乐谱生成前保存诊断，后续音高失败也能保留诊断文件。

量化使用下拍原点和分母正确的小节长度，保留小节内休止及整小节休止；只裁剪完整的前导空小节。文本谱的空格也是时间槽，不要让编辑器清理行首或行尾空格。当前输出格式只有一个小节时长，音频乐谱按全曲固定 BPM 近似，渐变速度、现场自由速度、变拍号可能逐渐错位；JSON 拍点可以用于检查这种偏差。MIDI 的速度事件参与量化，但文本播放仍只能按首个速度近似。

## 依赖许可

新引入的第三方音频源码只有 [dr_mp3](https://github.com/mackron/dr_libs/blob/master/dr_mp3.h)，按文件中的 MIT No Attribution（MIT-0）授权使用，保留其许可和版本来源说明；它不引入 Python 传递依赖。桥接、节奏分析及单音提取均为项目内原创实现。项目自身仍采用 GPLv3；本次约束是新增外部仓库许可，不会改变项目原有许可。

```bash
python -m unittest discover -s tests -v
```
