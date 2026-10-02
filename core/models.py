"""
GenMelodies - 数据模型
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Note:
    """单个音符"""
    pitch: int          # MIDI 音高 (0-127, 60=C4)
    start: float        # 起始时间（秒）
    end: float          # 结束时间（秒）
    velocity: int = 80  # 力度


@dataclass
class ParsedTrack:
    """解析后的音轨"""
    notes: List[Note]
    tempo: float = 120.0                       # BPM
    time_signature: Tuple[int, int] = (4, 4)   # (分子, 分母)
    tempo_changes: List[Tuple[float, float]] = field(default_factory=list)
    # tempo_changes: [(time_sec, bpm), ...]
    time_sig_changes: List[Tuple[float, int, int]] = field(default_factory=list)
    # time_sig_changes: [(time_sec, numerator, denominator), ...]


@dataclass
class ParsedPiece:
    """解析后的完整乐曲"""
    tracks: List[ParsedTrack]
    ticks_per_beat: int = 480
    # 合并所有音轨后的一维音符列表（按时间排序）
    merged_notes: List[Note] = field(default_factory=list)
    # 合并后的速度/拍号信息
    global_tempo: float = 120.0
    global_time_signature: Tuple[int, int] = (4, 4)
    # 小节网格起点（秒）；MIDI 默认以文件零点为基准，音频由下拍估计决定。
    grid_origin: float = 0.0
    rhythm_analysis: Optional[dict] = None


# ── 流水线中间类型 ──────────────────────────────────────


@dataclass
class MusicUnit:
    """
    一个音乐时间单元：单音 或 和弦
    
    - 单音: pitches 长度为 1
    - 和弦: pitches 长度 >= 2
    """
    pitches: List[int]       # MIDI 音高列表
    start: float             # 单元起始时间
    end: float               # 单元结束时间
    is_grace: bool = False   # 是否为装饰音


@dataclass
class MeasureSlot:
    """
    一个小节内的一个时间槽
    
    - CHORD:  多个音符同时按下
    - NOTE:   单个音符
    - REST:   休止符（pitches 为空）
    - GRACE:  装饰音（快速奏出后紧跟主音）
    """
    pitches: List[int]       # 映射后的音高/键位（空=休止）
    is_grace: bool = False
