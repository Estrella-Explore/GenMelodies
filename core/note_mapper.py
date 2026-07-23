"""
音符映射器

MIDI 音高 → 原神乐器键位（数字谱 / 字母谱）
"""

from typing import List, Tuple, Optional
from .models import MusicUnit, MeasureSlot


# ── 原神乐器 21 键布局 ──────────────────────────────────

# 字母谱键盘布局（C 大调 3 个八度）
LETTER_MAP: List[List[str]] = [
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],      # 低八度 C3-B3 (MIDI 48-54)
    ['A', 'S', 'D', 'F', 'G', 'H', 'J'],      # 中八度 C4-B4 (MIDI 60-66)
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U'],      # 高八度 C5-B5 (MIDI 72-78)
]

# 数字谱音级名
DEGREE_NAMES = ['1', '2', '3', '4', '5', '6', '7']

# MIDI 音高 → (八度偏移, 音级索引 0-6)
# C=0, D=2, E=4, F=5, G=7, A=9, B=11  (MIDI pitch mod 12)
PITCH_TO_DEGREE = {
    0: 0, 1: 0,   # C / C# → 1
    2: 1, 3: 1,   # D / D# → 2
    4: 2,         # E → 3
    5: 3, 6: 3,   # F / F# → 4
    7: 4, 8: 4,   # G / G# → 5
    9: 5, 10: 5,  # A / A# → 6
    11: 6,        # B → 7
}

# C 大调自然音（白键）
DIATONIC_MIDI = {0, 2, 4, 5, 7, 9, 11}


class NoteMapper:
    """
    将 MIDI 音高映射到原神乐器可演奏的键位

    核心逻辑：
    1. 非自然音 → 就近自然音
    2. 超出 21 键范围 → 八度折叠
    3. 输出数字谱或字母谱
    """

    # 21 键覆盖的 MIDI 音高范围（C3-B5）
    MIN_PITCH = 48   # C3
    MAX_PITCH = 83   # B5

    def __init__(self, format_type: str = 'number'):
        """
        Args:
            format_type: 'number' | 'letter'
        """
        self.format_type = format_type

    def map_units(self, units: List[MusicUnit]) -> List[MusicUnit]:
        """将 MusicUnit 列表中的 MIDI 音高映射为可演奏音高"""
        for unit in units:
            if unit.is_grace:
                # 装饰音也映射
                unit.pitches = [self._map_pitch(p) for p in unit.pitches]
            else:
                unit.pitches = [self._map_pitch(p) for p in unit.pitches]
        return units

    def to_display(self, midi_pitch: int) -> str:
        """
        单个 MIDI 音高 → 乐谱显示字符串

        Returns:
            数字谱: '1'~'7' 带 +/- 前缀
            字母谱: 'A'~'U' 单个字母
        """
        midi_pitch = self._map_pitch(midi_pitch)

        if self.format_type == 'letter':
            return self._to_letter(midi_pitch)
        else:
            return self._to_number(midi_pitch)

    # ── 类常量 ───────────────────────────────────────────

    # MIDI semitone → C大调音级索引 (0-6)
    SEMITONE_TO_DEGREE = {
        0: 0, 1: 0,   # C / C# → 1
        2: 1, 3: 1,   # D / D# → 2
        4: 2,         # E → 3
        5: 3, 6: 3,   # F / F# → 4
        7: 4, 8: 4,   # G / G# → 5
        9: 5, 10: 5,  # A / A# → 6
        11: 6,        # B → 7
    }
    # 音级索引 → MIDI semitone
    DEGREE_TO_SEMITONE = [0, 2, 4, 5, 7, 9, 11]

    # ── 内部方法 ─────────────────────────────────────────

    @classmethod
    def _map_pitch(cls, midi: int) -> int:
        """将一个 MIDI 音高映射到 21 键范围内的最接近音高"""
        # 1. 钳制到 21 键范围
        midi = max(cls.MIN_PITCH, min(cls.MAX_PITCH, midi))

        # 2. 非自然音 → 就近自然音（C大调白键）
        semitone = midi % 12
        if semitone not in DIATONIC_MIDI:
            best = midi
            best_dist = 999
            for st in DIATONIC_MIDI:
                # 向上和向下找最近自然音
                for direction in [1, -1]:
                    oct_delta = 0 if direction == 1 else -12 if semitone < st else 0
                    candidate = midi - semitone + st + oct_delta
                    dist = abs(candidate - midi)
                    if dist < best_dist:
                        best_dist = dist
                        best = candidate
            midi = best

        # 3. 确保仍然在范围内
        midi = max(cls.MIN_PITCH, min(cls.MAX_PITCH, midi))

        return midi

    def _to_number(self, midi: int) -> str:
        """MIDI 音高 → 数字谱字符串"""
        octave_offset = (midi - 60) // 12
        semitone = midi % 12
        degree_idx = self.SEMITONE_TO_DEGREE.get(semitone, 0)
        degree = DEGREE_NAMES[degree_idx]

        if octave_offset > 0:
            return '+' * octave_offset + degree
        elif octave_offset < 0:
            return '-' * abs(octave_offset) + degree
        else:
            return degree

    def _to_letter(self, midi: int) -> str:
        """MIDI 音高 → 字母谱字符串"""
        octave_idx = (midi - NoteMapper.MIN_PITCH) // 12
        octave_idx = max(0, min(2, octave_idx))
        semitone = midi % 12
        degree_idx = self.SEMITONE_TO_DEGREE.get(semitone, 0)
        return LETTER_MAP[octave_idx][degree_idx]
