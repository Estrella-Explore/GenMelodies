"""
和弦检测器

将时间上接近的音符分组为和弦。
"""

from typing import List, Tuple
from .models import Note, MusicUnit


class ChordDetector:
    """
    基于时间窗口的和弦检测

    起始时间差 < time_window 的音符视为同一和弦。
    """

    def __init__(self, time_window: float = 0.02):
        """
        Args:
            time_window: 时间窗口（秒），音符起始时间差在此范围内视为同时
                         20ms 适合 MIDI；音频可适当增大到 40-60ms
        """
        self.time_window = time_window

    def detect(self, notes: List[Note]) -> List[MusicUnit]:
        """
        检测和弦并将音符分组为 MusicUnit 列表

        Args:
            notes: 按 start 排序的音符列表

        Returns:
            MusicUnit 列表（按时间排序，每个 Unit 是单音或和弦）
        """
        if not notes:
            return []

        units: List[MusicUnit] = []
        current_group: List[Note] = [notes[0]]
        group_start = notes[0].start

        for note in notes[1:]:
            if note.start - group_start <= self.time_window:
                # 仍在同一和弦窗口内
                current_group.append(note)
            else:
                # 结束当前组，开始新组
                units.append(self._make_unit(current_group))
                current_group = [note]
                group_start = note.start

        # 最后一组
        units.append(self._make_unit(current_group))

        return units

    @staticmethod
    def _make_unit(group: List[Note]) -> MusicUnit:
        """一组音符 → MusicUnit"""
        pitches = [n.pitch for n in group]
        # 和弦按音高排序（低→高）
        pitches.sort()
        return MusicUnit(
            pitches=pitches,
            start=group[0].start,
            end=max(n.end for n in group),
        )
