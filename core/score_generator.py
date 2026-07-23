"""
乐谱生成器

将量化后的小节数据格式化为刻师傅乐谱文本。
"""

from typing import List
from .models import MeasureSlot
from .note_mapper import NoteMapper


class ScoreGenerator:
    """乐谱文本生成器"""

    def __init__(self, format_type: str = 'number'):
        """
        Args:
            format_type: 'number' | 'letter'
        """
        self.format_type = format_type
        self.mapper = NoteMapper(format_type=format_type)

    def generate(
        self,
        measures: List[List[MeasureSlot]],
        measure_duration: float,        # 每小节时长（秒）
        beats_per_measure: int = 4,
    ) -> str:
        """
        生成完整乐谱文本

        Args:
            measures: 量化后的小节列表
            measure_duration: 每小节时长（秒）
            beats_per_measure: 每小节拍数（保留，备用）

        Returns:
            乐谱文本
        """
        lines: List[str] = []

        # 第一行：小节时长
        lines.append(str(measure_duration))

        # 每个小节一行（过长则折行）
        # 简化：所有小节连成一行，但遵循合理长度
        current_line_parts: List[str] = []
        chars_in_line = 0
        MAX_LINE_CHARS = 80

        for measure in measures:
            measure_str = self._format_measure(measure)
            if chars_in_line + len(measure_str) > MAX_LINE_CHARS and current_line_parts:
                lines.append(''.join(current_line_parts))
                current_line_parts = []
                chars_in_line = 0
            current_line_parts.append(measure_str)
            chars_in_line += len(measure_str)

        if current_line_parts:
            lines.append(''.join(current_line_parts))

        return '\n'.join(lines) + '\n'

    def _format_measure(self, slots: List[MeasureSlot]) -> str:
        """格式化单小节"""
        if not slots:
            return '/'

        parts: List[str] = []
        for slot in slots:
            parts.append(self._format_slot(slot))

        return ''.join(parts) + '/'

    def _format_slot(self, slot: MeasureSlot) -> str:
        """格式化单个时间槽"""
        if not slot.pitches:
            # 休止符
            return ' '

        # 音符 / 和弦
        if len(slot.pitches) == 1:
            content = self.mapper.to_display(slot.pitches[0])
        else:
            # 和弦
            inner = ''.join(
                self.mapper.to_display(p) for p in sorted(slot.pitches)
            )
            if slot.is_grace:
                content = f'[{inner}]'
            else:
                content = f'({inner})'

        return content
