"""
节奏量化器

将自由时间的 MusicUnit 对齐到小节/拍网格，支持变速。
"""

from typing import List, Tuple
from .models import MusicUnit, MeasureSlot


class Quantizer:
    """节奏量化器，支持变速"""

    def __init__(
        self,
        beats_per_measure: int = 4,
        slots_per_beat: int = 4,  # 16 分音符精度
    ):
        self.beats_per_measure = beats_per_measure
        self.slots_per_beat = slots_per_beat

    def quantize(
        self,
        units: List[MusicUnit],
        tempo_changes: List[Tuple[float, float]],  # [(time_sec, bpm), ...]
    ) -> List[List[MeasureSlot]]:
        """将 MusicUnit 列表量化为小节网格"""
        if not units:
            return []

        if not tempo_changes:
            tempo_changes = [(0.0, 120.0)]

        # ── 构建变速感知的小节边界 ──
        measure_starts = self._build_measure_boundaries(units, tempo_changes)

        # ── 裁剪前导静音：从第一个有音符的小节开始 ──
        first_note = units[0].start
        trim_idx = 0
        for i, m_start in enumerate(measure_starts):
            if i + 1 < len(measure_starts) and measure_starts[i + 1] > first_note:
                trim_idx = i
                break
        measure_starts = measure_starts[trim_idx:]

        # ── 将音符分配到小节 ──
        measures_units: List[List[MusicUnit]] = []
        current: List[MusicUnit] = []
        m_idx = 0

        for u in units:
            while m_idx + 1 < len(measure_starts) and u.start >= measure_starts[m_idx + 1]:
                measures_units.append(current)
                current = []
                m_idx += 1

            m_start = measure_starts[m_idx] if m_idx < len(measure_starts) else 0.0
            rel_unit = MusicUnit(
                pitches=list(u.pitches),
                start=u.start - m_start,
                end=u.end - m_start,
                is_grace=u.is_grace,
            )
            current.append(rel_unit)

        if current:
            measures_units.append(current)

        # ── 每个小节 → 槽位序列 ──
        result: List[List[MeasureSlot]] = []
        for i, m_units in enumerate(measures_units):
            m_start = measure_starts[i] if i < len(measure_starts) else 0.0
            m_end = measure_starts[i + 1] if i + 1 < len(measure_starts) else m_start + 2.0
            m_dur = m_end - m_start
            slots = self._build_slots(m_units, m_dur)
            result.append(slots)

        return result

    def _build_measure_boundaries(
        self,
        units: List[MusicUnit],
        tempo_changes: List[Tuple[float, float]],
    ) -> List[float]:
        """根据变速信息构建小节边界时间点列表"""
        tempo_changes = sorted(tempo_changes, key=lambda x: x[0])

        first_note = units[0].start
        last_note = max(u.end for u in units)

        boundaries = []
        current_time = first_note
        tc_idx = 0
        current_bpm = tempo_changes[0][1]

        while current_time < last_note + 10.0:
            while tc_idx + 1 < len(tempo_changes) and tempo_changes[tc_idx + 1][0] <= current_time:
                tc_idx += 1
                current_bpm = tempo_changes[tc_idx][1]

            boundaries.append(current_time)
            measure_dur = (60.0 / current_bpm) * self.beats_per_measure
            current_time += measure_dur

            if len(boundaries) > 10000:
                break

        return boundaries

    def _build_slots(
        self, units: List[MusicUnit], measure_dur: float
    ) -> List[MeasureSlot]:
        """固定槽数小节网格"""
        n_slots = self.beats_per_measure * self.slots_per_beat
        slot_width = measure_dur / n_slots

        if not units:
            return [MeasureSlot(pitches=[], is_grace=False)]

        occupied = [False] * n_slots
        unit_to_slot: dict = {}
        for u in units:
            si = int(u.start / slot_width)
            ei = int(u.end / slot_width)
            si = max(0, min(n_slots - 1, si))
            ei = max(si, min(n_slots - 1, ei))
            unit_to_slot.setdefault(si, []).append(u)
            for oi in range(si, ei + 1):
                occupied[oi] = True

        slots: List[MeasureSlot] = []
        rest_run = 0
        for si in range(n_slots):
            if si in unit_to_slot:
                if rest_run > 0:
                    slots.append(MeasureSlot(pitches=[], is_grace=False))
                    rest_run = 0
                slot_units = unit_to_slot[si]
                pitches: List[int] = []
                is_grace = False
                for su in slot_units:
                    pitches.extend(su.pitches)
                    if su.is_grace:
                        is_grace = True
                slots.append(MeasureSlot(
                    pitches=sorted(set(pitches)),
                    is_grace=is_grace,
                ))
            elif occupied[si]:
                rest_run += 1
            else:
                rest_run += 1

        if rest_run > 0:
            slots.append(MeasureSlot(pitches=[], is_grace=False))

        return slots
