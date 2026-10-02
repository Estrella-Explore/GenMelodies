"""按真实小节相位和四分音符 BPM 量化；空槽不压缩，保留演奏时间。"""

import bisect
import math
from typing import List, Tuple

from .models import MusicUnit, MeasureSlot


class Quantizer:
    def __init__(self, beats_per_measure: int = 4, slots_per_beat: int = 4,
                 beat_denominator: int = 4, grid_origin: float = 0.0):
        if beats_per_measure < 1 or slots_per_beat < 1 or beat_denominator < 1:
            raise ValueError("拍数、分母和每拍槽数必须为正整数")
        if not math.isfinite(grid_origin):
            raise ValueError("小节网格起点必须是有限数")
        self.beats_per_measure = beats_per_measure
        self.slots_per_beat = slots_per_beat
        self.beat_denominator = beat_denominator
        self.grid_origin = grid_origin

    def quantize(self, units: List[MusicUnit],
                 tempo_changes: List[Tuple[float, float]]) -> List[List[MeasureSlot]]:
        if not units:
            return []
        units = sorted(units, key=lambda unit: unit.start)
        if any(not math.isfinite(u.start) or not math.isfinite(u.end) or u.end < u.start
               for u in units):
            raise ValueError("音符时间必须有限且结束不早于开始")
        times, bpms, cumulative = self._tempo_map(tempo_changes)

        def beat_at(seconds):
            index = max(0, bisect.bisect_right(times, seconds) - 1)
            return cumulative[index] + (seconds - times[index]) * bpms[index] / 60.0

        origin = beat_at(self.grid_origin)
        bar_beats = self.beats_per_measure * 4.0 / self.beat_denominator
        n_slots = self.beats_per_measure * self.slots_per_beat
        first_bar = math.floor((beat_at(units[0].start) - origin) / bar_beats + 1e-9)
        last_beat = max(beat_at(u.end) for u in units)
        last_bar = max(first_bar, math.ceil((last_beat - origin) / bar_beats - 1e-9) - 1)
        last_bar = max(last_bar, max(math.floor((beat_at(u.start) - origin) / bar_beats + 1e-9)
                                     for u in units))
        if last_bar - first_bar + 1 > 10000:
            raise ValueError("量化超过 10000 小节，请检查 BPM 或输入时长")
        result = [[MeasureSlot(pitches=[]) for _ in range(n_slots)]
                  for _ in range(last_bar - first_bar + 1)]
        for unit in units:
            position = (beat_at(unit.start) - origin) / bar_beats
            absolute_slot = math.floor(position * n_slots + 0.5)
            bar, slot = divmod(absolute_slot, n_slots)
            if bar > last_bar:
                bar, slot = last_bar, n_slots - 1
            if bar < first_bar:
                bar, slot = first_bar, 0
            target = result[bar - first_bar][slot]
            target.pitches = sorted(set(target.pitches).union(unit.pitches))
            target.is_grace = target.is_grace or unit.is_grace
        return result

    @staticmethod
    def _tempo_map(tempo_changes):
        changes = sorted(tempo_changes or [(0.0, 120.0)])
        if any(not math.isfinite(t) or not math.isfinite(bpm) or bpm <= 0
               for t, bpm in changes):
            raise ValueError("速度必须是有限的正 BPM，速度事件时间必须有限")
        collapsed = dict(changes)
        times = sorted(collapsed)
        bpms = [collapsed[t] for t in times]
        cumulative = [0.0]
        for i in range(1, len(times)):
            cumulative.append(cumulative[-1] + (times[i] - times[i - 1]) * bpms[i - 1] / 60.0)
        return times, bpms, cumulative
