"""
音符简化器

在保留旋律的前提下，删减不适合原神乐器演奏的内容：
- 过滤极短音符（噪音）
- 合并连续相同音高
- 识别装饰音
- 限制和弦大小
"""

from typing import List
from .models import MusicUnit


class Simplifier:
    """
    音符简化器

    参数均可配置，方便调优。
    """

    def __init__(
        self,
        min_duration: float = 0.03,
        merge_threshold: float = 0.005,
        grace_threshold: float = 0.06,
        grace_ratio: float = 3.0,
        max_chord_notes: int = 5,
        extract_melody: bool = False,
        melody_voices: int = 1,
    ):
        self.min_duration = min_duration
        self.merge_threshold = merge_threshold
        self.grace_threshold = grace_threshold
        self.grace_ratio = grace_ratio
        self.max_chord_notes = max_chord_notes
        self.extract_melody = extract_melody
        self.melody_voices = melody_voices  # 旋律声部数（1=最高音, 2=最高+最低）

    def simplify(self, units: List[MusicUnit]) -> List[MusicUnit]:
        """主简化流程"""
        units = self._filter_noise(units)
        units = self._merge_same_pitch(units)
        units = self._detect_grace_notes(units)
        units = self._limit_chord_size(units)
        if self.extract_melody:
            units = self._extract_melody_voices(units)
        return units

    # ── 子步骤 ───────────────────────────────────────────

    def _filter_noise(self, units: List[MusicUnit]) -> List[MusicUnit]:
        """过滤极短音符（< min_duration，通常为噪音）"""
        return [
            u for u in units
            if (u.end - u.start) >= self.min_duration or len(u.pitches) > 1
        ]
        # 和弦即使短也保留（因为多个音符同时出现不太可能是噪音）

    def _merge_same_pitch(self, units: List[MusicUnit]) -> List[MusicUnit]:
        """
        合并碎片化的同音高音符

        仅在两个音符都极短（<50ms，MIDI解析碎片）且紧密相邻时才合并。
        正常的同音重复（如 Twinkle 的 C C）不会被合并。
        """
        if len(units) < 2:
            return units

        merged: List[MusicUnit] = [units[0]]
        for curr in units[1:]:
            prev = merged[-1]
            prev_dur = prev.end - prev.start
            curr_dur = curr.end - curr.start

            if (
                len(prev.pitches) == 1
                and len(curr.pitches) == 1
                and prev.pitches[0] == curr.pitches[0]
                and (curr.start - prev.end) < self.merge_threshold
                and prev_dur < 0.05  # 前一个音符极短 → 可能是碎片
                and curr_dur < 0.05  # 当前音符也极短 → 碎片合并
            ):
                prev.end = curr.end
            else:
                merged.append(curr)

        return merged

    def _detect_grace_notes(self, units: List[MusicUnit]) -> List[MusicUnit]:
        """
        检测装饰音

        规则：短音符（< grace_threshold）紧接一个时长 ≥ 短音符×grace_ratio 的音符，
        将短音符标记为装饰音。
        """
        for i in range(len(units) - 1):
            u = units[i]
            next_u = units[i + 1]
            dur = u.end - u.start
            next_dur = next_u.end - next_u.start

            if (
                dur < self.grace_threshold
                and (next_u.start - u.end) < self.grace_threshold
                and next_dur >= dur * self.grace_ratio
            ):
                u.is_grace = True

        return units

    def _limit_chord_size(self, units: List[MusicUnit]) -> List[MusicUnit]:
        """削减过大和弦"""
        for u in units:
            if len(u.pitches) > self.max_chord_notes and not u.is_grace:
                lo = u.pitches[0]
                hi = u.pitches[-1]
                mid = u.pitches[1:-1]
                n_mid = self.max_chord_notes - 2
                if n_mid > 0 and mid:
                    step = max(1, len(mid) // n_mid)
                    kept_mid = mid[::step][:n_mid]
                else:
                    kept_mid = []
                u.pitches = sorted([lo] + kept_mid + [hi])
        return units

    def _extract_melody_voices(self, units: List[MusicUnit]) -> List[MusicUnit]:
        """
        从多声部音符中提取旋律线

        - melody_voices=1: 每个和弦只保留最高音（单旋律）
        - melody_voices=2: 保留最高音 + 最低音（旋律+低音）
        - 单音保留不变
        """
        for u in units:
            if len(u.pitches) <= self.melody_voices or u.is_grace:
                continue
            sorted_p = sorted(u.pitches)
            if self.melody_voices == 1:
                u.pitches = [sorted_p[-1]]
            elif self.melody_voices == 2:
                u.pitches = [sorted_p[0], sorted_p[-1]]
            else:
                # 均匀抽样
                step = max(1, len(sorted_p) // self.melody_voices)
                u.pitches = sorted_p[::step][:self.melody_voices]

        return units
