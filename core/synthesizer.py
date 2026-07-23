"""
乐谱反向合成器

将刻师傅乐谱 (.txt) 反向合成为 MIDI 文件用于试听。
"""

import re
from typing import List, Tuple, Optional
import mido


# ── 键盘布局 ──────────────────────────────────────────

LETTER_MAP = [
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],   # 低八度 C3-B3
    ['A', 'S', 'D', 'F', 'G', 'H', 'J'],   # 中八度 C4-B4
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U'],   # 高八度 C5-B5
]

MIN_MIDI = 48   # C3

# degree 0-6 → MIDI semitone
DEGREE_SEMITONE = [0, 2, 4, 5, 7, 9, 11]

# 字母 → MIDI 音高
LETTER_TO_MIDI: dict = {}
for oct_i, row in enumerate(LETTER_MAP):
    for deg_i, ch in enumerate(row):
        LETTER_TO_MIDI[ch] = MIN_MIDI + oct_i * 12 + DEGREE_SEMITONE[deg_i]


class ScoreSynthesizer:
    """乐谱 → MIDI 合成器"""

    def __init__(self):
        self._events: List[Tuple[float, mido.Message]] = []
        """收集 (abs_sec, mido_message) 事件"""

    def synthesize(self, score_text: str, output_path: str, beats_per_measure: int = 4):
        """将乐谱文本合成为 MIDI 文件"""
        sections = self._parse_score(score_text)
        if not sections:
            raise ValueError("无法解析乐谱")

        self._events = []
        abs_time = 0.0

        for sec in sections:
            measure_dur = sec['measure_duration']
            bpm = 60.0 * beats_per_measure / measure_dur
            tempo_us = int(mido.bpm2tempo(bpm))

            self._events.append((abs_time,
                mido.MetaMessage('set_tempo', tempo=tempo_us, time=0)))

            for line_content in sec['lines']:
                abs_time = self._render_line(line_content, measure_dur, abs_time)

        # 构建 MIDI
        mid = mido.MidiFile(ticks_per_beat=480)
        track = self._build_track()
        mid.tracks.append(track)
        mid.save(output_path)

    def _build_track(self) -> mido.MidiTrack:
        """将收集的 (abs_sec, msg) 事件转换为 MIDI track"""
        if not self._events:
            return mido.MidiTrack()

        # 按时间排序
        sorted_events = sorted(self._events, key=lambda x: x[0])

        # 收集 tempo 变化
        tempo_changes = []
        for abs_sec, msg in sorted_events:
            if msg.type == 'set_tempo':
                tempo_changes.append((abs_sec, msg.tempo))
        if not tempo_changes:
            tempo_changes = [(0.0, 500000)]

        TPB = 480
        track = mido.MidiTrack()
        prev_sec = 0.0
        tempo_idx = 0
        current_tempo = tempo_changes[0][1]

        for abs_sec, msg in sorted_events:
            while tempo_idx + 1 < len(tempo_changes) and tempo_changes[tempo_idx + 1][0] <= abs_sec:
                tempo_idx += 1
                current_tempo = tempo_changes[tempo_idx][1]

            delta_sec = max(0.0, abs_sec - prev_sec)
            delta_tick = int(delta_sec * TPB * 1_000_000 / current_tempo)
            msg.time = max(0, delta_tick)
            track.append(msg)
            prev_sec = abs_sec

        track.append(mido.MetaMessage('end_of_track', time=0))
        return track

    # ── 解析 ────────────────────────────────────────────

    def _parse_score(self, text: str) -> List[dict]:
        """
        解析乐谱文本 → 段落列表

        每段: {'measure_duration': float, 'lines': [[slots, ...], ...]}
        """
        lines = text.strip().split('\n')
        sections: List[dict] = []
        current_measure_dur: Optional[float] = None
        current_lines: List[List[dict]] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # 空行：段落分隔（开始新行组不影响，忽略即可）
                continue

            # 检查是否为小节时长行
            try:
                dur = float(stripped)
                # 保存前一段
                if current_measure_dur is not None:
                    sections.append({
                        'measure_duration': current_measure_dur,
                        'lines': current_lines,
                    })
                current_measure_dur = dur
                current_lines = []
                continue
            except ValueError:
                pass

            # 普通乐谱行
            if current_measure_dur is None:
                current_measure_dur = 0.5  # 默认
            slots = self._parse_line(stripped)
            if slots:
                current_lines.append(slots)

        # 最后一段
        if current_measure_dur is not None:
            sections.append({
                'measure_duration': current_measure_dur,
                'lines': current_lines,
            })

        return sections

    def _parse_line(self, line: str) -> List[List[dict]]:
        """
        解析一行乐谱 → 小节列表

        每小节: [{'type': 'chord'|'note'|'rest'|'grace',
                  'pitches': [midi, ...]}]
        """
        # 按 / 分割小节
        measures = line.split('/')
        # 移除末尾空段
        if measures and measures[-1].strip() == '':
            measures = measures[:-1]

        result = []
        for measure in measures:
            slots = self._parse_measure(measure)
            if slots:
                result.append(slots)
        return result

    def _parse_measure(self, measure: str) -> List[dict]:
        """解析单小节内容 → 槽位列表"""
        slots = []
        i = 0
        n = len(measure)

        while i < n:
            ch = measure[i]

            if ch == '(':
                # 和弦
                end = measure.find(')', i)
                if end == -1:
                    end = n
                inner = measure[i+1:end]
                pitches = self._parse_notes(inner)
                slots.append({'type': 'chord', 'pitches': pitches})
                i = end + 1

            elif ch == '[':
                # 装饰音
                end = measure.find(']', i)
                if end == -1:
                    end = n
                inner = measure[i+1:end]
                pitches = self._parse_notes(inner)
                slots.append({'type': 'grace', 'pitches': pitches})
                i = end + 1

            elif ch == ' ':
                # 休止
                slots.append({'type': 'rest', 'pitches': []})
                i += 1

            elif ch in '+-':
                # 八度前缀 + 数字
                octave = ch
                i += 1
                while i < n and measure[i] in '+-':
                    octave += measure[i]
                    i += 1
                if i < n and measure[i] in '1234567':
                    midi = self._number_to_midi(octave + measure[i])
                    slots.append({'type': 'note', 'pitches': [midi]})
                    i += 1

            elif ch in '1234567':
                # 数字谱音符
                midi = self._number_to_midi(ch)
                slots.append({'type': 'note', 'pitches': [midi]})
                i += 1

            elif 'A' <= ch <= 'Z':
                # 字母谱音符
                midi = LETTER_TO_MIDI.get(ch)
                if midi is not None:
                    slots.append({'type': 'note', 'pitches': [midi]})
                i += 1

            else:
                i += 1

        return slots

    def _parse_notes(self, inner: str) -> List[int]:
        """
        解析和弦/装饰音内部字符串 → MIDI 音高列表

        支持数字谱 ('137-1-2') 和字母谱 ('VAW')
        """
        pitches = []
        i = 0
        n = len(inner)

        while i < n:
            ch = inner[i]

            if ch in '+-':
                octave = ch
                i += 1
                while i < n and inner[i] in '+-':
                    octave += inner[i]
                    i += 1
                if i < n and inner[i] in '1234567':
                    pitches.append(self._number_to_midi(octave + inner[i]))
                    i += 1

            elif ch in '1234567':
                pitches.append(self._number_to_midi(ch))
                i += 1

            elif 'A' <= ch <= 'Z':
                midi = LETTER_TO_MIDI.get(ch)
                if midi is not None:
                    pitches.append(midi)
                i += 1

            else:
                i += 1

        return sorted(set(pitches))

    @staticmethod
    def _number_to_midi(token: str) -> int:
        """数字谱 token → MIDI 音高"""
        # 统计 +/- 数量
        octave_shift = 0
        degree = None
        for ch in token:
            if ch == '+':
                octave_shift += 1
            elif ch == '-':
                octave_shift -= 1
            elif ch in '1234567':
                degree = int(ch) - 1  # 0-6
        if degree is None:
            return 60
        semitone = DEGREE_SEMITONE[degree]
        return 60 + octave_shift * 12 + semitone

    # ── MIDI 渲染 ───────────────────────────────────────

    def _render_line(
        self, line: List[List[dict]], measure_dur: float, abs_time: float
    ) -> float:
        """渲染一行乐谱 → 收集事件，返回新的绝对时间"""
        for measure in line:
            abs_time = self._render_measure(measure, measure_dur, abs_time)
        return abs_time

    def _render_measure(
        self, slots: List[dict], measure_dur: float, abs_time: float
    ) -> float:
        """渲染单小节 → 收集 MIDI 事件"""
        if not slots:
            return abs_time + measure_dur

        n_slots = len(slots)
        slot_dur = measure_dur / n_slots
        grace_dur = min(0.04, slot_dur * 0.3)

        for i, slot in enumerate(slots):
            slot_start = abs_time + i * slot_dur

            if slot['type'] == 'rest' or not slot['pitches']:
                continue

            if slot['type'] == 'grace':
                for p in slot['pitches']:
                    self._emit_note(p, slot_start, grace_dur)
            else:
                note_dur = min(slot_dur * 0.85, 2.0)
                for p in slot['pitches']:
                    self._emit_note(p, slot_start, note_dur)

        return abs_time + measure_dur

    def _emit_note(
        self, pitch: int, start_sec: float, duration_sec: float,
        velocity: int = 80
    ):
        """发出一个音符事件（存入事件列表）"""
        self._events.append((start_sec,
            mido.Message('note_on', note=pitch, velocity=velocity, time=0)))
        self._events.append((start_sec + duration_sec,
            mido.Message('note_off', note=pitch, velocity=0, time=0)))

    # ── 废弃 ─────────────────────────────────────────

    def _finalize_timing(self, track):
        pass  # 不再使用
