"""
MIDI 文件解析器

支持 .mid / .midi 格式，提取音符、速度、拍号信息。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
import bisect
import mido

from .models import Note, ParsedTrack, ParsedPiece


class BaseParser(ABC):
    """解析器抽象基类——未来音频解析器实现同一接口"""

    @abstractmethod
    def parse(self, filepath: str, track_index: Optional[int] = None) -> ParsedPiece:
        """解析音乐文件 → ParsedPiece"""
        ...


class MidiParser(BaseParser):
    """MIDI 文件解析器"""

    def parse(self, filepath: str, track_index: Optional[int] = None) -> ParsedPiece:
        mid = mido.MidiFile(filepath)
        tpb = mid.ticks_per_beat

        # ── 第一遍：扫描全部轨道收集 tempo / time_sig ──
        tempo_events: List[Tuple[int, int]] = []
        time_sig_events: List[Tuple[int, int, int]] = []

        for track in mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == 'set_tempo':
                    tempo_events.append((abs_tick, msg.tempo))
                elif msg.type == 'time_signature':
                    time_sig_events.append((abs_tick, msg.numerator, msg.denominator))

        if not tempo_events:
            tempo_events = [(0, 500000)]
        # 去重：同一 tick 的重复事件只保留第一个
        seen_ticks = set()
        unique_tempo = []
        for tick, tempo in sorted(tempo_events):
            if tick not in seen_ticks:
                unique_tempo.append((tick, tempo))
                seen_ticks.add(tick)
        tempo_events = unique_tempo

        first_bpm = mido.tempo2bpm(tempo_events[0][1])
        first_time_sig = time_sig_events[0][1:] if time_sig_events else (4, 4)

        tempo_map = self._build_tempo_map(tempo_events, tpb)

        # ── 第二遍：提取所有轨道音符 ──
        if track_index is not None:
            # 用户指定轨道
            if track_index < len(mid.tracks):
                target = [mid.tracks[track_index]]
            else:
                target = [mid.tracks[-1]]
        else:
            # 合并所有含音符的轨道
            target = [
                t for t in mid.tracks
                if any(m.type in ('note_on', 'note_off') for m in t)
            ]
            if not target:
                target = mid.tracks

        all_notes: List[Note] = []
        for track in target:
            notes = self._extract_notes(track, tempo_map, tpb)
            all_notes.extend(notes)

        all_notes.sort(key=lambda n: (n.start, n.pitch))

        # tempo/time_sig 变化 → 秒
        tempo_changes_sec = [
            (self._tick_to_sec(tick, tempo_map, tpb), mido.tempo2bpm(us))
            for tick, us in tempo_events
        ]
        time_sig_changes_sec = [
            (self._tick_to_sec(tick, tempo_map, tpb), num, den)
            for tick, num, den in time_sig_events
        ]

        track_obj = ParsedTrack(
            notes=all_notes,
            tempo=first_bpm,
            time_signature=first_time_sig,
            tempo_changes=tempo_changes_sec,
            time_sig_changes=time_sig_changes_sec,
        )

        return ParsedPiece(
            tracks=[track_obj],
            ticks_per_beat=tpb,
            merged_notes=all_notes,
            global_tempo=first_bpm,
            global_time_signature=first_time_sig,
        )

    # ── 内部方法 ──────────────────────────────────────────

    @staticmethod
    def _build_tempo_map(
        tempo_events: List[Tuple[int, int]], tpb: int
    ) -> List[Tuple[int, float, int]]:
        """构建 (tick, cumulative_seconds, tempo_us) 映射"""
        result = []
        cum = 0.0
        prev_tick = 0
        prev_tempo = 500000

        for tick, tempo_us in tempo_events:
            delta = tick - prev_tick
            cum += (delta * prev_tempo) / (tpb * 1_000_000)
            result.append((tick, cum, prev_tempo))
            prev_tick = tick
            prev_tempo = tempo_us

        # 追加一个"终点"标记，方便后续计算
        result.append((float('inf'), cum, prev_tempo))
        return result

    @staticmethod
    def _tick_to_sec(tick: int, tmap: List[Tuple[int, float, int]], tpb: int = 480) -> float:
        """tick → 秒"""
        if not tmap:
            return tick * 0.0005
        ticks = [t for t, _, _ in tmap]
        idx = bisect.bisect_right(ticks, tick) - 1
        if idx < 0:
            return 0.0
        base_tick, base_sec, tempo_us = tmap[idx]
        return base_sec + (tick - base_tick) * tempo_us / (tpb * 1_000_000)

    @staticmethod
    def _extract_notes(
        track: mido.MidiTrack,
        tmap: List[Tuple[int, float, int]],
        tpb: int = 480,
    ) -> List[Note]:
        """从 MIDI 轨道提取音符"""
        notes = []
        abs_tick = 0
        active: Dict[int, Tuple[int, int]] = {}

        for msg in track:
            abs_tick += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                active[msg.note] = (abs_tick, msg.velocity)

            elif msg.type == 'note_off' or (
                msg.type == 'note_on' and msg.velocity == 0
            ):
                if msg.note in active:
                    st_tick, vel = active.pop(msg.note)
                    start_sec = MidiParser._tick_to_sec(st_tick, tmap, tpb)
                    end_sec = MidiParser._tick_to_sec(abs_tick, tmap, tpb)
                    if end_sec > start_sec:
                        notes.append(Note(
                            pitch=msg.note, start=start_sec,
                            end=end_sec, velocity=vel,
                        ))

        # 未关闭的音符给默认时长
        for pitch, (st_tick, vel) in active.items():
            st = MidiParser._tick_to_sec(st_tick, tmap, tpb)
            notes.append(Note(pitch=pitch, start=st, end=st + 0.5, velocity=vel))

        return notes
