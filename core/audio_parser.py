"""音频入口：MIT 解码；节奏分析与粗略单音音高提取互不依赖。"""

from typing import Optional

import numpy as np

from .models import Note, ParsedPiece, ParsedTrack
from .audio_rhythm import analyze_samples
from .audio_decode import decode_audio


AUDIO_EXTENSIONS = ('.mp3', '.wav')


class AudioParser:
    def decode(self, filepath: str):
        return decode_audio(filepath)

    def analyze_file(self, filepath: str, *, bpm=None, time_signature=None, beat_offset=None):
        samples, sample_rate = self.decode(filepath)
        return analyze_samples(samples, sample_rate, bpm=bpm,
                               time_signature=time_signature, beat_offset=beat_offset)

    def parse(self, filepath: str, track_index: Optional[int] = None, *,
              bpm=None, time_signature=None, beat_offset=None) -> ParsedPiece:
        if track_index is not None:
            raise ValueError("音频没有 MIDI 轨道索引；请移除 --track")
        samples, sample_rate = self.decode(filepath)
        rhythm = analyze_samples(samples, sample_rate, bpm=bpm,
                                 time_signature=time_signature, beat_offset=beat_offset)
        return self.parse_samples(samples, sample_rate, rhythm)

    def parse_samples(self, samples, sample_rate, rhythm) -> ParsedPiece:
        if rhythm.bpm is None:
            raise ValueError("无法可靠估计 BPM；先用 --analyze-rhythm 检查，再用 --bpm 指定四分音符 BPM")
        if rhythm.time_signature is None:
            raise ValueError("无法可靠确定拍号；先用 --analyze-rhythm 检查，再用 --time-signature 指定，例如 4/4")
        notes = self.estimate_monophonic_notes(samples, sample_rate)
        track = ParsedTrack(notes=notes, tempo=rhythm.bpm, time_signature=rhythm.time_signature,
                            tempo_changes=[(0.0, rhythm.bpm)])
        return ParsedPiece(tracks=[track], merged_notes=notes, global_tempo=rhythm.bpm,
                           global_time_signature=rhythm.time_signature,
                           grid_origin=rhythm.grid_origin, rhythm_analysis=rhythm.to_dict())

    @staticmethod
    def estimate_monophonic_notes(samples, sample_rate):
        """YIN 差分/FFT 自相关的简易单音提取。仅适合独奏，不做和弦或人声分离。"""
        signal = np.asarray(samples, dtype=np.float64)
        if signal.ndim == 2 and signal.shape[1] > 0:
            # 取能量最大的声道，避免反相立体声混成零。
            signal = signal[:, int(np.argmax(np.mean(signal ** 2, axis=0)))]
        if signal.ndim != 1 or sample_rate <= 0 or not np.all(np.isfinite(signal)):
            raise ValueError("音高提取需要有限单声道采样和正采样率")
        frame_size = max(1024, 2 ** int(np.ceil(np.log2(sample_rate * 0.09))))
        hop = max(1, int(sample_rate * 0.01))
        if len(signal) < frame_size:
            return []
        min_lag = max(2, int(sample_rate / 1200))
        max_lag = min(frame_size // 2, int(sample_rate / 65))
        pitches, amplitudes = [], []
        threshold = max(0.0005, float(np.max(np.abs(signal))) * 0.012)
        for start in range(0, len(signal) - frame_size + 1, hop):
            frame = signal[start:start + frame_size]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            amplitudes.append(rms)
            if rms < threshold:
                pitches.append(-1)
                continue
            spectrum = np.fft.rfft(frame, n=2 * frame_size)
            autocorrelation = np.fft.irfft(np.abs(spectrum) ** 2)[:max_lag + 1]
            energy = np.concatenate(([0.0], np.cumsum(frame * frame)))
            lags = np.arange(max_lag + 1)
            difference = np.maximum(energy[frame_size - lags] + energy[frame_size]
                                    - energy[lags] - 2 * autocorrelation, 0)
            cmnd = np.ones_like(difference)
            cmnd[1:] = difference[1:] * lags[1:] / np.maximum(np.cumsum(difference[1:]), 1e-20)
            candidates = np.flatnonzero(cmnd[min_lag:max_lag] < 0.18) + min_lag
            if not len(candidates):
                pitches.append(-1)
                continue
            lag = int(candidates[0])
            while lag + 1 < max_lag and cmnd[lag + 1] < cmnd[lag]:
                lag += 1
            y0, y1, y2 = cmnd[lag - 1:lag + 2]
            curvature = y0 - 2 * y1 + y2
            refined = lag + (0.5 * (y0 - y2) / curvature if abs(curvature) > 1e-12 else 0)
            midi = int(round(69 + 12 * np.log2(sample_rate / refined / 440)))
            pitches.append(midi if 36 <= midi <= 96 else -1)
        smoothed = list(pitches)
        for i in range(1, len(pitches) - 1):
            if pitches[i - 1] == pitches[i + 1] and pitches[i] >= 0:
                smoothed[i] = pitches[i - 1]
        notes = []
        start = 0
        for index in range(1, len(smoothed) + 1):
            if index < len(smoothed) and smoothed[index] == smoothed[start]:
                continue
            pitch = smoothed[start]
            if pitch >= 0 and (index - start) * hop / sample_rate >= 0.06:
                note_start = max(0.0, (start * hop + frame_size / 2 - hop / 2) / sample_rate)
                note_end = min(len(signal) / sample_rate,
                               (index * hop + frame_size / 2 - hop / 2) / sample_rate)
                level = float(np.mean(amplitudes[start:index]))
                velocity = min(127, max(30, int(80 + 20 * np.log10(max(level, 1e-6) / 0.1))))
                notes.append(Note(pitch=pitch, start=note_start, end=note_end, velocity=velocity))
            start = index
        return notes
