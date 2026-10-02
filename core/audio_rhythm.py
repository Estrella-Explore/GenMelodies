"""Original, NumPy-only beat and meter estimation.

Audio does not uniquely encode notation: an unaccented click track cannot tell
3/4 from 4/4, and a dotted-quarter pulse need not imply 6/8.  The estimator
therefore reports candidates and can abstain instead of manufacturing a meter.
BPM is always quarter notes per minute, including compound meter.  Beat times
in 6/8 describe dotted-quarter conducting pulses, three eighth notes apart.
"""
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AudioRhythmResult:
    """Confidence fields are heuristic evidence scores, not probabilities."""
    bpm: Optional[float] = None
    time_signature: Optional[tuple[int, int]] = None
    beat_times: list[float] = field(default_factory=list)
    downbeat_times: list[float] = field(default_factory=list)
    grid_origin: float = 0.0
    tempo_candidates: list[dict] = field(default_factory=list)
    meter_candidates: list[dict] = field(default_factory=list)
    tempo_confidence: float = 0.0
    meter_confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    beat_unit: str = "quarter"
    bpm_unit: str = "quarter-note"
    duration: float = 0.0

    @property
    def beat_offset(self) -> Optional[float]:
        return self.beat_times[0] if self.beat_times else None

    @property
    def downbeat_offset(self) -> Optional[float]:
        return self.downbeat_times[0] if self.downbeat_times else None

    def to_dict(self) -> dict:
        return asdict(self)


def _mono(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples)
    if audio.ndim not in (1, 2) or (audio.ndim == 2 and audio.shape[1] not in (1, 2)):
        raise ValueError("samples must be mono or sample-major stereo audio")
    if not np.issubdtype(audio.dtype, np.number):
        raise ValueError("samples must be numeric")
    if np.issubdtype(audio.dtype, np.complexfloating):
        raise ValueError("samples must be real audio")
    if not np.all(np.isfinite(audio)):
        raise ValueError("samples contain NaN or infinity")
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float64)
        if info.min == 0:
            audio -= (info.max + 1) / 2
        audio /= max(abs(info.min), info.max + 1)
    else:
        audio = audio.astype(np.float64, copy=False)
    if audio.ndim == 2:
        combined = np.mean(audio, axis=1)
        # Preserve rhythm in recordings whose stereo channels cancel in mono.
        energy = np.mean(audio * audio, axis=0) if len(audio) else np.zeros(audio.shape[1])
        if len(audio) and np.mean(combined * combined) < .1 * np.max(energy):
            combined = audio[:, int(np.argmax(energy))]
        audio = combined
    if len(audio):
        audio = audio - np.mean(audio)
    return audio


def _reduce_rate(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    """FIR low-pass before integer decimation; no external resampler required."""
    factor = max(1, int(sample_rate // 11025))
    if factor == 1:
        return audio, float(sample_rate)
    half = 12 * factor
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.sinc(x / factor) * np.hanning(len(x))
    kernel /= np.sum(kernel)
    # Short FIR and decimation keep STFT cost independent of input sample rate.
    return np.convolve(audio, kernel, mode="same")[::factor], sample_rate / factor


def _features(audio: np.ndarray, sample_rate: float) -> tuple[np.ndarray, float]:
    size = 1024 if sample_rate >= 8000 else 512
    hop = max(1, int(round(sample_rate / 100)))
    padded = np.pad(audio, (size // 2, size // 2))
    count = 1 + (len(padded) - size) // hop
    frames = np.lib.stride_tricks.sliding_window_view(padded, size)[::hop]
    window = np.hanning(size)
    frequency = np.fft.rfftfreq(size, 1 / sample_rate)
    bands = [(frequency >= lo) & (frequency < hi) for lo, hi in
             ((35, 220), (220, 2000), (2000, sample_rate / 2 + 1))]
    feature = np.zeros((count, 3), dtype=np.float64)
    previous = np.zeros(size // 2 + 1)
    for start in range(0, count, 256):
        spectrum = np.abs(np.fft.rfft(frames[start:start + 256] * window, axis=1))
        # Square-root compression retains soft attacks next to loud drums.
        spectrum = np.sqrt(spectrum)
        differences = np.maximum(np.diff(np.vstack((previous, spectrum)), axis=0), 0)
        previous = spectrum[-1]
        for index, band in enumerate(bands):
            feature[start:start + len(spectrum), index] = np.sum(differences[:, band], axis=1)
    # Remove slow changes and the stationary floor independently in each band.
    radius = 15
    for index in range(3):
        values = feature[:, index]
        # The recording boundary represents silence before the first sample.
        # Replicating the first attack as a median floor would erase beat zero.
        local = np.lib.stride_tricks.sliding_window_view(np.pad(values, radius), 2 * radius + 1)
        values = np.maximum(values - np.median(local, axis=1), 0)
        scale = np.percentile(values, 95)
        feature[:, index] = np.minimum(values / max(scale, 1e-12), 5)
    return feature, hop / sample_rate


def _peaks(feature: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    envelope = feature @ np.array([.36, .34, .30])
    envelope = np.convolve(envelope, np.array([.2, .6, .2]), mode="same")
    threshold = max(.12, float(np.percentile(envelope, 70)) * .6)
    candidates = np.flatnonzero((envelope[1:-1] >= envelope[:-2]) &
                               (envelope[1:-1] > envelope[2:]) &
                               (envelope[1:-1] > threshold)) + 1
    if len(envelope) > 1 and envelope[0] > envelope[1] and envelope[0] > threshold:
        candidates = np.r_[0, candidates]
    # Greedy suppression keeps flams from becoming two equal onsets.
    keep = []
    gap = max(1, round(.065 / step))
    for peak in candidates:
        if keep and peak - keep[-1] < gap:
            if envelope[peak] > envelope[keep[-1]]:
                keep[-1] = int(peak)
        else:
            keep.append(int(peak))
    indices = np.asarray(keep, dtype=int)
    return indices * step, envelope[indices], envelope


def _autocorrelation(envelope: np.ndarray, step: float) -> tuple[np.ndarray, float]:
    """Average normalized local periodicity, avoiding one loud section's vote."""
    max_lag = min(len(envelope) - 1, int(round(4.5 / step)))
    accumulated = np.zeros(max_lag + 1)
    votes = 0
    window = max(1, int(round(12 / step)))
    stride = max(1, int(round(6 / step)))
    for start in range(0, max(1, len(envelope) - window // 2), stride):
        segment = envelope[start:start + window]
        if len(segment) < 200 or np.max(segment) < .15:
            continue
        segment = segment - np.mean(segment)
        nfft = 1 << (2 * len(segment) - 1).bit_length()
        transformed = np.fft.rfft(segment, nfft)
        correlation = np.fft.irfft(transformed * np.conj(transformed), nfft)[:max_lag + 1]
        norm = np.dot(segment, segment)
        if norm > 1e-10:
            correction = len(segment) / np.maximum(len(segment) - np.arange(len(correlation)), len(segment) * .4)
            accumulated[:len(correlation)] += correlation / norm * correction
            votes += 1
    return accumulated / max(1, votes), step


def _sample_strength(times: np.ndarray, onset_times: np.ndarray,
                     strengths: np.ndarray, tolerance: float) -> np.ndarray:
    if not len(onset_times):
        return np.zeros(len(times))
    right = np.searchsorted(onset_times, times)
    before = np.clip(right - 1, 0, len(onset_times) - 1)
    after = np.clip(right, 0, len(onset_times) - 1)
    left_value = strengths[before] * np.exp(-.5 * ((times - onset_times[before]) / tolerance) ** 2)
    right_value = strengths[after] * np.exp(-.5 * ((times - onset_times[after]) / tolerance) ** 2)
    return np.maximum(left_value, right_value)


def _phase(period: float, onset_times: np.ndarray, strengths: np.ndarray) -> float:
    bins = 128
    phase = onset_times % period / period * bins
    first = np.floor(phase).astype(int)
    histogram = np.bincount(first, weights=strengths * (1 - (phase - first)), minlength=bins)
    histogram += np.bincount((first + 1) % bins, weights=strengths * (phase - first), minlength=bins)
    sigma = max(1, .022 / period * bins)
    offsets = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    smooth = sum(np.roll(histogram, int(offset)) * np.exp(-.5 * (offset / sigma) ** 2) for offset in offsets)
    return int(np.argmax(smooth)) * period / bins


def _tempo_score(period: float, onset_times: np.ndarray, strengths: np.ndarray,
                 duration: float, correlation: np.ndarray, step: float) -> dict:
    phase = _phase(period, onset_times, strengths)
    first = phase + np.ceil((onset_times[0] - phase - .06) / period) * period
    beats = np.arange(first, min(duration, onset_times[-1] + .08), period)
    if len(beats) < 4:
        return {"score": 0., "bpm": 60 / period, "phase": phase, "coverage": 0., "periodicity": 0.}
    tolerance = min(.05, period * .09)
    beat_values = _sample_strength(beats, onset_times, strengths, tolerance)
    half_values = _sample_strength(beats + period / 2, onset_times, strengths, tolerance)
    third_values = (_sample_strength(beats + period / 3, onset_times, strengths, tolerance) +
                    _sample_strength(beats + 2 * period / 3, onset_times, strengths, tolerance)) / 2
    typical = max(float(np.percentile(strengths, 70)), 1e-8)
    salience = float(np.mean(np.minimum(beat_values / typical, 2)))
    coverage = float(np.mean(beat_values > typical * .27))
    contrast = (float(np.mean(beat_values)) - max(float(np.mean(half_values)), float(np.mean(third_values)))) / typical
    lags = np.arange(len(correlation)) * step
    acf = [max(0., float(np.interp(period * multiple, lags, correlation))) for multiple in (1, 2, 3, 4)]
    periodicity = .55 * acf[0] + .25 * acf[1] + .12 * acf[2] + .08 * acf[3]
    # Coverage discourages subdividing every beat into silence; contrast favors
    # the accent-bearing pulse over an equally regular train of eighth notes.
    distance = np.abs((onset_times - phase + period / 2) % period - period / 2)
    important = np.minimum(strengths / typical, 2) ** 4
    explained = float(np.sum(important * np.exp(-.5 * (distance / tolerance) ** 2)) / max(np.sum(important), 1e-8))
    score = (.30 * salience + .20 * coverage + .24 * periodicity +
             .12 * max(0., contrast) + .22 * explained)
    return {"bpm": 60 / period, "score": score, "phase": float(first),
            "coverage": coverage, "periodicity": periodicity, "explained_attacks": explained}


def _tempo(onset_times: np.ndarray, strengths: np.ndarray, duration: float,
           correlation: np.ndarray, step: float) -> list[dict]:
    # Candidate lags, their octaves, and a coarse sweep survive syncopation and
    # broad autocorrelation peaks without assuming a fixed 120 BPM default.
    indices = np.arange(1, len(correlation) - 1)
    maxima = indices[(correlation[indices] >= correlation[indices - 1]) &
                     (correlation[indices] > correlation[indices + 1])]
    maxima = maxima[(maxima * step >= .25) & (maxima * step <= 1.5)]
    maxima = sorted(maxima, key=lambda index: correlation[index], reverse=True)[:18]
    bpms = list(np.arange(40, 241, 5, dtype=float))
    for lag in maxima:
        value = 60 / (lag * step)
        bpms.extend(value * multiplier for multiplier in (.5, 1., 2.) if 40 <= value * multiplier <= 240)
    ranked = sorted((_tempo_score(60 / bpm, onset_times, strengths, duration, correlation, step)
                     for bpm in bpms), key=lambda item: item["score"], reverse=True)
    seeds = []
    for candidate in ranked:
        if all(abs(np.log2(candidate["bpm"] / seed["bpm"])) > .055 for seed in seeds):
            seeds.append(candidate)
            if len(seeds) == 8:
                break
    refined = []
    for seed in seeds:
        # A fine enough sweep prevents beat drift over multi-minute files.
        spacing = min(.15, 15 / max(duration, 10))
        values = np.arange(max(40, seed["bpm"] - 3), min(240, seed["bpm"] + 3) + spacing / 2, spacing)
        best = max((_tempo_score(60 / bpm, onset_times, strengths, duration, correlation, step)
                    for bpm in values), key=lambda item: item["score"])
        refined.append(best)
    return sorted(refined, key=lambda item: item["score"], reverse=True)


def _track_beats(period: float, first: float, duration: float,
                 onset_times: np.ndarray, strengths: np.ndarray) -> np.ndarray:
    # Robustly fit phase/period to attacks near the winning grid before tracking.
    indices = np.rint((onset_times - first) / period)
    residual = onset_times - (first + indices * period)
    chosen = np.abs(residual) < min(.08, period * .13)
    if np.count_nonzero(chosen) >= 6:
        x = indices[chosen]
        y = onset_times[chosen]
        weight = np.minimum(strengths[chosen], np.percentile(strengths, 90))
        for _ in range(3):
            matrix = np.column_stack((x, np.ones(len(x)))) * np.sqrt(weight[:, None])
            fit = np.linalg.lstsq(matrix, y * np.sqrt(weight), rcond=None)[0]
            error = y - (fit[0] * x + fit[1])
            weight *= np.minimum(1., .025 / np.maximum(np.abs(error), 1e-6))
        if abs(fit[0] / period - 1) < .025:
            period, first = float(fit[0]), float(fit[1])
    if -.06 <= first < 0 and onset_times[0] < .06:
        first = 0.
    while first < 0:
        first += period
    while first >= period and first - period >= onset_times[0] - .08:
        first -= period
    grid = np.arange(first, duration, period)
    beats = []
    local_period = period
    for predicted in grid:
        # Accumulate gentle phase correction while keeping missing beats.
        expected = predicted if not beats else beats[-1] + local_period
        left = np.searchsorted(onset_times, expected - period * .18)
        right = np.searchsorted(onset_times, expected + period * .18)
        near = np.arange(left, right)
        if len(near):
            values = strengths[near] * np.exp(-.5 * ((onset_times[near] - expected) / (.065 * period)) ** 2)
            attack = float(onset_times[near[int(np.argmax(values))]])
            beat = .75 * expected + .25 * attack
            if beats:
                local_period = float(np.clip(.96 * local_period + .04 * (beat - beats[-1]), .9 * period, 1.1 * period))
        else:
            beat = expected
        if 0 <= beat < duration:
            beats.append(beat)
    return np.asarray(beats)


def _resolve_octave(candidates: list[dict], feature: np.ndarray, step: float) -> list[dict]:
    """Alternating bass and treble attacks provide evidence for a faster pulse.

    Loud backbeats otherwise make a snare-only half-tempo grid score better
    than the actual kick/snare pulse, particularly under background noise.
    """
    winner = candidates[0]
    faster = next((item for item in candidates[1:]
                   if abs(item["bpm"] / winner["bpm"] - 2) < .035), None)
    if (faster is None or winner["score"] - faster["score"] > .25 or
            faster["coverage"] < .7 or faster["periodicity"] < .35):
        return candidates
    period = 60 / winner["bpm"]
    phases = []
    for column in (0, 2):
        band = np.repeat(feature[:, column:column + 1], 3, axis=1)
        times, weights, envelope = _peaks(band, step)
        if len(times) < 6:
            return candidates
        correlation, _ = _autocorrelation(envelope, step)
        if float(np.interp(period, np.arange(len(correlation)) * step, correlation)) < .3:
            return candidates
        phases.append(_phase(period, times, weights ** 3))
    separation = abs((phases[0] - phases[1] + period / 2) % period - period / 2) / period
    if separation > .37:
        faster["score"] = winner["score"] + .025
        faster["octave_evidence"] = "Alternating bass and treble attacks support this pulse."
        return sorted(candidates, key=lambda item: item["score"], reverse=True)
    return candidates


def _local_tempos(feature: np.ndarray, step: float, reference: float) -> list[dict]:
    """Recover a changing pulse from short overlapping, independently scored windows."""
    width = int(round(6 / step))
    stride = width // 2
    windows = []
    examined = 0
    for start in range(0, len(feature) - width + 1, stride):
        examined += 1
        section = feature[start:start + width]
        times, strengths, envelope = _peaks(section, step)
        if len(times) < 7:
            continue
        correlation, _ = _autocorrelation(envelope, step)
        ranked = _resolve_octave(_tempo(times, strengths, width * step, correlation, step), section, step)
        nearby = [item for item in ranked if abs(np.log2(item["bpm"] / reference)) < .3
                  and item["score"] >= ranked[0]["score"] - .17]
        if not nearby:
            continue
        best = nearby[0]
        if best["periodicity"] < .23 or best["coverage"] < .58:
            continue
        windows.append({"time": (start + width / 2) * step, "bpm": float(best["bpm"]),
                        "phase": start * step + best["phase"],
                        "confidence": .65 * best["periodicity"] + .2 * best["coverage"]})
    return windows if len(windows) >= 3 and len(windows) >= .6 * examined else []


def _resolve_bar_pulse(candidates: list[dict], feature: np.ndarray, step: float,
                       onset_times: np.ndarray, strengths: np.ndarray, duration: float) -> list[dict]:
    """A repeated waltz downbeat can win autocorrelation as a whole-bar pulse.

    Promote its three-times-faster alternative only if that grid independently
    recovers a consistent three-beat accent pattern. No preferred BPM is used.
    """
    winner = candidates[0]
    faster = next((item for item in candidates[1:]
                   if abs(item["bpm"] / winner["bpm"] - 3) < .04), None)
    if faster is None or faster["coverage"] < .6 or faster["periodicity"] < .4:
        return candidates
    low_times, low_strengths, _ = _peaks(np.repeat(feature[:, :1], 3, axis=1), step)
    high_times, high_strengths, _ = _peaks(np.repeat(feature[:, 2:3], 3, axis=1), step)
    for candidate in (winner, faster):
        beats = _track_beats(60 / candidate["bpm"], candidate["phase"], duration, onset_times, strengths)
        signature, confidence, _, _, diagnostics = _meter(
            beats, onset_times, strengths, low_times, low_strengths, high_times, high_strengths)
        if candidate is winner:
            if signature is not None:
                return candidates
        elif signature == (3, 4) and confidence > .6:
            faster["score"] = winner["score"] + .025
            faster["metrical_evidence"] = "A repeating three-beat accent pattern supports beats rather than whole bars."
            return sorted(candidates, key=lambda item: item["score"], reverse=True)
    return candidates


def _track_variable(windows: list[dict], duration: float, onset_times: np.ndarray,
                    strengths: np.ndarray) -> np.ndarray:
    centers = np.array([item["time"] for item in windows])
    periods = 60 / np.array([item["bpm"] for item in windows])
    current = windows[0]["phase"]
    # Walk backwards to the first audible region using the earliest local pulse.
    while current - periods[0] >= onset_times[0] - .06:
        current -= periods[0]
    beats = []
    while current < duration:
        period = float(np.interp(current, centers, periods))
        left = np.searchsorted(onset_times, current - .18 * period)
        right = np.searchsorted(onset_times, current + .18 * period)
        near = np.arange(left, right)
        if len(near):
            scores = strengths[near] * np.exp(-.5 * ((onset_times[near] - current) / (.07 * period)) ** 2)
            attack = onset_times[near[int(np.argmax(scores))]]
            current = .4 * current + .6 * float(attack)
        if current >= 0:
            beats.append(current)
        current += period
    return np.asarray(beats)


def _meter(beats: np.ndarray, onset_times: np.ndarray, strengths: np.ndarray,
           low_times: np.ndarray, low_strengths: np.ndarray,
           high_times: Optional[np.ndarray] = None, high_strengths: Optional[np.ndarray] = None) -> tuple[Optional[tuple[int, int]], float, int, float, list[dict]]:
    if len(beats) < 9:
        return None, 0., 0, 0., []
    period = float(np.median(np.diff(beats)))
    tolerance = min(.045, period * .08)
    broad = _sample_strength(beats, onset_times, strengths, tolerance)
    bass = _sample_strength(beats, low_times, low_strengths, tolerance)
    if np.max(bass) > 0:
        bass *= np.max(broad) / np.max(bass)
    # Spectral flux used square-root compression to retain quiet attacks. Undo
    # it here so a modest but repeatable bar accent can carry metric evidence.
    accents = .25 * broad ** 2 + .75 * bass ** 2
    # A loud snare can contain as much low-band flux as a kick. Consider its
    # relative bass prominence as independent timbral accent evidence, while
    # retaining absolute accents for instruments whose timbre stays constant.
    relative_bass = bass / np.maximum(broad, .3 * max(float(np.mean(broad)), 1e-8))
    relative_bass *= np.max(broad) / max(float(np.max(relative_bass)), 1e-8)
    accent_profiles = (accents, .25 * broad ** 2 + .75 * relative_bass ** 2)
    half = _sample_strength(beats[:-1] + period / 2, onset_times, strengths, tolerance)
    third = (_sample_strength(beats[:-1] + period / 3, onset_times, strengths, tolerance) +
             _sample_strength(beats[:-1] + 2 * period / 3, onset_times, strengths, tolerance)) / 2
    pulse = max(float(np.mean(broad)), 1e-8)
    triple_evidence = float(np.mean(third) / pulse - np.mean(half) / pulse)
    if high_times is not None and high_strengths is not None:
        treble = _sample_strength(beats, high_times, high_strengths, tolerance)
        treble_half = _sample_strength(beats[:-1] + period / 2, high_times, high_strengths, tolerance)
        treble_third = (_sample_strength(beats[:-1] + period / 3, high_times, high_strengths, tolerance) +
                        _sample_strength(beats[:-1] + 2 * period / 3, high_times, high_strengths, tolerance)) / 2
        # Quiet cymbal subdivisions can disappear in the combined envelope,
        # even though their repeated triplet positions remain audible.
        treble_evidence = float(np.mean(treble_third - treble_half) / max(np.mean(treble), .3 * pulse))
        triple_evidence = max(triple_evidence, treble_evidence)
    candidates = []
    for count, signature in ((3, (3, 4)), (4, (4, 4)), (2, (6, 8))):
        if signature == (6, 8) and triple_evidence < .18:
            continue
        for offset in range(count):
            available = (len(accents) - offset) // count
            if available < 3:
                continue
            for profile in accent_profiles:
                bars = profile[offset:offset + available * count].reshape(available, count)
                scale = np.maximum(np.mean(bars, axis=1), 1e-8)
                normalized = bars / scale[:, None]
                other = np.mean(normalized[:, 1:], axis=1)
                contrast = normalized[:, 0] - other
                if count == 4:
                    # Alternating kick/snare accents repeat every two pulses.
                    # Grouping two identical pairs must not manufacture a
                    # stronger four-pulse bar than the compound alternative.
                    contrast = np.minimum(contrast, normalized[:, 0] - normalized[:, 2])
                positive = float(np.mean(contrast > .12))
                strength = float(np.median(contrast))
                stability = 1 / (1 + float(np.median(np.abs(contrast - np.median(contrast)))))
                score = max(0., strength) * positive * stability
                if signature == (6, 8):
                    score += min(.15, max(0., triple_evidence) * .22) * positive
                candidates.append((score, signature, offset, positive, strength))
    if not candidates:
        return None, 0., 0, triple_evidence, []
    candidates.sort(reverse=True)
    diagnostics = []
    for score, signature, offset, positive, strength in candidates:
        if any(item["time_signature"] == signature for item in diagnostics):
            continue
        diagnostics.append({"time_signature": signature, "score": float(score),
                            "confidence": float(np.clip(score * .65, 0, 1)),
                            "downbeat_index": int(offset), "downbeat_time": float(beats[offset]),
                            "accent_contrast": float(strength), "consistent_bars": float(positive),
                            "triplet_subdivision_evidence": float(triple_evidence)})
    best = candidates[0]
    alternative = next((item[0] for item in candidates if item[1] != best[1]), 0.)
    confidence = float(np.clip(best[0] * .65 + max(0, best[0] - alternative) * .5, 0, 1))
    # Similar 3/4 and 6/8 evidence, or equal accents, are genuinely ambiguous.
    if best[0] < .18 or best[3] < .7 or best[0] - alternative < .075:
        return None, min(confidence, .39), 0, triple_evidence, diagnostics
    return best[1], confidence, best[2], triple_evidence, diagnostics


def analyze_samples(samples: np.ndarray, sample_rate: int, bpm: Optional[float] = None,
                    time_signature: Optional[tuple[int, int]] = None,
                    beat_offset: Optional[float] = None) -> AudioRhythmResult:
    """Analyze decoded audio, with optional explicit notation/grid overrides.

    A manual BPM is quarter-note BPM. ``beat_offset`` is the first bar's start
    in seconds, and may precede the first audible attack (a pickup or rest).
    Tempo candidates are conducting-pulse BPM until compound meter is resolved;
    their ``pulse_bpm`` field remains explicit after conversion to quarter BPM.
    Automatic search covers 40--240 conducting pulses/minute; explicit BPM may
    cover 20--400 quarter notes/minute. Confidence is evidence, not probability.
    """
    if isinstance(sample_rate, bool) or not np.isfinite(sample_rate) or sample_rate < 1000:
        raise ValueError("sample_rate must be at least 1000 Hz")
    if bpm is not None and (not np.isfinite(bpm) or not 20 <= bpm <= 400):
        raise ValueError("bpm must be finite and between 20 and 400 quarter notes/minute")
    if beat_offset is not None and (not np.isfinite(beat_offset) or beat_offset < 0):
        raise ValueError("beat_offset must be a finite nonnegative time in seconds")
    if time_signature is not None:
        if (len(time_signature) != 2 or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in time_signature)
                or not 1 <= time_signature[0] <= 32 or time_signature[1] not in (1, 2, 4, 8, 16, 32)):
            raise ValueError("time_signature must contain a positive numerator and a power-of-two denominator")
        time_signature = tuple(int(value) for value in time_signature)
    audio = _mono(samples)
    duration = len(audio) / sample_rate
    result = AudioRhythmResult(duration=duration)
    compound = bool(time_signature and time_signature[1] == 8 and time_signature[0] >= 6 and time_signature[0] % 3 == 0)
    if len(audio) < sample_rate * 2 or (len(audio) and np.sqrt(np.mean(audio * audio)) < 1e-7):
        result.warnings.append("Audio is silent or too short for reliable automatic rhythm analysis.")
        return _overrides(result, bpm, time_signature, beat_offset, compound)
    audio, analysis_rate = _reduce_rate(audio, int(sample_rate))
    feature, step = _features(audio, analysis_rate)
    onset_times, strengths, envelope = _peaks(feature, step)
    if len(onset_times) < 5 or onset_times[-1] - onset_times[0] < 2:
        result.warnings.append("Too few separate attacks to estimate rhythm reliably.")
        return _overrides(result, bpm, time_signature, beat_offset, compound)
    correlation, _ = _autocorrelation(envelope, step)
    candidates = _resolve_octave(_tempo(onset_times, strengths, duration, correlation, step), feature, step)
    if time_signature is None or time_signature == (3, 4):
        candidates = _resolve_bar_pulse(candidates, feature, step, onset_times, strengths, duration)
    result.tempo_candidates = candidates[:6]
    winner = candidates[0]
    other = next((item for item in candidates[1:] if abs(np.log2(item["bpm"] / winner["bpm"])) > .12), None)
    separation = max(0., winner["score"] - (other["score"] if other else 0.))
    confidence = float(np.clip(winner["periodicity"] * .65 + winner["coverage"] * .2 + separation * .8, 0, 1))
    windows = []
    if bpm is None and duration >= 15 and (winner["periodicity"] < .32 or winner.get("explained_attacks", 0) < .4):
        windows = _local_tempos(feature, step, winner["bpm"])
        if windows:
            local_bpms = np.array([item["bpm"] for item in windows])
            variation = (np.percentile(local_bpms, 90) - np.percentile(local_bpms, 10)) / np.median(local_bpms)
            if variation > .055:
                winner = dict(winner, bpm=float(np.median(local_bpms)), coverage=.8,
                              periodicity=float(np.mean([item["confidence"] for item in windows])))
                confidence = float(np.clip(winner["periodicity"], 0, 1))
                result.warnings.append("Tempo changes across the recording; reported BPM is a median and beat times follow local estimates.")
            else:
                windows = []
    result.tempo_confidence = confidence
    if winner["periodicity"] < .14 or winner["coverage"] < .46 or confidence < .3:
        result.warnings.append("No stable periodic pulse was found; automatic BPM is unknown.")
        return _overrides(result, bpm, time_signature, beat_offset, compound)
    period = (90 if compound else 60) / bpm if bpm is not None else 60 / winner["bpm"]
    phase = _phase(period, onset_times, strengths)
    phase += np.ceil((onset_times[0] - phase - .06) / period) * period
    beats = (_track_variable(windows, duration, onset_times, strengths) if windows else
             _track_beats(period, phase, duration, onset_times, strengths))
    low_feature = np.zeros_like(feature)
    low_feature[:, :] = feature[:, :1]
    low_times, low_strengths, _ = _peaks(low_feature, step)
    high_times, high_strengths, _ = _peaks(np.repeat(feature[:, 2:3], 3, axis=1), step)
    inferred_meter, meter_confidence, downbeat_index, triple_evidence, meter_candidates = _meter(beats, onset_times, strengths, low_times, low_strengths, high_times, high_strengths)
    result.meter_candidates = meter_candidates
    signature = time_signature or inferred_meter
    compound = bool(signature and signature[1] == 8 and signature[0] >= 6 and signature[0] % 3 == 0)
    # An explicit meter resolves pulse notation. Quarter BPM is never silently
    # interpreted as the dotted-quarter tempo displayed by some music apps.
    multiplier = 1.5 if compound else 1.
    result.bpm = float(bpm if bpm is not None else winner["bpm"] * multiplier)
    result.beat_unit = "dotted-quarter" if compound else "quarter"
    result.time_signature = signature
    result.meter_confidence = 1. if time_signature is not None else meter_confidence
    if signature is None:
        result.warnings.append("Bar accents do not distinguish a reliable time signature; specify the meter manually.")
        if triple_evidence >= .18:
            result.warnings.append("Triplet subdivisions are audible, but the pulse may represent dotted quarters; quarter-note BPM could be 1.5 times the reported value.")
    if other and (abs(np.log2(other["bpm"] / winner["bpm"])) > .8) and separation < .06:
        result.warnings.append("Half/double-time interpretations have similar evidence; inspect tempo candidates.")
    if len(beats) > 2 and np.std(np.diff(beats)) / np.mean(np.diff(beats)) > .035:
        result.warnings.append("Beat spacing varies; a single global BPM is only an approximation.")
    result.beat_times = beats.tolist()
    result.grid_origin = float(beats[downbeat_index] if len(beats) else 0.)
    if signature:
        count = signature[0] // 3 if compound else signature[0] * 4 / signature[1]
        # Most conventional meters have integral quarter-note pulse counts.
        if count == int(count):
            result.downbeat_times = beats[downbeat_index::int(count)].tolist()
        else:
            result.downbeat_times = np.arange(result.grid_origin, duration, 60 / result.bpm * count).tolist()
    for candidate in result.tempo_candidates:
        candidate["pulse_bpm"] = candidate["bpm"]
        candidate["bpm"] *= multiplier
        candidate["bpm_unit"] = "quarter-note"
        if signature is None and triple_evidence >= .18:
            candidate["quarter_bpm_if_compound"] = candidate["pulse_bpm"] * 1.5
    if beat_offset is not None:
        return _overrides(result, result.bpm, signature, beat_offset, compound)
    return result


def _overrides(result: AudioRhythmResult, bpm: Optional[float], signature: Optional[tuple[int, int]],
               offset: Optional[float], compound: bool) -> AudioRhythmResult:
    result.time_signature = signature
    result.bpm = float(bpm) if bpm is not None else None
    result.beat_unit = "dotted-quarter" if compound else "quarter"
    if signature is not None:
        result.meter_confidence = 1.
    if bpm is not None:
        period = (90 if compound else 60) / bpm
        result.grid_origin = float(offset or 0.)
        result.beat_times = np.arange(result.grid_origin, result.duration, period).tolist()
        if signature:
            bar_length = signature[0] * 4 / signature[1] * 60 / bpm
            result.downbeat_times = np.arange(result.grid_origin, result.duration, bar_length).tolist()
    elif offset is not None:
        result.grid_origin = float(offset)
    return result


# Backward-compatible names for callers that describe the operation as rhythm.
analyze_rhythm = analyze_samples
RhythmAnalysis = AudioRhythmResult
