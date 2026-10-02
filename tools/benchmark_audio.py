"""Deterministic synthetic acceptance and performance harness for MP3 rhythm analysis.

This measures algorithm behavior on controlled signals; it is not a substitute for
evaluation on licensed real recordings. Requires only NumPy and the project code.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep this import at execution time so generation utilities remain usable while
# the implementation is being integrated. Adjust this one line if the public
# function is intentionally exported from a different module.
def _get_analyzer():
    try:
        from core.audio_rhythm import analyze_samples
        return analyze_samples
    except ImportError as exc:
        raise RuntimeError("Could not import core.audio_rhythm.analyze_samples; audio implementation is not available") from exc

SR = 22050


def _envelope(n: int, decay: float) -> np.ndarray:
    return np.exp(-np.arange(n, dtype=np.float32) / (SR * decay))


def _kick(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    phase = 2 * np.pi * (95 * t - 33 * t * t)
    return (np.sin(phase) * _envelope(n, .095)).astype(np.float32)


def _snare(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    noise = rng.standard_normal(n).astype(np.float32)
    # A short noisy body with a pitched component; no special audio packages.
    return (0.72 * noise + 0.28 * np.sin(2 * np.pi * 185 * t)) * _envelope(n, .075)


def _hat(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(n).astype(np.float32)
    # First difference is a cheap high-pass for a transient cymbal-like sound.
    x[1:] -= x[:-1]
    return x * _envelope(n, .022)


def _tone(n: int, midi: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    f = 440 * 2 ** ((midi - 69) / 12)
    # Harmonic-rich pluck with a soft decay and deterministic phase.
    wave = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 5))
    return (wave * _envelope(n, .22)).astype(np.float32)


def make_music(*, bpm: float = 120, meter: tuple[int, int] = (4, 4), bars: int = 8,
               intro: float = 0, melody: bool = True, syncopation: bool = False,
               strong_downbeats: bool = False,
               noise: float = 0, tempo_end: float | None = None,
               sustained_only: bool = False, noise_only: bool = False, stereo: bool = False,
               seed: int = 127) -> np.ndarray:
    """Render a compact drum-and-melody fixture as normalized float32 PCM."""
    rng = np.random.default_rng(seed)
    num, den = meter
    quarters_per_bar = num * 4 / den
    total_quarters = bars * quarters_per_bar
    # Integrate linearly changing quarter-note tempo if requested.
    if tempo_end is None:
        duration = total_quarters * 60 / bpm
        beat_times = intro + np.arange(total_quarters + 1) * 60 / bpm
    else:
        # Solve per-quarter durations on a linear BPM trajectory.
        q = np.arange(total_quarters + 1, dtype=np.float64)
        bpms = bpm + (tempo_end - bpm) * q / max(total_quarters, 1)
        beat_times = intro + np.concatenate(([0.0], np.cumsum(60 / bpms[:-1])))
        duration = beat_times[-1] - intro

    def quarter_time(qpos: float) -> float:
        whole = int(math.floor(qpos))
        frac = qpos - whole
        return float(beat_times[whole] + frac * (beat_times[whole + 1] - beat_times[whole]))

    n_total = int(math.ceil((intro + duration + .15) * SR))
    out = np.zeros((n_total, 2 if stereo else 1), dtype=np.float32)

    if sustained_only:
        # Stable chord with no beat-locked transients should be treated as low
        # evidence: a detector should abstain or report weak confidence.
        for midi in (48, 55, 60, 64):
            tone = _tone(n_total, midi)
            out[:, 0] += tone * .15
            if stereo:
                out[:, 1] += tone * .14
    elif not noise_only:
        eighth = (beat_times[1] - beat_times[0]) / 2 if len(beat_times) > 1 else 0
        # Drum accents follow meter: quarter-note pattern for simple meters;
        # in 6/8, place accents at the first and fourth eighth note.
        for bar in range(bars):
            bar_q = bar * quarters_per_bar
            if den == 8 and num == 6:
                positions = [(bar_q + 0, "kick"), (bar_q + 1.5, "snare")]
                sub_positions = [bar_q + x * .5 for x in range(6)]
            else:
                positions = [(bar_q, "kick")]
                if num >= 3:
                    positions.append((bar_q + 1, "snare"))
                if num >= 4:
                    positions.extend([(bar_q + 2, "kick"), (bar_q + 3, "snare")])
                sub_positions = [bar_q + x * .5 for x in range(int(quarters_per_bar * 2))]
            for qpos, kind in positions:
                idx = int(round(quarter_time(qpos) * SR))
                n = min(int(.26 * SR), n_total - idx)
                if n <= 0:
                    continue
                hit = _kick(n) if kind == "kick" else _snare(n, rng)
                strength = (1.45 if qpos == bar_q else .48) if (strong_downbeats and kind == "kick") else (.88 if kind == "kick" else .62)
                out[idx:idx+n, 0] += hit * strength
                if stereo:
                    # Mild stereo placement to test downmixing/channel handling.
                    out[idx:idx+n, 1] += hit * (strength * .8 if kind == "kick" else .8)
            for j, qpos in enumerate(sub_positions):
                if syncopation and j == 0 and bar % 2:
                    continue
                idx = int(round(quarter_time(qpos) * SR))
                n = min(int(.045 * SR), n_total - idx)
                if n > 0:
                    hit = _hat(n, rng) * .10
                    out[idx:idx+n, 0] += hit
                    if stereo:
                        out[idx:idx+n, 1] += hit * .85
        if melody:
            # A diatonic phrase on offbeats gives useful harmonic rhythm while
            # retaining percussion as the primary pulse evidence.
            scale = (60, 62, 64, 67, 69, 67, 64, 62)
            for i in range(bars * 8):
                qpos = i * .5 + .5
                if qpos >= total_quarters:
                    continue
                qidx = int(qpos)
                frac = qpos - qidx
                t = beat_times[qidx] + frac * (beat_times[qidx+1] - beat_times[qidx])
                idx = int(round(t * SR))
                n = min(int(.32 * SR), n_total - idx)
                if n > 0:
                    note = _tone(n, scale[i % len(scale)]) * .12
                    out[idx:idx+n, 0] += note
                    if stereo:
                        out[idx:idx+n, 1] += note * .92
    if noise:
        out += rng.standard_normal(out.shape).astype(np.float32) * noise
    peak = float(np.max(np.abs(out)))
    if peak:
        out *= min(1.0, .98 / peak)
    return out[:, 0] if not stereo else out


@dataclass
class Case:
    name: str
    expected_bpm: float | None
    meter: tuple[int, int] | None = None
    expected_grid: float | None = None
    expected_end_bpm: float | None = None
    min_tempo_confidence: float | None = .35
    min_meter_confidence: float | None = None
    max_meter_confidence: float | None = None
    expect_tempo_rising: bool = False
    kwargs: dict | None = None


CASES = [
    Case("plain_4_4_120", 120, (4, 4), expected_grid=0.0, kwargs={"bpm": 120}),
    Case("strong_downbeat_4_4", 120, (4, 4), expected_grid=0.0, min_meter_confidence=.15,
         kwargs={"bpm": 120, "strong_downbeats": True}),
    Case("fractional_4_4_97_3", 97.3, (4, 4), kwargs={"bpm": 97.3}),
    Case("fast_4_4_178_5", 178.5, (4, 4), kwargs={"bpm": 178.5}),
    Case("slow_4_4_62_5", 62.5, (4, 4), kwargs={"bpm": 62.5}),
    Case("triple_3_4", 114, (3, 4), kwargs={"bpm": 114, "meter": (3, 4)}),
    Case("compound_6_8", 120, (6, 8), kwargs={"bpm": 120, "meter": (6, 8)}),
    Case("syncopated_stereo_intro", 103, (4, 4), expected_grid=1.17,
         kwargs={"bpm": 103, "intro": 1.17, "syncopation": True, "stereo": True}),
    Case("half_double_disambiguation", 84, (4, 4),
         kwargs={"bpm": 84, "meter": (4, 4), "melody": True}),
    Case("tempo_drift_96_to_128", 112, (4, 4), expect_tempo_rising=True,
         kwargs={"bpm": 96, "tempo_end": 128, "bars": 12}),
    Case("noisy_but_periodic", 110, (4, 4), kwargs={"bpm": 110, "noise": .05}),
    Case("white_noise_abstention", None, None, min_tempo_confidence=None,
         max_meter_confidence=.5, kwargs={"noise_only": True, "noise": .22}),
    Case("sustained_chord_abstention", None, None, min_tempo_confidence=None,
         max_meter_confidence=.5, kwargs={"sustained_only": True}),
]

# A deterministic challenge set, first run after initial tuning. Once its cases
# were reported, they became available for final regression work; report them as
# a challenge/regression score rather than a blind holdout.
_SWEEP_VARIANTS = [
    *((bpm, (4, 4)) for bpm in (57.2, 72.5, 88, 103.25, 119, 137.5, 156, 178, 202, 219, 81.4, 166.7)),
    *((bpm, (3, 4)) for bpm in (65.5, 92, 111, 144.5, 190, 205.2)),
    *((bpm, (6, 8)) for bpm in (60, 82.5, 100, 125, 160, 200)),
]
SWEEP_CASES = []
for _i, (_bpm, _meter) in enumerate(_SWEEP_VARIANTS):
    _unique_meter = _i in (0, 2, 12, 14, 18, 20)
    SWEEP_CASES.append(Case(
        name=f"challenge_{_meter[0]}_{_meter[1]}_{_bpm:g}_seed{4000 + _i}",
        expected_bpm=float(_bpm), meter=_meter,
        min_tempo_confidence=.35,
        min_meter_confidence=.15 if _unique_meter else None,
        kwargs={
            "bpm": float(_bpm), "meter": _meter, "bars": 8,
            "seed": 4000 + _i,
            "melody": _i % 3 != 1,
            "stereo": _i % 4 == 2,
            "noise": (0.0, .005, .015, .03)[_i % 4],
            "strong_downbeats": _unique_meter,
        },
    ))

# Independent sample selected after the challenge cases were frozen. Keep its
# different tempos/seeds as a second reproducible check; do not call it real-data
# validation because it uses the same synthetic signal generator.
_SECONDARY_VARIANTS = [
    (59.5, (4, 4)), (111.7, (4, 4)), (200.5, (4, 4)), (133.33, (4, 4)),
    (70, (3, 4)), (122.5, (3, 4)), (178.2, (3, 4)), (210.5, (3, 4)),
    (64.2, (6, 8)), (90, (6, 8)), (133, (6, 8)), (181.4, (6, 8)),
]
SECONDARY_CASES = []
for _i, (_bpm, _meter) in enumerate(_SECONDARY_VARIANTS):
    _unique_meter = _i in (0, 2, 4, 6, 8, 10)
    SECONDARY_CASES.append(Case(
        name=f"secondary_{_meter[0]}_{_meter[1]}_{_bpm:g}_seed{6050 + _i}",
        expected_bpm=float(_bpm), meter=_meter, min_tempo_confidence=.35,
        min_meter_confidence=.15 if _unique_meter else None,
        kwargs={
            "bpm": float(_bpm), "meter": _meter, "bars": 8,
            "seed": 6050 + _i,
            "melody": _i % 3 != 1,
            "stereo": _i % 4 == 2,
            "noise": (0.0, .005, .015, .03)[_i % 4],
            "strong_downbeats": _unique_meter,
        },
    ))


def _field(result, name, default=None):
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def run_cases(cases: list[Case] | None = None) -> list[dict]:
    rows = []
    for case in cases or CASES:
        samples = make_music(**(case.kwargs or {}))
        start = time.perf_counter()
        result = _get_analyzer()(samples, SR)
        elapsed = time.perf_counter() - start
        bpm = _field(result, "bpm")
        meter = _field(result, "time_signature")
        tc = _field(result, "tempo_confidence")
        mc = _field(result, "meter_confidence")
        warnings = _field(result, "warnings", []) or []
        beat_times = np.asarray(_field(result, "beat_times", []) or [], dtype=float)
        downbeats = np.asarray(_field(result, "downbeat_times", []) or [], dtype=float)
        candidates = _field(result, "tempo_candidates", []) or []
        failures = []
        if case.expected_bpm is None:
            if bpm is not None and tc is not None and tc > .5:
                failures.append("assigned a confident tempo to an unmetered fixture")
        elif bpm is None or abs(float(bpm) - case.expected_bpm) / case.expected_bpm > .06:
            failures.append(f"tempo outside 6% (expected {case.expected_bpm}, got {bpm})")
        if case.meter and meter is not None:
            actual = tuple(meter)
            if actual != case.meter and (case.min_meter_confidence is not None or (mc is not None and mc >= .45)):
                failures.append(f"confident wrong meter (expected {case.meter}, got {actual})")
        elif case.meter and case.min_meter_confidence is not None:
            failures.append(f"meter unresolved (expected {case.meter})")
        grid = _field(result, "grid_origin")
        phase_error = None
        if case.expected_grid is not None and grid is not None and bpm:
            beat_unit = _field(result, "beat_unit", "quarter")
            period = (90.0 / float(bpm)) if beat_unit == "dotted-quarter" else (60.0 / float(bpm))
            phase_error = abs(((float(grid) - case.expected_grid + period / 2) % period) - period / 2)
            if phase_error > .12:
                failures.append(f"beat-grid phase error {phase_error:.3f}s modulo pulse period")
        if case.expect_tempo_rising and beat_times.size >= 8:
            gaps = np.diff(beat_times)
            edge = max(2, len(gaps) // 4)
            early_bpm = 60 / float(np.median(gaps[:edge]))
            late_bpm = 60 / float(np.median(gaps[-edge:]))
            if late_bpm <= early_bpm * 1.08:
                failures.append(f"tempo drift not reflected in beat grid ({early_bpm:.1f} -> {late_bpm:.1f} BPM)")
        if case.min_tempo_confidence is not None and (tc is None or tc < case.min_tempo_confidence):
            failures.append(f"tempo confidence below {case.min_tempo_confidence}")
        if case.min_meter_confidence is not None and (mc is None or mc < case.min_meter_confidence):
            failures.append(f"meter confidence below {case.min_meter_confidence}")
        if case.max_meter_confidence is not None and mc is not None and mc > case.max_meter_confidence:
            failures.append(f"meter confidence too high for ambiguity ({mc})")
        rows.append({
            "case": case.name, "duration_s": len(samples) / SR if samples.ndim == 1 else len(samples) / SR,
            "elapsed_s": round(elapsed, 4), "rtf": round(elapsed / (len(samples) / SR), 5),
            "bpm": bpm, "meter": meter, "tempo_confidence": tc,
            "meter_confidence": mc, "grid_origin": grid,
            "beat_offset": _field(result, "beat_offset"),
            "downbeat_offset": _field(result, "downbeat_offset"),
            "beat_grid_phase_error_s": phase_error,
            "beats": int(beat_times.size),
            "downbeats": int(downbeats.size), "tempo_candidates": candidates,
            "warnings": warnings,
            "failures": failures,
        })
    return rows


def run_long_probe(minutes: float = 3) -> dict:
    samples = make_music(bpm=123, bars=max(1, math.ceil(minutes * 60 * 123 / 240)), melody=True)
    start = time.perf_counter()
    result = _get_analyzer()(samples, SR)
    elapsed = time.perf_counter() - start
    return {"duration_s": len(samples) / SR, "elapsed_s": round(elapsed, 3),
            "rtf": round(elapsed / (len(samples) / SR), 5),
            "bpm": _field(result, "bpm"), "tempo_confidence": _field(result, "tempo_confidence"),
            "beats": len(_field(result, "beat_times", []) or [])}


def run_challenge_sweep() -> dict:
    """Summarize a fixed 24-case synthetic challenge set."""
    rows = run_cases(SWEEP_CASES)
    within = []
    octave = []
    unknown = []
    for row, case in zip(rows, SWEEP_CASES):
        bpm = row["bpm"]
        if bpm is None:
            unknown.append(case.name)
            continue
        ratio = float(bpm) / case.expected_bpm
        if abs(ratio - 1) <= .06:
            within.append(case.name)
        elif abs(ratio - .5) <= .06 or abs(ratio - 2) <= .06:
            octave.append(case.name)
    meter_cases = [(row, case) for row, case in zip(rows, SWEEP_CASES)
                   if case.min_meter_confidence is not None]
    meter_hits = sum(tuple(row["meter"]) == case.meter for row, case in meter_cases if row["meter"] is not None)
    meter_wrong = sum(tuple(row["meter"]) != case.meter for row, case in meter_cases if row["meter"] is not None)
    meter_unknown = sum(row["meter"] is None for row, _case in meter_cases)
    return {
        "scope": "synthetic challenge set, now used for regression; not real-recording accuracy",
        "cases": len(rows),
        "bpm_within_6_percent": len(within),
        "bpm_octave_mistakes": len(octave),
        "bpm_unknown": len(unknown),
        "bpm_other_errors": len(rows) - len(within) - len(octave) - len(unknown),
        "bpm_within_6_percent_names": within,
        "bpm_octave_mistake_names": octave,
        "bpm_unknown_names": unknown,
        "clear_meter_cases": len(meter_cases),
        "clear_meter_hits": meter_hits,
        "clear_meter_wrong": meter_wrong,
        "clear_meter_unknown": meter_unknown,
        "case_results": rows,
    }


def run_secondary_sweep() -> dict:
    """Run a separately seeded 12-case synthetic sample."""
    rows = run_cases(SECONDARY_CASES)
    within = octave = unknown = 0
    for row, case in zip(rows, SECONDARY_CASES):
        bpm = row["bpm"]
        if bpm is None:
            unknown += 1
            continue
        ratio = float(bpm) / case.expected_bpm
        if abs(ratio - 1) <= .06:
            within += 1
        elif abs(ratio - .5) <= .06 or abs(ratio - 2) <= .06:
            octave += 1
    meter_cases = [(row, case) for row, case in zip(rows, SECONDARY_CASES)
                   if case.min_meter_confidence is not None]
    return {
        "scope": "separately seeded synthetic sample; not real-recording accuracy",
        "cases": len(rows),
        "bpm_within_6_percent": within,
        "bpm_octave_mistakes": octave,
        "bpm_unknown": unknown,
        "bpm_other_errors": len(rows) - within - octave - unknown,
        "clear_meter_cases": len(meter_cases),
        "clear_meter_hits": sum(row["meter"] is not None and tuple(row["meter"]) == case.meter for row, case in meter_cases),
        "clear_meter_wrong": sum(row["meter"] is not None and tuple(row["meter"]) != case.meter for row, case in meter_cases),
        "clear_meter_unknown": sum(row["meter"] is None for row, _case in meter_cases),
        "case_results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--performance", action="store_true", help="also analyze a three-minute fixture")
    parser.add_argument("--sweep", action="store_true", help="run 24 deterministic synthetic challenge fixtures")
    parser.add_argument("--secondary-sweep", action="store_true", help="run 12 separately seeded synthetic fixtures")
    parser.add_argument("--minutes", type=float, default=3.0, help="long-fixture duration for --performance")
    args = parser.parse_args()
    if args.performance and (not math.isfinite(args.minutes) or args.minutes <= 0):
        parser.error("--minutes must be a finite positive number")
    rows = run_cases()
    report = {"scope": "synthetic controlled signals only; not real-recording accuracy",
              "sample_rate": SR, "cases": rows,
              "meter_summary": {"resolved": sum(row["meter"] is not None for row in rows),
                                "unknown": sum(row["meter"] is None for row in rows)}}
    if args.performance:
        report["long_probe"] = run_long_probe(args.minutes)
    if args.sweep:
        report["challenge_sweep"] = run_challenge_sweep()
    if args.secondary_sweep:
        report["secondary_sweep"] = run_secondary_sweep()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(report["scope"])
        print(f"meter results: {report['meter_summary']['resolved']} resolved, {report['meter_summary']['unknown']} unknown")
        for row in rows:
            print(f"{row['case']:30} bpm={row['bpm']} meter={row['meter']} conf={row['tempo_confidence']} {row['elapsed_s']}s "
                  f"{'FAIL: ' + '; '.join(row['failures']) if row['failures'] else 'PASS'}")
        if args.performance:
            print("long probe:", report["long_probe"])
        if args.sweep:
            sweep = report["challenge_sweep"]
            print(f"challenge: BPM within6={sweep['bpm_within_6_percent']}/{sweep['cases']}, "
                  f"octave={sweep['bpm_octave_mistakes']}, unknown={sweep['bpm_unknown']}, "
                  f"other={sweep['bpm_other_errors']}; clear meter hit/wrong/unknown="
                  f"{sweep['clear_meter_hits']}/{sweep['clear_meter_wrong']}/{sweep['clear_meter_unknown']} "
                  f"of {sweep['clear_meter_cases']}")
        if args.secondary_sweep:
            sweep = report["secondary_sweep"]
            print(f"secondary: BPM within6={sweep['bpm_within_6_percent']}/{sweep['cases']}, "
                  f"octave={sweep['bpm_octave_mistakes']}, unknown={sweep['bpm_unknown']}, "
                  f"other={sweep['bpm_other_errors']}; clear meter hit/wrong/unknown="
                  f"{sweep['clear_meter_hits']}/{sweep['clear_meter_wrong']}/{sweep['clear_meter_unknown']} "
                  f"of {sweep['clear_meter_cases']}")
    failed = any(r["failures"] for r in rows)
    for key in ("challenge_sweep", "secondary_sweep"):
        if key not in report:
            continue
        sweep = report[key]
        failed = failed or any(sweep[key] > 0 for key in (
            "bpm_octave_mistakes", "bpm_unknown", "bpm_other_errors", "clear_meter_wrong"
        ))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
