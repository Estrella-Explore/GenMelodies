"""Known-timing signals exercise DSP without depending on music downloads."""
import unittest

import numpy as np

from core.audio_rhythm import analyze_samples


def rhythm_audio(bpm=120, meter=4, duration=18., offset=.31, subdivisions=2,
                 compound=False, seed=13, missing=0., jitter=0., stereo=False):
    """Overlapping bass, snare/noise and ringing notes, not impulse clicks."""
    sample_rate = 11025
    random = np.random.default_rng(seed)
    audio = np.zeros(int(duration * sample_rate))
    pulse = (90 if compound else 60) / bpm
    steps = 3 if compound else subdivisions
    count = meter // 3 if compound else meter
    events = np.arange(offset, duration - .1, pulse / steps)
    for index, when in enumerate(events):
        beat_index, subdivision = divmod(index, steps)
        downbeat = beat_index % count == 0 and subdivision == 0
        if not downbeat and random.random() < missing:
            continue
        when += random.normal(0, jitter)
        start = int(when * sample_rate)
        length = min(int(.18 * sample_rate), len(audio) - start)
        t = np.arange(length) / sample_rate
        amplitude = 1. if downbeat else (.54 if subdivision == 0 else .22)
        sound = (.55 * np.sin(2 * np.pi * (80 if downbeat else 170) * t) * np.exp(-t * 32)
                 + .18 * random.normal(size=length) * np.exp(-t * 65)
                 + .2 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 14))
        audio[start:start + length] += amplitude * sound
    if stereo:
        audio = np.column_stack((audio, -audio))
    return audio, sample_rate


class AudioRhythmTests(unittest.TestCase):
    def test_straight_meter_and_nonzero_grid(self):
        audio, rate = rhythm_audio(bpm=123, meter=4)
        result = analyze_samples(audio, rate)
        self.assertAlmostEqual(result.bpm, 123, delta=1.)
        self.assertEqual(result.time_signature, (4, 4))
        self.assertAlmostEqual(result.grid_origin, .31, delta=.06)
        self.assertGreater(result.tempo_confidence, .4)

    def test_waltz_with_stereo_phase_cancellation(self):
        audio, rate = rhythm_audio(bpm=93, meter=3, stereo=True)
        result = analyze_samples(audio, rate)
        self.assertAlmostEqual(result.bpm, 93, delta=1.)
        self.assertEqual(result.time_signature, (3, 4))

    def test_compound_meter_has_quarter_note_bpm(self):
        audio, rate = rhythm_audio(bpm=156, meter=6, compound=True)
        result = analyze_samples(audio, rate)
        self.assertEqual(result.time_signature, (6, 8))
        self.assertAlmostEqual(result.bpm, 156, delta=1.5)
        self.assertEqual(result.beat_unit, "dotted-quarter")
        self.assertAlmostEqual(np.median(np.diff(result.beat_times)), 90 / 156, delta=.02)

    def test_jitter_missing_attacks(self):
        audio, rate = rhythm_audio(bpm=137, meter=4, duration=26, missing=.2, jitter=.006)
        result = analyze_samples(audio, rate)
        self.assertAlmostEqual(result.bpm, 137, delta=1.5)
        self.assertEqual(result.time_signature, (4, 4))

    def test_unaccented_pulse_does_not_invent_meter(self):
        rate = 11025
        audio = np.zeros(rate * 14)
        click = np.random.default_rng(8).normal(size=200) * np.exp(-np.arange(200) / 35)
        for when in np.arange(.2, 13.9, .5):
            index = int(when * rate)
            audio[index:index + len(click)] += click
        result = analyze_samples(audio, rate)
        self.assertAlmostEqual(result.bpm, 120, delta=1.)
        self.assertIsNone(result.time_signature)

    def test_nonrhythmic_inputs_abstain(self):
        rate = 11025
        random = np.random.default_rng(44)
        for audio in (np.zeros(rate * 10), np.sin(2 * np.pi * 440 * np.arange(rate * 10) / rate),
                      random.normal(0, .1, size=rate * 12), np.zeros(rate // 2)):
            with self.subTest(kind=float(np.std(audio))):
                result = analyze_samples(audio, rate)
                self.assertIsNone(result.bpm)
                self.assertIsNone(result.time_signature)

    def test_manual_compound_grid_on_silence(self):
        result = analyze_samples(np.zeros(11025 * 10), 11025, bpm=180,
                                 time_signature=(6, 8), beat_offset=.25)
        self.assertEqual(result.bpm, 180)
        self.assertEqual(result.time_signature, (6, 8))
        self.assertAlmostEqual(result.beat_times[1] - result.beat_times[0], .5)
        self.assertEqual(result.downbeat_times[:2], [.25, 1.25])

    def test_nonfinite_rejected_and_result_serializable(self):
        with self.assertRaises(ValueError):
            analyze_samples(np.array([0., np.nan]), 11025)
        result = analyze_samples(np.zeros(11025), 11025)
        self.assertIn("tempo_candidates", result.to_dict())


if __name__ == "__main__":
    unittest.main()
