"""Known-timing signals exercise DSP without depending on music downloads."""
import unittest
import json

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


def sparse_drums(bpm, compound=False, seed=77, noise=.015):
    """A quiet third waltz beat, or two distinct compound conducting pulses."""
    rate = 11025
    random = np.random.default_rng(seed)
    quarter = 60 / bpm
    bar = 3 * quarter
    audio = np.zeros(int((8 * bar + .2) * rate))
    for measure in range(8):
        for position in range(6):
            start = int((measure * bar + position * quarter / 2) * rate)
            t = np.arange(int(.18 * rate)) / rate
            hat = random.normal(size=len(t))
            hat[1:] -= hat[:-1]
            sound = .08 * hat * np.exp(-t * 80)
            if position == 0:
                sound += .8 * np.sin(2 * np.pi * 90 * t) * np.exp(-t * 18)
            elif position == (3 if compound else 2):
                sound += .45 * random.normal(size=len(t)) * np.exp(-t * 25)
            audio[start:start + len(sound)] += sound
    audio += random.normal(0, noise, size=len(audio))
    return audio, rate


class AudioRhythmTests(unittest.TestCase):
    def test_straight_meter_and_nonzero_grid(self):
        audio, rate = rhythm_audio(bpm=123, meter=4)
        result = analyze_samples(audio, rate)
        self.assertAlmostEqual(result.bpm, 123, delta=1.)
        self.assertEqual(result.time_signature, (4, 4))
        self.assertAlmostEqual(result.grid_origin, .31, delta=.06)
        self.assertGreater(result.tempo_confidence, .4)
        self.assertEqual(result.meter_candidates[0]["time_signature"], (4, 4))
        json.dumps(result.to_dict())

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

    def test_fast_waltz_does_not_mistake_whole_bars_for_beats(self):
        for bpm in (147., 187.5, 207.):
            with self.subTest(bpm=bpm):
                result = analyze_samples(*sparse_drums(bpm, noise=.01))
                self.assertAlmostEqual(result.bpm, bpm, delta=2.)
                self.assertEqual(result.time_signature, (3, 4))

    def test_compound_tempo_range_with_quiet_triplet_cymbals(self):
        for bpm in (61., 85., 129., 162.):
            with self.subTest(bpm=bpm):
                result = analyze_samples(*sparse_drums(bpm, compound=True))
                self.assertAlmostEqual(result.bpm, bpm, delta=2.)
                self.assertEqual(result.time_signature, (6, 8))
                self.assertEqual(result.beat_unit, "dotted-quarter")

    def test_attack_at_recording_start_is_not_removed(self):
        audio, rate = rhythm_audio(bpm=123, meter=4, offset=0.)
        result = analyze_samples(audio, rate)
        self.assertAlmostEqual(result.grid_origin, 0., delta=.06)

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
        self.assertTrue(all(item["score"] < .18 for item in result.meter_candidates))

    def test_unmetered_triplets_expose_compound_bpm_interpretation(self):
        rate = 11025
        audio = np.zeros(rate * 18)
        random = np.random.default_rng(95)
        t = np.arange(1200) / rate
        sound = (np.sin(2 * np.pi * 110 * t) + .2 * random.normal(size=len(t))) * np.exp(-t * 40)
        for index, when in enumerate(np.arange(.3, 17.8, .2)):
            start = int(when * rate)
            audio[start:start + len(sound)] += sound * (1 if index % 3 == 0 else .24)
        result = analyze_samples(audio, rate)
        self.assertIsNone(result.time_signature)
        self.assertTrue(any("1.5 times" in warning for warning in result.warnings))
        self.assertTrue(result.tempo_candidates)
        for candidate in result.tempo_candidates:
            self.assertAlmostEqual(candidate["quarter_bpm_if_compound"], candidate["pulse_bpm"] * 1.5)

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
