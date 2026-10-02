"""Focused, deterministic checks for the public audio rhythm-analysis contract."""
import unittest

from tools.benchmark_audio import CASES, run_cases


class AudioRhythmAcceptance(unittest.TestCase):
    def test_clear_simple_triple_and_compound_meters(self):
        names = {"plain_4_4_120", "strong_downbeat_4_4", "triple_3_4", "compound_6_8"}
        rows = [row for row in run_cases([c for c in CASES if c.name in names])]
        failures = {row["case"]: row["failures"] for row in rows if row["failures"]}
        self.assertFalse(failures, failures)

    def test_fractional_tempo_and_intro_grid(self):
        names = {"fractional_4_4_97_3", "syncopated_stereo_intro"}
        rows = run_cases([c for c in CASES if c.name in names])
        failures = {row["case"]: row["failures"] for row in rows if row["failures"]}
        self.assertFalse(failures, failures)

    def test_unmetered_signals_do_not_get_confident_meter(self):
        names = {"white_noise_abstention", "sustained_chord_abstention"}
        rows = run_cases([c for c in CASES if c.name in names])
        failures = {row["case"]: row["failures"] for row in rows if row["failures"]}
        self.assertFalse(failures, failures)

    def test_noisy_and_changing_tempo_remain_trackable(self):
        names = {"noisy_but_periodic", "tempo_drift_96_to_128"}
        rows = run_cases([c for c in CASES if c.name in names])
        failures = {row["case"]: row["failures"] for row in rows if row["failures"]}
        self.assertFalse(failures, failures)


if __name__ == "__main__":
    unittest.main()
