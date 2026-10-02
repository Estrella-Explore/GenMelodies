"""文件入口、人工覆盖、MIDI 兼容和量化时间回归。"""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave

import mido
import numpy as np

import genmelodies
from core.audio_parser import AudioParser
from core.audio_rhythm import AudioRhythmResult
from core.models import MusicUnit
from core.quantizer import Quantizer
from core.score_generator import ScoreGenerator


def write_wav(path, samples, rate=22050):
    data = np.asarray(samples)
    channels = 1 if data.ndim == 1 else data.shape[1]
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((np.clip(data, -1, 1) * 32767).astype('<i2').tobytes())


class AudioIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = genmelodies.main(list(args))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_wav_analysis_json_overrides_without_pitch(self):
        path = self.directory / '静音.wav'
        write_wav(path, np.zeros(22050 * 4))
        code, output, _ = self.invoke(str(path), '--analyze-rhythm', '--json', '--bpm', '120',
                                     '--time-signature', '6/8', '--beat-offset', '0.25')
        self.assertEqual(code, 0)
        analysis = json.loads(output)
        self.assertEqual(analysis['bpm'], 120)
        self.assertEqual(analysis['time_signature'], [6, 8])
        self.assertAlmostEqual(analysis['grid_origin'], 0.25)
        self.assertEqual(analysis['bpm_unit'], 'quarter-note')
        self.assertAlmostEqual(analysis['beat_times'][1] - analysis['beat_times'][0], 0.75)
        self.assertFalse(path.with_suffix('.txt').exists())

    def test_silence_unknown_and_diagnostics_survive_score_refusal(self):
        path = self.directory / 'silence.wav'
        diagnostics = self.directory / 'rhythm.json'
        write_wav(path, np.zeros(22050 * 4))
        code, _, error = self.invoke(str(path), '--rhythm-json', str(diagnostics))
        self.assertEqual(code, 1)
        self.assertIn('--bpm', error)
        analysis = json.loads(diagnostics.read_text(encoding='utf-8'))
        self.assertIsNone(analysis['bpm'])
        self.assertIsNone(analysis['time_signature'])
        self.assertFalse(path.with_suffix('.txt').exists())

    def test_audio_missing_meter_never_defaults_to_four_four(self):
        rhythm = AudioRhythmResult(bpm=120)
        with self.assertRaisesRegex(ValueError, '--time-signature'):
            AudioParser().parse_samples(np.zeros(22050), 22050, rhythm)

    def test_real_mp3_cli_decode_analyze_and_manual_grid(self):
        ffmpeg = os.environ.get('GENMELODIES_TEST_FFMPEG') or shutil.which('ffmpeg')
        if not ffmpeg or Path(ffmpeg).suffix.lower() in ('.bat', '.cmd'):
            self.skipTest('可选 MP3 fixture 编码器不可用；运行依赖不需要 ffmpeg')
        wav_path, mp3_path = self.directory / '点击.wav', self.directory / '点击.mp3'
        rate = 22050
        signal = np.zeros(rate * 8)
        rng = np.random.default_rng(12)
        pulse = rng.normal(size=600) * np.exp(-np.arange(600) / 120)
        for index in range(16):
            start = int((.25 + index * .5) * rate)
            signal[start:start + 600] += pulse * (.5 if index % 4 == 0 else .2)
        write_wav(wav_path, signal, rate)
        subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', str(wav_path), '-c:a', 'libmp3lame',
                        '-b:a', '128k', str(mp3_path)], check=True, capture_output=True, timeout=30)
        code, output, error = self.invoke(str(mp3_path), '--analyze-rhythm', '--json', '--bpm', '120',
                                         '--time-signature', '4/4', '--beat-offset', '0.25')
        self.assertEqual(code, 0, error)
        analysis = json.loads(output)
        self.assertEqual(analysis['bpm'], 120)
        self.assertEqual(analysis['time_signature'], [4, 4])
        self.assertAlmostEqual(analysis['grid_origin'], .25)
        self.assertGreater(len(analysis['beat_times']), 12)
        self.assertGreater(len(analysis['downbeat_times']), 2)

    def test_pitch_independent_of_rhythm_and_antiphase_stereo(self):
        rate = 22050
        samples = .2 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
        notes = AudioParser.estimate_monophonic_notes(np.column_stack((samples, -samples)), rate)
        self.assertGreater(len(notes), 0)
        self.assertTrue(all(note.pitch == 69 for note in notes))
        self.assertGreater(notes[0].end - notes[0].start, .8)

    def test_audio_single_note_generates_score_with_manual_grid(self):
        path = self.directory / 'solo.wav'
        rate = 22050
        write_wav(path, .2 * np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate), rate)
        code, _, error = self.invoke(str(path), '--bpm', '120', '--time-signature', '3/4')
        self.assertEqual(code, 0, error)
        score = path.with_suffix('.txt').read_text(encoding='utf-8')
        self.assertAlmostEqual(float(score.splitlines()[0]), 1.5)
        self.assertIn('6', score)

    def test_cli_invalid_overrides_fail_before_file_access(self):
        for flag, value in [('--bpm', 'nan'), ('--bpm', '0'), ('--bpm', 'inf'),
                            ('--bpm', '19.99'), ('--bpm', '400.01'), ('--bpm', '1e308'),
                            ('--time-signature', '3/3'), ('--time-signature', '0/4'),
                            ('--time-signature', '4/4/4'), ('--beat-offset', '-1')]:
            with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    genmelodies.parse_args(['missing.mp3', flag, value])
                self.assertEqual(raised.exception.code, 2)

    def test_cli_bpm_limits_inclusive_for_audio_and_midi(self):
        for extension in ('mp3', 'mid'):
            for bpm in ('20', '400'):
                with self.subTest(extension=extension, bpm=bpm):
                    args = genmelodies.parse_args([f'missing.{extension}', '--bpm', bpm])
                    self.assertEqual(args.bpm, float(bpm))

    def test_cli_output_collisions_refused_without_touching_inputs(self):
        source = self.directory / 'original.mp3'
        source.write_bytes(b'original-data')
        default_score = source.with_suffix('.txt')
        cases = [
            [str(source), '--rhythm-json', str(default_score)],
            [str(source), '-o', str(self.directory / 'same.json'), '--rhythm-json', str(self.directory / 'same.json')],
            [str(source), '-o', str(source)],
            [str(self.directory / 'original.synth.mid'), '-o', str(default_score), '--synthesize'],
            [str(source), '-o', str(default_score), '--synthesize', '--rhythm-json', str(default_score.with_suffix('.synth.mid'))],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    genmelodies.parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)
        self.assertEqual(source.read_bytes(), b'original-data')
        self.assertFalse(default_score.exists())

    def test_midi_default_command_and_compound_meter(self):
        path = self.directory / 'meter.mid'
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=500000))
        track.append(mido.MetaMessage('time_signature', numerator=6, denominator=8))
        track.append(mido.Message('note_on', note=60, velocity=90, time=240))
        track.append(mido.Message('note_off', note=60, time=240))
        midi.save(path)
        code, _, error = self.invoke(str(path), '--no-simplify')
        self.assertEqual(code, 0, error)
        lines = path.with_suffix('.txt').read_text(encoding='utf-8').splitlines()
        self.assertEqual(float(lines[0]), 1.5)
        self.assertEqual(len(lines[1].split('/')[0]), 24)
        self.assertTrue(lines[1].startswith('    1'))

    def test_synthesis_preserves_leading_rests_and_whole_empty_bar(self):
        from core.midi_parser import MidiParser
        from core.synthesizer import ScoreSynthesizer
        units = [MusicUnit([60], .5, .6), MusicUnit([62], 4.5, 4.6)]
        measures = Quantizer().quantize(units, [(0, 120)])
        text = ScoreGenerator().generate(measures, 2)
        path = self.directory / 'roundtrip.mid'
        ScoreSynthesizer().synthesize(text, str(path))
        notes = MidiParser().parse(str(path)).merged_notes
        self.assertEqual([note.pitch for note in notes], [60, 62])
        self.assertAlmostEqual(notes[0].start, .5, places=2)
        self.assertAlmostEqual(notes[1].start, 4.5, places=2)

    def test_corrupt_audio_has_actionable_error_without_traceback(self):
        path = self.directory / 'broken.wav'
        path.write_bytes(b'not audio')
        code, _, error = self.invoke(str(path), '--analyze-rhythm')
        self.assertEqual(code, 1)
        self.assertIn('错误:', error)
        self.assertNotIn('Traceback', error)


class QuantizerTimingTests(unittest.TestCase):
    def test_fixed_rest_slots_keep_onset_time_in_serialized_score(self):
        measures = Quantizer().quantize([MusicUnit([60], .5, .6), MusicUnit([62], 1.5, 1.6)], [(0, 120)])
        self.assertEqual(len(measures[0]), 16)
        self.assertEqual([i for i, slot in enumerate(measures[0]) if slot.pitches], [4, 12])
        text = ScoreGenerator().generate(measures, 2)
        self.assertEqual(text.splitlines()[1], '    1       2   /')

    def test_downbeat_origin_and_complete_empty_bars(self):
        units = [MusicUnit([60], .75, .85), MusicUnit([62], 4.75, 4.85)]
        measures = Quantizer(grid_origin=.25).quantize(units, [(0, 120)])
        self.assertEqual(len(measures), 3)
        self.assertEqual(measures[0][4].pitches, [60])
        self.assertTrue(all(not slot.pitches for slot in measures[1]))
        self.assertEqual(measures[2][4].pitches, [62])

    def test_compound_denominator_and_midbar_tempo_change(self):
        compound = Quantizer(6, beat_denominator=8)
        measures = compound.quantize([MusicUnit([60], 0, .1), MusicUnit([62], 1.5, 1.6)], [(0, 120)])
        self.assertEqual(len(measures), 2)
        self.assertEqual(len(measures[0]), 24)
        # 第一秒是 2 拍；之后 60 BPM，故 t=2 是第 3 拍（槽 12）。
        measures = Quantizer().quantize([MusicUnit([60], 2, 2.1)], [(0, 120), (1, 60)])
        self.assertEqual(measures[0][12].pitches, [60])

    def test_pickup_before_first_detected_downbeat_keeps_phase(self):
        measures = Quantizer(grid_origin=1).quantize([MusicUnit([60], .5, .6)], [(0, 120)])
        self.assertEqual(measures[0][12].pitches, [60])

    def test_nonpositive_tempo_is_rejected(self):
        with self.assertRaises(ValueError):
            Quantizer().quantize([MusicUnit([60], 0, 1)], [(0, 0)])

    def test_extreme_public_api_tempo_has_clear_validation_error(self):
        with self.assertRaisesRegex(ValueError, '有限拍点'):
            Quantizer().quantize([MusicUnit([60], 3, 4)], [(0, 1e308)])


if __name__ == '__main__':
    unittest.main()
