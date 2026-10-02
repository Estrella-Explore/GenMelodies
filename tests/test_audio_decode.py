"""PCM edge cases and optional round-trip MP3 decoding with a developer encoder.

FFmpeg is a fixture generator only, never a runtime/project dependency. Set
GENMELODIES_TEST_FFMPEG to a locally installed encoder executable to enable the
round-trip tests when ffmpeg is not on PATH.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np

from core.audio_decode import AudioDecodeError, decode_audio
from tools.build_audio_decoder import DecoderBuildError, build_decoder


class AudioDecodeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_wav(self, width, values, channels=1):
        path = self.directory / f"音频_{width}.wav"
        values = np.asarray(values, dtype=np.int64)
        if width == 1:
            content = values.astype(np.uint8).tobytes()
        elif width == 3:
            packed = values & 0xFFFFFF
            content = np.column_stack((packed & 255, (packed >> 8) & 255,
                                       (packed >> 16) & 255)).astype(np.uint8).tobytes()
        else:
            content = values.astype("<i2" if width == 2 else "<i4").tobytes()
        with wave.open(str(path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(width)
            output.setframerate(32000)
            output.writeframes(content)
        return path

    def test_pcm_integer_boundaries(self):
        for width in (1, 2, 3, 4):
            with self.subTest(width=width):
                bound = 2 ** (8 * width - 1)
                values = [0, 128, 255] if width == 1 else [-bound, 0, bound - 1]
                audio, rate = decode_audio(self.write_wav(width, values))
                expected = (np.asarray(values, dtype=np.float64) - 128) / 128 if width == 1 else np.asarray(values, dtype=np.float64) / bound
                self.assertEqual(rate, 32000)
                self.assertEqual(audio.shape, (3, 1))
                self.assertEqual(audio.dtype, np.float32)
                np.testing.assert_allclose(audio[:, 0], expected, atol=6e-8)

    def test_stereo_phase_is_preserved(self):
        audio, _ = decode_audio(self.write_wav(2, [10000, -10000, -12000, 12000], channels=2))
        self.assertEqual(audio.shape, (2, 2))
        np.testing.assert_array_equal(audio[:, 0], -audio[:, 1])

    def test_truncated_and_empty_wav_rejected(self):
        path = self.write_wav(2, [1, 2, 3, 4])
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(AudioDecodeError, "truncated"):
            decode_audio(path)
        empty = self.write_wav(2, [])
        with self.assertRaises(AudioDecodeError):
            decode_audio(empty)

    def test_invalid_wav_and_unsupported_format(self):
        path = self.directory / "invalid.wav"
        path.write_bytes(b"This is not RIFF PCM")
        with self.assertRaises(AudioDecodeError):
            decode_audio(path)
        with self.assertRaisesRegex(AudioDecodeError, "Unsupported"):
            decode_audio(self.directory / "audio.flac")

    def test_empty_mp3_rejected_without_compiler(self):
        path = self.directory / "empty.mp3"
        path.write_bytes(b"")
        with mock.patch("core.audio_decode._native_decoder", side_effect=AssertionError("must not compile")):
            with self.assertRaisesRegex(AudioDecodeError, "empty"):
                decode_audio(path)

    def test_missing_compiler_message(self):
        with mock.patch.dict(os.environ, {"CC": "missing-genmelodies-c-compiler"}):
            with self.assertRaisesRegex(DecoderBuildError, "CC compiler executable"):
                build_decoder(cache_dir=self.directory)

    def test_invalid_mp3_rejected(self):
        # A decoder compiler/cache is needed only for this native test.
        try:
            build_decoder()
        except DecoderBuildError as exc:
            self.skipTest(str(exc))
        path = self.directory / "damaged.mp3"
        path.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"garbage" * 100)
        with self.assertRaisesRegex(AudioDecodeError, "No valid MPEG"):
            decode_audio(path)

    def test_mp3_round_trip_mono_and_stereo(self):
        ffmpeg = os.environ.get("GENMELODIES_TEST_FFMPEG") or shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")
        if not ffmpeg or Path(ffmpeg).suffix.lower() in (".cmd", ".bat"):
            self.skipTest("No development FFmpeg executable; set GENMELODIES_TEST_FFMPEG")
        try:
            build_decoder()
        except DecoderBuildError as exc:
            self.skipTest(str(exc))
        time = np.arange(32000) / 32000
        signal = (14000 * np.sin(2 * np.pi * 440 * time)).astype(np.int16)
        for channels in (1, 2):
            with self.subTest(channels=channels):
                values = signal if channels == 1 else np.column_stack((signal, -signal)).ravel()
                wav_path = self.write_wav(2, values, channels=channels)
                mp3_path = self.directory / f"试听声相_{channels}.mp3"
                subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(wav_path),
                                "-c:a", "libmp3lame", "-b:a", "128k", str(mp3_path)],
                               check=True, capture_output=True, timeout=30)
                audio, rate = decode_audio(mp3_path)
                self.assertEqual(rate, 32000)
                self.assertEqual(audio.shape[1], channels)
                self.assertLess(abs(len(audio) - 32000), 2304)
                self.assertTrue(np.isfinite(audio).all())
                self.assertGreater(float(np.sqrt(np.mean(audio ** 2))), 0.2)
                if channels == 2:
                    self.assertLess(float(np.corrcoef(audio.T)[0, 1]), -0.95)


if __name__ == "__main__":
    unittest.main()
