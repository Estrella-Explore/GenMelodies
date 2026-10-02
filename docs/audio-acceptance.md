# Audio rhythm acceptance checks

`tools/benchmark_audio.py` generates deterministic percussion and pitched fixtures
with NumPy and sends their PCM arrays through the public `analyze_samples` API. It
covers ordinary, fractional, slow and fast quarter-note tempi; 3/4 and 6/8;
syncopation, stereo, a silent lead-in, tempo drift, noise, and signals without a
stable pulse. Run it with the project Python environment:

```powershell
.venv\Scripts\python tools/benchmark_audio.py
.venv\Scripts\python tools/benchmark_audio.py --performance --minutes 3
.venv\Scripts\python tools/benchmark_audio.py --sweep
.venv\Scripts\python tools/benchmark_audio.py --secondary-sweep
```

The three-minute probe reports elapsed time and real-time factor. The fixtures are
controlled signals, not licensed recordings, and their scores must not be described
as real-music accuracy. Local MP3s may be used separately for smoke and performance
checks, but there is no verified human BPM or meter annotation for them. Do not
commit those audio files or present prior software estimates as ground truth.

For an optional MP3 round trip, a developer may encode a fixture with an already
installed FFmpeg and analyze it through `core.audio_decode.decode_audio`. FFmpeg is
not an application dependency and no encoder binary or media fixture is vendored.

The fixed 24-case challenge set was selected independently after initial tuning,
then shared for the final tuning pass. It is a regression challenge now, not a blind
holdout. Before those cases were shared, the first run scored 17/24 within 6%, with
no octave errors or unknown BPMs and 7 larger errors. Those errors came from fast
3/4 and 6/8 examples, where estimates landed near one-third or two-thirds of expected
quarter-note BPM. On the final tested rhythm implementation it scores 24/24 within
6%, with no octave errors or unknown BPMs; its six specially accented meter cases
score 6/6 exact. A separately selected 12-case sample scores 12/12 within 6%, with
no octave errors, unknowns, or other BPM errors; its six clear-meter checks are also
exact. Both samples use the same small synthetic signal generator, so treat them as
regression evidence rather than real-recording accuracy estimates.

The ordinary 13-case set passes all tempo and abstention checks; BPM is within 6%
for all 11 rhythmic cases. It resolves meter on 3/13 fixtures and leaves 10 unknown
where the evidence is ambiguous or unmetered. The three-minute constant-tempo probe
took 0.84 seconds and returned 123.03 BPM. The verified `core/audio_rhythm.py` source
had SHA-256 `3657b9b9a076d82679f03ef106073168bba00ab5f925810e0332dfa95bf8d3d1` at the
start and end of the serialized 13+24+12 validation. These are controlled synthetic
scores only, not evidence of accuracy on real recordings. Local MP3 smoke clips have
no verified annotations.

Final local MP3 smoke results on the verified revision used only unannotated files outside
the repository. The 199.1, 170.0, and 142.6 second clips decoded in 0.15–0.26 seconds
and analyzed in 1.21, 8.13, and 7.01 seconds. The reported BPMs were 154.01, 63.98,
and 101.54; one clip had a local-tempo-variation warning and another reported a
possible dotted-quarter alternative. These values are smoke diagnostics only, not
accuracy scores, because the clips have no verified BPM or meter labels.
