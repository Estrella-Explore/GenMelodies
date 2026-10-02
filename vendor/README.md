# MP3 decoder provenance

The only added external repository is [mackron/dr_libs](https://github.com/mackron/dr_libs),
used under its offered **MIT No Attribution (MIT-0)** license. MIT-0 is the MIT
license without the notice-preservation condition. We retain all notices anyway.

- File: `dr_mp3.h`, upstream header reports `v0.7.4 - TBD`, copied **without modifications**.
- Pinned upstream commit: `dfe8377631000664666519fdb83da193fd8037f4`.
- Source: https://raw.githubusercontent.com/mackron/dr_libs/dfe8377631000664666519fdb83da193fd8037f4/dr_mp3.h
- SHA-256: `997b7ee18de6e6b81e2a83f1ea9fc62aef25c62b28d48db95635f49e65de0a2f`.
- Selected license: [LICENSE.dr_mp3](LICENSE.dr_mp3), also retained in the header.
- Upstream includes code derived from minimp3 dedicated to the public domain
  under CC0. Its original notice is retained at the end of `dr_mp3.h`. No separate
  minimp3 repository or Python wrapper is imported.

`native/dr_mp3_bridge.c` and `tools/build_audio_decoder.py` are original project
code. Compilation uses an existing local GCC, Clang, or MSVC executable. Nothing
downloads a compiler, binary, FFmpeg, audio backend, cffi, or other dependency.
WAV decoding uses Python's standard library and the existing NumPy dependency.
