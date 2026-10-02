"""Build the vendored MIT dr_mp3 decoder with a local C compiler.

No downloads, shell execution, FFmpeg, or Python build dependencies are used.
Run ``python tools/build_audio_decoder.py`` to prepare the decoder explicitly.
``CC`` may name a compiler executable; flags in CC are deliberately unsupported.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DecoderBuildError(RuntimeError):
    """The optional MP3 native decoder could not be built locally."""


def _cache_directory() -> Path:
    preferred = PROJECT_ROOT / ".audio-native"
    try:
        preferred.mkdir(exist_ok=True)
        with tempfile.TemporaryFile(dir=preferred):
            pass
        return preferred
    except OSError:
        if sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Caches"
        else:
            base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        fallback = base / "GenMelodies" / "audio-native"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def library_path(cache_dir: Path | None = None) -> Path:
    """Cache binaries by source digest, target OS, architecture and pointer size."""
    digest = hashlib.sha256()
    for relative in ("native/dr_mp3_bridge.c", "vendor/dr_mp3.h"):
        digest.update((PROJECT_ROOT / relative).read_bytes())
    digest.update(f"{sys.platform}-{platform.machine()}-{struct.calcsize('P')}".encode())
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    return (cache_dir or _cache_directory()) / f"dr_mp3_{digest.hexdigest()[:20]}{suffix}"


def _compiler() -> str:
    requested = os.environ.get("CC")
    if requested:
        executable = shutil.which(requested)
        if executable:
            return executable
        raise DecoderBuildError(f"CC compiler executable does not exist: {requested!r}")
    for name in ("gcc", "clang", "cl"):
        executable = shutil.which(name)
        if executable:
            return executable
    raise DecoderBuildError(
        "MP3 decoding needs a local C compiler (GCC, Clang, or MSVC). "
        "Install one and put it on PATH, or set CC to its executable, then run "
        "python tools/build_audio_decoder.py. PCM WAV decoding needs no compiler."
    )


def build_decoder(cache_dir: Path | None = None, force: bool = False) -> Path:
    target = library_path(cache_dir)
    if target.is_file() and not force:
        return target
    compiler = _compiler()
    target.parent.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / "native" / "dr_mp3_bridge.c"
    with tempfile.TemporaryDirectory(prefix="dr-mp3-build-", dir=target.parent) as temporary:
        output = Path(temporary) / target.name
        compiler_name = Path(compiler).name.lower()
        if compiler_name in ("cl", "cl.exe"):
            command = [compiler, "/nologo", "/O2", "/LD", "/MT", str(source),
                       "/link", f"/OUT:{output}"]
        else:
            command = [compiler, "-O2", "-std=c99"]
            if sys.platform == "win32":
                command += ["-shared", "-static-libgcc"]
            elif sys.platform == "darwin":
                command += ["-dynamiclib", "-fPIC"]
            else:
                command += ["-shared", "-fPIC"]
            command += [str(source), "-o", str(output)]
            if sys.platform != "win32":
                command += ["-lm"]
        environment = os.environ.copy()
        environment["PATH"] = str(Path(compiler).parent) + os.pathsep + environment.get("PATH", "")
        try:
            result = subprocess.run(command, cwd=temporary, env=environment,
                                    capture_output=True, text=True, errors="replace",
                                    timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DecoderBuildError(f"Cannot build MP3 decoder with {compiler}: {exc}") from exc
        if result.returncode != 0 or not output.is_file():
            diagnostic = (result.stdout + result.stderr).strip()[-6000:]
            raise DecoderBuildError(
                f"MP3 decoder compilation failed using {compiler} "
                f"(exit code {result.returncode}):\n"
                f"{diagnostic or 'No compiler diagnostic; check compiler runtime DLLs and toolchain installation.'}"
            )
        try:
            os.replace(output, target)
        except PermissionError:
            # A simultaneous builder may already have published/loaded this DLL.
            if force or not target.is_file():
                raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild the cached decoder")
    args = parser.parse_args()
    try:
        print(build_decoder(force=args.force))
    except (DecoderBuildError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
