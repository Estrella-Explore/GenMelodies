"""GenMelodies：MIDI 乐谱转换、MP3/WAV 节奏分析和粗略单音乐谱。"""

import argparse
import json
import math
import sys
from pathlib import Path

from core import MidiParser, ChordDetector, Simplifier, Quantizer, ScoreGenerator


def bpm_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("BPM 必须是 20–400 之间的有限数") from exc
    if not math.isfinite(number) or not 20 <= number <= 400:
        raise argparse.ArgumentTypeError("BPM 必须是 20–400 之间的有限数")
    return number


def nonnegative_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是有限的非负数") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("必须是有限的非负数")
    return number


def time_signature(value):
    try:
        numerator, denominator = map(int, value.split('/'))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("拍号格式必须是 N/D，例如 4/4 或 6/8") from exc
    if not 1 <= numerator <= 32 or denominator not in (1, 2, 4, 8, 16, 32):
        raise argparse.ArgumentTypeError("拍号分子应在 1–32，分母应为 1、2、4、8、16 或 32")
    return numerator, denominator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='GenMelodies — MIDI 乐谱 / MP3、WAV 节奏分析')
    parser.add_argument('input', help='输入文件 (.mid/.midi/.mp3/.wav)')
    parser.add_argument('-o', '--output', help='乐谱输出文件；分析模式下用于保存 JSON')
    parser.add_argument('-f', '--format', choices=['number', 'letter'], default='number')
    parser.add_argument('-t', '--track', type=int, help='MIDI 音轨索引 (0-based)')
    parser.add_argument('--analyze-rhythm', action='store_true', help='仅分析音频节奏，不提取音高或生成乐谱')
    parser.add_argument('--json', action='store_true', help='分析模式：向标准输出写纯 JSON（诊断写 stderr）')
    parser.add_argument('--rhythm-json', help='另存节奏诊断 JSON，包含候选、置信指标和拍点')
    parser.add_argument('--bpm', type=bpm_float, help='人工指定四分音符 BPM（20–400），包括 6/8')
    parser.add_argument('--time-signature', type=time_signature, help='人工指定拍号，例如 3/4、6/8')
    parser.add_argument('--beat-offset', type=nonnegative_float, help='人工指定第一小节下拍时间（秒）')
    parser.add_argument('--no-simplify', action='store_true')
    parser.add_argument('--min-duration', type=nonnegative_float, default=0.03)
    parser.add_argument('--merge-threshold', type=nonnegative_float, default=0.005)
    parser.add_argument('--grace-threshold', type=nonnegative_float, default=0.06)
    parser.add_argument('--max-chord', type=int, default=5)
    parser.add_argument('--chord-window', type=nonnegative_float, default=0.02)
    parser.add_argument('--extract-melody', action='store_true', help='MIDI 多声部简化；音频不会分离伴奏')
    parser.add_argument('--synthesize', action='store_true', help='生成乐谱后合成试听 MIDI')
    parser.add_argument('--version', action='version', version='GenMelodies 0.1.0')
    args = parser.parse_args(argv)
    if args.json and not args.analyze_rhythm:
        parser.error('--json 必须与 --analyze-rhythm 一起使用')
    if args.track is not None and args.track < 0:
        parser.error('--track 不得为负数')
    if args.max_chord < 1:
        parser.error('--max-chord 必须为正整数')
    if args.analyze_rhythm and args.synthesize:
        parser.error('--synthesize 需要生成乐谱，不能与 --analyze-rhythm 一起使用')
    input_path = Path(args.input).resolve()
    output_path = Path(args.output) if args.output else (
        None if args.analyze_rhythm else Path(args.input).with_suffix('.txt'))
    outputs = [('输出文件', output_path)]
    if args.rhythm_json:
        outputs.append(('节奏诊断 JSON', Path(args.rhythm_json).resolve()))
    if args.synthesize:
        outputs.append(('试听 MIDI', output_path.with_suffix('.synth.mid')))
    seen_outputs = {}
    for label, path in outputs:
        if path is None:
            continue
        path = path.resolve()
        if path == input_path:
            parser.error(f'{label}路径不能覆盖输入文件')
        if path in seen_outputs:
            parser.error(f'{label}与{seen_outputs[path]}不能使用相同路径')
        seen_outputs[path] = label
    return args


def write_json(filepath, data):
    Path(filepath).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                              encoding='utf-8')


def main(argv=None):
    args = parse_args(argv)
    source = Path(args.input)
    if not source.is_file():
        print(f'错误: 文件不存在: {source}', file=sys.stderr)
        return 1
    ext = source.suffix.lower()
    if ext not in ('.mid', '.midi', '.mp3', '.wav'):
        print(f'错误: 不支持 {ext}；支持 .mid / .midi / .mp3 / .wav', file=sys.stderr)
        return 1
    is_audio = ext in ('.mp3', '.wav')
    if not is_audio and (args.analyze_rhythm or args.rhythm_json):
        print('错误: 节奏音频诊断适用于 MP3/WAV；MIDI 使用文件内速度和拍号事件', file=sys.stderr)
        return 1
    if is_audio and args.track is not None:
        print('错误: 音频没有 MIDI 音轨索引，请移除 --track', file=sys.stderr)
        return 1
    try:
        if is_audio:
            from core.audio_parser import AudioParser
            from core.audio_rhythm import analyze_samples
            audio_parser = AudioParser()
            samples, sample_rate = audio_parser.decode(str(source))
            rhythm = analyze_samples(samples, sample_rate, bpm=args.bpm,
                                     time_signature=args.time_signature, beat_offset=args.beat_offset)
            diagnostics = rhythm.to_dict()
            if args.rhythm_json:
                write_json(args.rhythm_json, diagnostics)
            for warning in rhythm.warnings:
                print(f'警告: {warning}', file=sys.stderr)
            if args.analyze_rhythm:
                if args.output:
                    write_json(args.output, diagnostics)
                if args.json:
                    print(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False))
                else:
                    bpm_text = f'{rhythm.bpm:.2f}' if rhythm.bpm is not None else '未知'
                    meter_text = '/'.join(map(str, rhythm.time_signature)) if rhythm.time_signature else '未知'
                    print(f'四分音符 BPM: {bpm_text}；拍号: {meter_text}；小节网格起点: {rhythm.grid_origin:.3f}s')
                    print(f'速度置信指标: {rhythm.tempo_confidence:.3f}；拍号置信指标: {rhythm.meter_confidence:.3f}')
                    if args.output:
                        print(f'诊断 JSON 已保存: {args.output}')
                return 0
            print('警告: 音频乐谱仅为粗略单音提取；完整歌曲中的人声、和弦和伴奏会产生误识别。', file=sys.stderr)
            piece = audio_parser.parse_samples(samples, sample_rate, rhythm)
        else:
            piece = MidiParser().parse(str(source), track_index=args.track)
            if args.bpm is not None:
                piece.global_tempo = args.bpm
                for track in piece.tracks:
                    track.tempo_changes = [(0.0, args.bpm)]
            if args.time_signature is not None:
                piece.global_time_signature = args.time_signature
            if args.beat_offset is not None:
                piece.grid_origin = args.beat_offset
        if not piece.merged_notes:
            raise ValueError('未找到有效音符；音频请先用 --analyze-rhythm 分析节奏，单音提取可能不适用于此录音')
        tempo = piece.global_tempo
        numerator, denominator = piece.global_time_signature
        units = ChordDetector(time_window=args.chord_window).detect(piece.merged_notes)
        if not args.no_simplify:
            units = Simplifier(min_duration=args.min_duration, merge_threshold=args.merge_threshold,
                               grace_threshold=args.grace_threshold, max_chord_notes=args.max_chord,
                               extract_melody=args.extract_melody).simplify(units)
        if not units:
            raise ValueError('简化后没有音符；尝试 --no-simplify 或降低 --min-duration')
        tempo_changes = (piece.tracks[0].tempo_changes if piece.tracks else []) or [(0.0, tempo)]
        quantizer = Quantizer(beats_per_measure=numerator, beat_denominator=denominator,
                              grid_origin=piece.grid_origin)
        measures = quantizer.quantize(units, tempo_changes)
        measure_duration = 60.0 / tempo * numerator * 4.0 / denominator
        if len(tempo_changes) > 1 or (piece.tracks and len(piece.tracks[0].time_sig_changes) > 1):
            print('警告: 当前文本谱只记录首个小节时长和拍号；变速、变拍号的播放只能近似。', file=sys.stderr)
        score = ScoreGenerator(format_type=args.format).generate(measures, measure_duration, numerator)
        output_path = Path(args.output) if args.output else source.with_suffix('.txt')
        output_path.write_text(score, encoding='utf-8')
        print(f'乐谱已生成: {output_path}；{len(measures)} 小节；BPM={tempo:.2f}；{numerator}/{denominator}')
        if args.synthesize:
            from core.synthesizer import ScoreSynthesizer
            synth_path = str(output_path.with_suffix('.synth.mid'))
            ScoreSynthesizer().synthesize(score, synth_path, beats_per_measure=numerator * 4 / denominator)
            print(f'试听 MIDI 已生成: {synth_path}')
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'错误: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
