"""
GenMelodies — 原神乐器自动演奏乐谱生成工具

将 MIDI 文件转换为原神乐器可用的数字谱或字母谱。
"""

import argparse
import sys
import os
from pathlib import Path

from core import MidiParser, ChordDetector, Simplifier, NoteMapper, Quantizer, ScoreGenerator


def parse_args():
    p = argparse.ArgumentParser(
        description='GenMelodies — 原神乐器自动演奏乐谱生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python genmelodies.py song.mid                 # MIDI 转数字谱
  python genmelodies.py song.mid -f letter       # MIDI 转字母谱
  python genmelodies.py song.mid -t 2            # 选择第 2 轨
  python genmelodies.py song.mid -o output.txt   # 指定输出文件
        """,
    )
    p.add_argument('input', help='输入 MIDI 文件 (.mid/.midi)')
    p.add_argument('-o', '--output', help='输出文件路径（默认与输入同名 .txt）')
    p.add_argument('-f', '--format', choices=['number', 'letter'],
                   default='number', help='输出格式 (默认: number)')
    p.add_argument('-t', '--track', type=int, default=None,
                   help='指定音轨索引 (0-based, 默认全部合并)')
    p.add_argument('--no-simplify', action='store_true',
                   help='禁用简化')
    p.add_argument('--min-duration', type=float, default=0.03,
                   help='最小音符时长 秒 (默认: 0.03)')
    p.add_argument('--merge-threshold', type=float, default=0.005,
                   help='同音合并阈值 秒 (默认: 0.005)')
    p.add_argument('--grace-threshold', type=float, default=0.06,
                   help='装饰音检测阈值 秒 (默认: 0.06)')
    p.add_argument('--max-chord', type=int, default=5,
                   help='和弦最大键数 (默认: 5)')
    p.add_argument('--chord-window', type=float, default=0.02,
                   help='和弦检测窗口 秒 (默认: 0.02)')
    p.add_argument('--extract-melody', action='store_true', default=False,
                   help='从多声部中提取旋律线（去掉伴奏音）')
    p.add_argument('--synthesize', action='store_true', default=False,
                   help='反向合成试听 MIDI 文件')
    p.add_argument('--version', action='version', version='GenMelodies 0.0.1')
    return p.parse_args()


def main():
    args = parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        sys.exit(1)

    ext = Path(args.input).suffix.lower()
    if ext not in ('.mid', '.midi'):
        print(f"错误: 不支持的文件格式: {ext}")
        print("当前仅支持 .mid / .midi 格式")
        sys.exit(1)

    print(f"处理 MIDI 文件: {args.input}")

    try:
        # ── 1. 解析 ──
        print("  [1/5] 解析 MIDI...")
        parser = MidiParser()
        piece = parser.parse(args.input, track_index=args.track)
        notes = piece.merged_notes
        tempo = piece.global_tempo
        time_sig = piece.global_time_signature
        print(f"  提取 {len(notes)} 个音符, BPM={tempo:.0f}, {time_sig[0]}/{time_sig[1]}")

        if not notes:
            print("错误: 未找到有效音符")
            sys.exit(1)

        # ── 2. 和弦检测 ──
        print("  [2/5] 检测和弦...")
        chord_detector = ChordDetector(time_window=args.chord_window)
        units = chord_detector.detect(notes)
        chords_count = sum(1 for u in units if len(u.pitches) > 1)
        print(f"  生成 {len(units)} 个音乐单元 ({chords_count} 个和弦)")

        # ── 3. 简化 ──
        if not args.no_simplify:
            print("  [3/5] 简化...")
            simplifier = Simplifier(
                min_duration=args.min_duration,
                merge_threshold=args.merge_threshold,
                grace_threshold=args.grace_threshold,
                max_chord_notes=args.max_chord,
                extract_melody=args.extract_melody,
            )
            before = len(units)
            units = simplifier.simplify(units)
            print(f"  简化: {before} → {len(units)} 个单元")

        # ── 4. 量化 ──
        print("  [4/5] 量化节奏...")
        tempo_changes = [(0.0, tempo)]  # 默认
        if piece.tracks:
            tempo_changes = piece.tracks[0].tempo_changes
        quantizer = Quantizer(beats_per_measure=time_sig[0])
        measures = quantizer.quantize(units, tempo_changes)
        print(f"  划分为 {len(measures)} 个小节")

        # 小节时长（取第一个速度用于首行显示）
        beat_duration = 60.0 / tempo
        measure_duration = beat_duration * time_sig[0]

        # ── 5. 生成乐谱 ──
        print("  [5/5] 生成乐谱...")
        generator = ScoreGenerator(format_type=args.format)
        score = generator.generate(measures, measure_duration, time_sig[0])

        # 输出
        output_path = args.output or str(Path(args.input).with_suffix('.txt'))
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(score)

        print(f"\n✓ 乐谱已生成: {output_path}")
        print(f"  格式: {'数字谱' if args.format == 'number' else '字母谱'}")
        print(f"  小节数: {len(measures)}")
        print(f"  小节时长: {measure_duration:.3f}s")

        # ── 6. 反向合成试听 ──
        if args.synthesize:
            from core.synthesizer import ScoreSynthesizer
            synth_path = str(Path(output_path).with_suffix('.synth.mid'))
            print(f"\n  [合成] 生成试听 MIDI: {synth_path}")
            try:
                synth = ScoreSynthesizer()
                synth.synthesize(score, synth_path)
                print(f"  ✓ 试听文件已生成: {synth_path}")
            except Exception as e:
                print(f"  ⚠ 合成失败: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
