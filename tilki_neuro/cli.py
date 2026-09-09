import argparse
import json
from pathlib import Path
from .data import synthetic, load
from .evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description='TILKI NEURO research toolkit')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('benchmark', 'evaluate'):
        p = sub.add_parser(name)
        if name == 'evaluate':
            p.add_argument('dataset', type=Path)
        p.add_argument('--output', type=Path, default=Path('outputs'))
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--idle-fa-target', type=float, default=.05)
    p = sub.add_parser('prepare-physionet')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--subjects', type=int, nargs='+', required=True)
    p.add_argument('--download', action='store_true', help='Explicitly allow MNE to download EDFs')
    p.add_argument('--output', type=Path, default=Path('data/physionet.npz'))
    p = sub.add_parser('study', help='Run the frozen spatial EEG comparison')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=Path('protocols/v0.2.json'))
    p.add_argument('--output', type=Path, default=Path('outputs/study-v0.2'))
    p = sub.add_parser('console', help='Inspect a stored study in the desktop research console')
    p.add_argument('report', type=Path)
    p = sub.add_parser('figures', help='Export a study evidence dashboard')
    p.add_argument('report', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/figures'))
    sub.add_parser('demo')
    args = parser.parse_args()
    if args.command == 'console':
        from .console import main as console
        console(args.report)
        return
    if args.command == 'figures':
        from .figures import export
        export(args.report, args.output)
        return
    if args.command == 'study':
        from .study import run
        run(args.root, args.protocol, args.output)
        return
    if args.command == 'demo':
        from .demo import main as demo
        demo()
        return
    if args.command == 'prepare-physionet':
        from .physionet import prepare
        data = prepare(args.root, args.subjects, args.download)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.save(args.output)
        print(f'Saved {len(data.y)} non-overlapping windows to {args.output}')
        return
    data = synthetic(seed=args.seed) if args.command == 'benchmark' else load(args.dataset)
    report, replay = evaluate(data, args.seed, args.idle_fa_target)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    # No raw signal replay is exported from real human datasets by default.
    if args.command == 'benchmark':
        (args.output/'synthetic_replay.json').write_text(json.dumps({'source': data.source, 'windows': replay}, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
