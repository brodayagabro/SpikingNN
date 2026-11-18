"""Local, explicit entry point for the AfferentBalance2026 experiments.

This file only translates CLI arguments into requests. It does not define a
model, genetic operation, plotting implementation or installed package command.
"""
import argparse
import json
from pathlib import Path
import sys

# Direct-file execution from any cwd. No home-directory or parent-depth assumptions.
for root in Path(__file__).resolve().parents:
    if (root/'src/SpikingNN').is_dir():
        sys.path.insert(0, str(root))
        break
from scripts.AfferentBalance2026.experiments.core.config import resolve_config
from scripts.AfferentBalance2026.experiments.core import artifacts
from scripts.AfferentBalance2026.experiments.core.run import (
    execute, execute_sweep, inspect_request, replay_solution, sweep_configurations,
)


def build_parser():
    """Define operation-specific flags; misplaced arguments are rejected by argparse."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('validate', 'train', 'finetune', 'sweep', 'replay', 'prepare'):
        command = commands.add_parser(action, help={
            'validate':'Resolve settings, dimensions, sources and budgets without integration.',
            'train':'Train from an explicit configuration.', 'finetune':'Continue from an explicit physical solution.',
            'sweep':'Run independent GA lines defined by configuration axes.',
            'replay':'Evaluate saved physical dynamics on explicit banks/grids.',
            'prepare':'Transform only explicitly named fields into a destination domain.'}[action])
        command.add_argument('--config', type=Path, required=action in ('validate', 'train', 'sweep', 'prepare'),
                             help='JSON; paths inside it are relative to that file.')
        if action in ('validate', 'finetune', 'replay', 'prepare'):
            command.add_argument('--solution', type=Path, required=action in ('finetune', 'replay', 'prepare'),
                                 help='Portable JSON, explicit legacy JSON/NPZ, or current-solution pointer.')
        if action != 'validate':
            command.add_argument('--output', type=Path, required=action in ('finetune', 'replay', 'prepare'),
                                 help='New result directory; CLI paths are relative to cwd.')
        if action in ('validate', 'train', 'finetune', 'sweep', 'replay'):
            command.add_argument('--workers', type=int, help='Override persistent process count.')
        if action in ('validate', 'train', 'finetune', 'sweep'):
            command.add_argument('--seed', type=int, help='Override GA seed (bank seeds are separate).')
        if action == 'replay':
            command.add_argument('--recording-stride', type=int, help='Record every Nth 0.1 ms step; metric sampling stays unchanged.')
        if action == 'validate':
            command.add_argument('--json', action='store_true', help='Print the full effective configuration, provenance and inspection.')
    compare = commands.add_parser('compare', help='Compare existing metrics within explicit protocol groups; never retrain.')
    compare.add_argument('--runs', type=Path, nargs='+', required=True)
    compare.add_argument('--output', type=Path, required=True)
    publish = commands.add_parser('publish', help='Explicitly update a checked current-solution pointer.')
    publish.add_argument('--solution', type=Path, required=True)
    publish.add_argument('--pointer', type=Path, required=True)
    report = commands.add_parser('report', help='Build a local evidence report from an explicit completed manifest.')
    report.add_argument('--manifest', type=Path, required=True)
    report.add_argument('--output', type=Path, required=True)
    return parser


def main(argv=None):
    """Resolve one request, dispatch it, and print a concise input error on failure."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.action == 'compare':
            result = artifacts.compare_runs(args.runs, args.output)
        elif args.action == 'publish':
            result = artifacts.publish(args.solution, args.pointer)
        elif args.action == 'report':
            from scripts.AfferentBalance2026.experiments.visualization import build_run_report
            result = build_run_report(args.manifest, args.output)
        else:
            overrides = {}
            for attribute, block, key in (('output', 'storage', 'output'), ('workers', 'runtime', 'workers'),
                                           ('seed', 'search', 'seed'), ('recording_stride', 'storage', 'recording_stride')):
                value = getattr(args, attribute, None)
                if value is not None:
                    overrides.setdefault(block, {})[key] = str(value) if isinstance(value, Path) else value
            effective = resolve_config(args.config, overrides=overrides, action=args.action,
                                       solution=getattr(args, 'solution', None))
            if args.action == 'validate':
                inspection = inspect_request(effective, check_output=False)
                if effective['sweep']['axes']:
                    lines = sweep_configurations(effective)
                    inspection['independent_lines'] = [inspect_request(line, check_output=False) for line in lines]
                result = dict(effective_config=effective, inspection=inspection) if args.json else inspection
            elif args.action in ('train', 'finetune'):
                result = execute(effective)
            elif args.action == 'sweep':
                result = execute_sweep(effective)
            elif args.action == 'prepare':
                result = artifacts.prepare_solution(effective)
            else:
                result = replay_solution(effective)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f'{args.action}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())


def legacy_main(experiment=None, *, sweep=False, warm=False, argv=None):
    """Translate historical flags once, keeping their documented meaning visible.

    Old A/B --generations lists are checkpoints of one line. New sweep axes use
    independent horizons. Historical Python APIs remain available for archives;
    these compatibility entry points write the unified result format.
    """
    from scripts.AfferentBalance2026.experiments.core.config import project_root, read_json
    parser = argparse.ArgumentParser(description='Compatibility entry point; prefer the local run.py commands.')
    if warm:
        parser.add_argument('--experiment', choices=list('ABCD'), required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=warm)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--population', type=int)
    parser.add_argument('--generations', type=int, nargs='+' if sweep and experiment in 'AB' else None)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--mutation-probability', type=float)
    parser.add_argument('--seed-genome', type=Path, required=warm)
    parser.add_argument('--extra-seed', type=Path, action='append', default=[])
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--degree', type=int, choices=[2, 3])
    parser.add_argument('--table-generations', type=int)
    parser.add_argument('--mode', choices=['replay', 'joint', 'staged', 'compare'], default='replay' if experiment == 'C' and not warm else None)
    parser.add_argument('--method', choices=['joint', 'staged', 'command_only'])
    parser.add_argument('--sources', type=Path, nargs='+')
    for flag, value_type in [('populations', int), ('probabilities', float), ('seeds', int), ('degrees', int)]:
        parser.add_argument('--'+flag, type=value_type, nargs='+')
    for flag in ('sigma-start', 'sigma-end'):
        parser.add_argument('--'+flag, type=float)
    parser.add_argument('--validation-every', type=int)
    parser.add_argument('--polarity-seeds', action=argparse.BooleanOptionalAction, default=None)
    if experiment == 'C' and not warm:
        parser.add_argument('--genome', type=Path)
        for flag in ('targets', 'angles', 'speeds'):
            parser.add_argument('--'+flag, type=float, nargs='+')
        current = parser.add_mutually_exclusive_group()
        current.add_argument('--currents', type=float, nargs=2)
        current.add_argument('--current-law', type=Path)
    args = parser.parse_args(argv)
    kind = args.experiment if warm else experiment
    names = dict(A='A_balance', B='B_balance', C='C_setpoint_balance', D='D_trajectory')
    config = args.config or (None if warm else project_root()/'experiments'/names[kind]/'genome_cfg.json')
    try:
        if kind == 'C' and args.mode == 'replay' and not warm:
            # Retain explicitly requested old constant-current replay as an archive API.
            from scripts.AfferentBalance2026.experiments.C_setpoint_balance.run_experiment import run
            from scripts.AfferentBalance2026.experiments.core.genome import GenomeConfig
            if any(getattr(args, key) is not None for key in ('population', 'generations', 'table_generations', 'mutation_probability')):
                raise ValueError('Training flags are inapplicable to --mode replay')
            if args.output is None:
                raise ValueError('Legacy replay requires explicit --output; prefer run.py replay --solution')
            result = run(GenomeConfig(config), args.output, genome=args.genome,
                         targets=args.targets or [0.], angles=args.angles or [30.], speeds=args.speeds or [0.],
                         currents=args.currents or [0., 0.], current_law=args.current_law, duration_s=args.duration)
        else:
            if kind == 'C' and not warm and any(getattr(args, name) is not None for name in ('genome', 'targets', 'angles', 'speeds', 'currents', 'current_law')):
                raise ValueError('Replay flags are inapplicable to training; use configuration banks')
            updates = {'search':{}, 'runtime':{}, 'storage':{}, 'initialization':{}}
            for key in ('population', 'seed', 'mutation_probability', 'sigma_start', 'sigma_end', 'validation_every', 'polarity_seeds', 'table_generations'):
                value = getattr(args, key)
                if value is not None:
                    updates['search'][key] = value
            if args.workers is not None:
                updates['runtime']['workers'] = args.workers
            if args.output is not None:
                updates['storage']['output'] = str(args.output)
            if args.duration is not None:
                updates['protocol'] = {('trajectory' if kind == 'D' else 'balance'):{'duration_s':args.duration}}
            if args.degree is not None:
                updates['genome'] = {'command':{'degree':args.degree}}
            if args.seed_genome and args.from_scratch:
                raise ValueError('--from-scratch conflicts with --seed-genome')
            if args.seed_genome:
                updates['initialization']['solution'] = str(args.seed_genome)
            elif kind == 'C' and not warm and not args.from_scratch:
                historical_seed = read_json(config).get('setpoint', {}).get('seed_genome')
                if historical_seed:
                    seed_path = (Path(config).resolve().parent/historical_seed).resolve()
                    updates['initialization']['solution'] = str(seed_path)
                    print(f'Compatibility: C seed_genome resolved explicitly to {seed_path}; '
                          'use --from-scratch to select the configured center.', file=sys.stderr)
            if args.extra_seed:
                updates['initialization']['extra_solutions'] = list(map(str, args.extra_seed))
            if args.sources:
                updates['initialization']['sources'] = list(map(str, args.sources))
            method = args.method or args.mode
            if method and method != 'compare':
                updates['search']['method'] = method
            if kind == 'D' and method in ('joint', 'command_only'):
                updates.setdefault('protocol', {})['selection'] = 'initial_checkpoint_validation_v1'
            action = 'finetune' if warm else 'sweep' if sweep and kind != 'D' else 'train'
            if args.generations is not None and not isinstance(args.generations, list):
                updates['search']['generations'] = args.generations
            if sweep and kind != 'D':
                old = read_json(config).get('sweep', {})
                updates['sweep'] = {'axes':dict(population=args.populations or ([args.population] if args.population else old.get('populations', [50])),
                    mutation_probability=args.probabilities or ([args.mutation_probability] if args.mutation_probability is not None else old.get('mutation_probabilities', [.3])),
                    seed=args.seeds or ([args.seed] if args.seed is not None else old.get('seeds', [42])))}
                if kind in 'AB':
                    checkpoints = args.generations or old.get('generations', [30,60,100,150,200])
                    updates['search'].update(generations=max(checkpoints), checkpoints=checkpoints)
                    print('Compatibility: --generations are checkpoints within one continuous GA line.', file=sys.stderr)
                else:
                    updates['sweep']['axes'].update(degree=args.degrees or ([args.degree] if args.degree else old.get('degrees', [2,3])),
                        method=old.get('methods', ['joint','staged']) if method == 'compare' else [method or 'joint'])
            effective = resolve_config(config, overrides=updates, action=action, solution=args.seed_genome if warm else None)
            if effective['experiment'] != kind:
                raise ValueError(f'--experiment {kind} conflicts with the source/configuration task {effective["experiment"]}')
            for message in effective['provenance']['compatibility_messages']:
                print('Compatibility: '+message, file=sys.stderr)
            result = execute_sweep(effective) if action == 'sweep' else execute(effective)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
