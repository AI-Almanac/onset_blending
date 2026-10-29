"""Command line: python -m onset_blending train | forecast | convert-specs.

    python -m onset_blending train configs/india_aifs.yml -o out/india
    python -m onset_blending forecast out/india/trained.blend \\
        --forecast aifs_ensv2=live/ens_2026.nc --forecast aifsv2=live/det_2026.nc \\
        --issue-date 2026-06-09 -o out/india/2026-06-09
"""

from __future__ import annotations

import argparse
import logging
import sys

from .legacy import specs as legacy_specs
from .trained import TrainedBlend
from .config import BlendConfig, ConfigError
from .forecast import IncompatibleForecast
from .training import Resources, train


def _resources(args) -> Resources:
    resources = Resources.default()
    if args.workers:
        resources.workers = args.workers
    return resources


def _train(args) -> int:
    result = train(BlendConfig.from_yaml(args.config), resources=_resources(args))
    out = result.save(args.output, intermediates=args.intermediate or ())
    print(f"Wrote {out}")
    return 0


def _forecast(args) -> int:
    forecasts: dict[str, list[str]] = {}
    for item in args.forecast:
        model, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"--forecast expects MODEL=FILE, got {item!r}")
        forecasts.setdefault(model, []).append(path)
    blend = TrainedBlend.load(args.blend)
    forecast = blend.forecast(forecasts, issue_date=args.issue_date, resources=_resources(args))
    print(f"Wrote {forecast.save(args.output)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m onset_blending", description=__doc__.split("\n\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("train", help="train and evaluate a blend from a config")
    run.add_argument("config")
    run.add_argument("-o", "--output", required=True, help="new directory for the results")
    run.add_argument("--intermediate", action="append", choices=["observed", "combined", "weekly"],
                     help="also save this stage's table (repeatable)")
    run.add_argument("--workers", type=int)
    run.set_defaults(handler=_train)

    apply = commands.add_parser("forecast", help="apply a trained blend to live forecast files")
    apply.add_argument("blend", help="a trained.blend file")
    apply.add_argument("--forecast", action="append", required=True, metavar="MODEL=FILE",
                       help="a live forecast file of a trained model (repeatable)")
    apply.add_argument("--issue-date", help="forecast this issue date only (YYYY-MM-DD)")
    apply.add_argument("-o", "--output", required=True, help="new directory for the forecast")
    apply.add_argument("--workers", type=int)
    apply.set_defaults(handler=_forecast)

    convert = commands.add_parser("convert-specs", help="convert a set of old stage specs", add_help=False)
    convert.set_defaults(handler=None)

    args, rest = parser.parse_known_args(argv)
    if args.command == "convert-specs":
        return legacy_specs.main(rest)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    try:
        return args.handler(args)
    except (ConfigError, IncompatibleForecast) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
