"""`terralens` command-line entry point; each subcommand delegates to a module's `main`."""

from __future__ import annotations

import argparse
import importlib

from terralens import __version__

COMMANDS = {
    "train": ("terralens.train", "train a land-cover classifier"),
    "evaluate": ("terralens.evaluate", "evaluate or compare checkpoints"),
    "predict": ("terralens.predictor", "classify tiles (optionally with Grad-CAM)"),
    "map": ("terralens.mapping", "create a land-cover map of a large scene"),
    "serve": ("terralens.api.app", "run the REST API and web interface"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="terralens", description="Satellite land-cover classification and mapping."
    )
    parser.add_argument("--version", action="version", version=f"terralens {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, (_, help_text) in COMMANDS.items():
        # add_help=False so `terralens <cmd> --help` reaches the subcommand's own parser.
        sub.add_parser(name, help=help_text, add_help=False)

    args, rest = parser.parse_known_args(argv)
    module = importlib.import_module(COMMANDS[args.command][0])
    module.main(rest)


if __name__ == "__main__":
    main()
