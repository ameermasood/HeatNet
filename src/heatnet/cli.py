"""Thin package-level command dispatcher for HeatNet."""

from __future__ import annotations

import argparse
import sys

from heatnet.commands.evaluate import main as evaluate_main
from heatnet.commands.predict import main as predict_main
from heatnet.commands.prepare_data import main as prepare_data_main
from heatnet.commands.train import main as train_main


def build_parser():
    parser = argparse.ArgumentParser(
        prog="heatnet",
        description="HeatNet command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("prepare-data", help="Prepare datasets and derived assets.")
    subparsers.add_parser("train", help="Train baseline or cross-fusion models.")
    subparsers.add_parser("predict", help="Run inference and save pose predictions.")
    subparsers.add_parser("evaluate", help="Run PnP and ADD evaluation.")

    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command = argv[0]
    remainder = argv[1:]

    if command == "prepare-data":
        return prepare_data_main(remainder)
    if command == "train":
        return train_main(remainder)
    if command == "predict":
        return predict_main(remainder)
    if command == "evaluate":
        return evaluate_main(remainder)

    parser.error(f"Unknown command: {command}")
    return 2
