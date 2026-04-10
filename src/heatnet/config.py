"""Simple JSON config helpers for HeatNet commands."""

from __future__ import annotations

import json
from pathlib import Path


def load_json_config(path):
    if not path:
        return {}

    config_path = Path(path)
    with open(config_path, "r") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")

    return data


def get_value(cli_value, config, key, default=None):
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default
