# Configs

This folder stores example run specifications for the unified task scripts.

These files can now be passed to the task scripts with `--config`.
CLI arguments still override values from the config file.

They are lightweight references so repeated runs have one documented place for:

- input paths
- output paths
- model choices
- training hyperparameters

Examples:

```bash
python3 -m heatnet train --config configs/train.example.json
python3 -m heatnet predict --config configs/predict.example.json
python3 -m heatnet evaluate --config configs/evaluate.example.json
python3 -m heatnet prepare-data --config configs/prepare_data.example.json sample-3d
```

The configs are intended to chain together. For example, `predict.example.json` writes to
`outputs/predictions/...`, and `evaluate.example.json` can consume that same prediction file.
