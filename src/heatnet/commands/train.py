"""Train baseline or cross-fusion models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from heatnet.config import get_value
from heatnet.config import load_json_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="heatnet train",
        description="Train HeatNet models.",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file. CLI arguments override config values.",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "cross_fusion"],
        help="Which model family to train.",
    )
    parser.add_argument("--image-dir", help="Directory of cropped RGB images.")
    parser.add_argument("--heatmap-dir", help="Directory of heatmap .pt files.")
    parser.add_argument(
        "--depth-dir",
        help="Directory of cropped depth images. Required for cross_fusion.",
    )
    parser.add_argument("--save-path", help="Path to save the best checkpoint.")
    parser.add_argument(
        "--history-path",
        help="Optional JSON path to save loss history.",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "silu", "mish"],
        help="Activation for the cross_fusion model.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "onecycle", "polynomial"],
        help="Learning rate scheduler.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-keypoints", type=int)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_json_config(args.config)

    import torch
    import torch.nn as nn

    from heatnet.data.datasets import RGBDHeatmapDataset
    from heatnet.data.datasets import RGBHeatmapDataset
    from heatnet.data.datasets import make_loaders
    from heatnet.models.baseline import KeypointHeatmapNet
    from heatnet.models.cross_fusion import CrossFuNet
    from heatnet.training.losses import focal_heatmap_loss
    from heatnet.training.trainer import build_scheduler
    from heatnet.training.trainer import train_model

    model_name = get_value(args.model, config, "model")
    image_dir = get_value(args.image_dir, config, "image_dir")
    heatmap_dir = get_value(args.heatmap_dir, config, "heatmap_dir")
    depth_dir = get_value(args.depth_dir, config, "depth_dir")
    save_path_arg = get_value(args.save_path, config, "save_path")
    history_path_arg = get_value(args.history_path, config, "history_path")
    activation_name = get_value(args.activation, config, "activation", "mish")
    scheduler_name = get_value(args.scheduler, config, "scheduler", "none")
    epochs = get_value(args.epochs, config, "epochs", 30)
    batch_size = get_value(args.batch_size, config, "batch_size", 16)
    learning_rate = get_value(args.learning_rate, config, "learning_rate", 1e-4)
    weight_decay = get_value(args.weight_decay, config, "weight_decay", 1e-5)
    num_keypoints = get_value(args.num_keypoints, config, "num_keypoints", 50)
    train_ratio = get_value(args.train_ratio, config, "train_ratio", 0.8)
    split_seed = get_value(args.split_seed, config, "split_seed", 8)
    num_workers = get_value(args.num_workers, config, "num_workers", 4)

    missing = [
        key
        for key, value in {
            "model": model_name,
            "image_dir": image_dir,
            "heatmap_dir": heatmap_dir,
            "save_path": save_path_arg,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    if model_name == "cross_fusion" and not depth_dir:
        raise ValueError("--depth-dir is required when --model cross_fusion is used.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "baseline":
        dataset = RGBHeatmapDataset(
            image_dir=image_dir,
            heatmap_dir=heatmap_dir,
        )
        model = KeypointHeatmapNet(num_keypoints=num_keypoints).to(device)
        uses_depth = False
    else:
        activation = build_activation(nn, activation_name)
        dataset = RGBDHeatmapDataset(
            image_dir=image_dir,
            depth_dir=depth_dir,
            heatmap_dir=heatmap_dir,
        )
        model = CrossFuNet(
            num_keypoints=num_keypoints,
            act_layer=activation,
        ).to(device)
        uses_depth = True

    train_loader, val_loader = make_loaders(
        dataset,
        batch_size=batch_size,
        train_ratio=train_ratio,
        seed=split_seed,
        num_workers=num_workers,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        epochs=epochs,
        train_loader=train_loader,
        base_lr=learning_rate,
        max_lr=max(learning_rate, learning_rate * 5),
    )

    save_path = Path(save_path_arg)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=focal_heatmap_loss,
        device=device,
        epochs=epochs,
        save_path=str(save_path),
        scheduler=scheduler,
        uses_depth=uses_depth,
    )

    if history_path_arg:
        history_path = Path(history_path_arg)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2))
        print(f"Saved history to {history_path}")


def build_activation(nn_module, name):
    if name == "relu":
        return nn_module.ReLU(inplace=True)
    if name == "silu":
        return nn_module.SiLU(inplace=True)
    if name == "mish":
        return nn_module.Mish(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")
