"""Run HeatNet inference and save predicted poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from heatnet.config import get_value
from heatnet.config import load_json_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="heatnet predict",
        description="Run HeatNet pose prediction.",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file. CLI arguments override config values.",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "cross_fusion"],
        help="Which predictor family to use.",
    )
    parser.add_argument("--yolo-model", help="YOLO checkpoint path.")
    parser.add_argument("--kpd-model", help="Keypoint model checkpoint path.")
    parser.add_argument("--kp3d-json", help="3D keypoints JSON path.")
    parser.add_argument("--image-dir", help="Directory of input RGB images.")
    parser.add_argument(
        "--depth-dir",
        help="Directory of depth images. Required for cross_fusion.",
    )
    parser.add_argument("--output-json", help="Where to save predicted poses.")
    parser.add_argument(
        "--activation",
        choices=["relu", "silu", "mish"],
        help="Activation for cross_fusion.",
    )
    parser.add_argument("--num-keypoints", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--limit", type=int, help="Optional image limit for quick runs.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_json_config(args.config)

    from heatnet.inference.predictor import RGBDPosePredictor
    from heatnet.inference.predictor import RGBPosePredictor
    from heatnet.inference.predictor import build_activation

    model_name = get_value(args.model, config, "model")
    yolo_model = get_value(args.yolo_model, config, "yolo_model")
    kpd_model = get_value(args.kpd_model, config, "kpd_model")
    kp3d_json = get_value(args.kp3d_json, config, "kp3d_json")
    image_dir = get_value(args.image_dir, config, "image_dir")
    depth_dir = get_value(args.depth_dir, config, "depth_dir")
    output_json = get_value(args.output_json, config, "output_json")
    activation_name = get_value(args.activation, config, "activation", "mish")
    num_keypoints = get_value(args.num_keypoints, config, "num_keypoints", 50)
    max_workers = get_value(args.max_workers, config, "max_workers", 4)
    limit = get_value(args.limit, config, "limit")

    missing = [
        key
        for key, value in {
            "model": model_name,
            "yolo_model": yolo_model,
            "kpd_model": kpd_model,
            "kp3d_json": kp3d_json,
            "image_dir": image_dir,
            "output_json": output_json,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    if model_name == "cross_fusion" and not depth_dir:
        raise ValueError("--depth-dir is required when --model cross_fusion is used.")

    image_paths = sorted(
        str(path)
        for path in Path(image_dir).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if limit:
        image_paths = image_paths[:limit]

    if model_name == "baseline":
        predictor = RGBPosePredictor(
            yolo_model_path=yolo_model,
            kpd_model_path=kpd_model,
            kp3d_path=kp3d_json,
            num_keypoints=num_keypoints,
        )
        keypoints_2d, pnp_results, skipped = predictor.predict_many(
            image_paths=image_paths,
            max_workers=max_workers,
        )
    else:
        depth_paths = [str(Path(depth_dir) / Path(path).name) for path in image_paths]
        predictor = RGBDPosePredictor(
            yolo_model_path=yolo_model,
            kpd_model_path=kpd_model,
            kp3d_path=kp3d_json,
            act_layer=build_activation(activation_name),
            num_keypoints=num_keypoints,
        )
        keypoints_2d, pnp_results, skipped = predictor.predict_many(
            image_paths=image_paths,
            depth_paths=depth_paths,
            max_workers=max_workers,
        )

    summary = {
        "counts": {
            "images": len(image_paths),
            "successful_pnp": len(pnp_results),
            "skipped": len(skipped),
        },
        "keypoints_2d": to_builtin(keypoints_2d),
        "poses": to_builtin(pnp_results),
        "skipped_images": skipped,
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    print(f"Processed images: {len(image_paths)}")
    print(f"Successful PnP: {len(pnp_results)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Saved predictions to {output_path}")


def to_builtin(value):
    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None

    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value
