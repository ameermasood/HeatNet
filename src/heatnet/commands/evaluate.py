"""Evaluate saved 2D keypoint predictions with PnP and ADD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from heatnet.config import get_value
from heatnet.config import load_json_config

DEFAULT_DIAMETER_MAP = {
    "01": 102.09865663,
    "02": 247.50624233,
    "03": 167.35486092,
    "04": 172.49224865,
    "05": 201.40358597,
    "06": 154.54551808,
    "07": 124.26430816,
    "08": 261.47178102,
    "09": 108.99920102,
    "10": 164.62758848,
    "11": 175.88933422,
    "12": 145.54287471,
    "13": 278.07811733,
    "14": 282.60129399,
    "15": 212.35825148,
}
DEFAULT_SYMMETRIC_OBJECTS = {"10", "11"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="heatnet evaluate",
        description="Evaluate saved keypoint predictions with PnP and ADD."
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file. CLI arguments override config values.",
    )
    parser.add_argument("--kp2d-json", help="Predicted 2D keypoints JSON.")
    parser.add_argument("--kp3d-json", help="3D keypoints JSON.")
    parser.add_argument("--gt-json", help="Ground-truth poses JSON.")
    parser.add_argument(
        "--output-json",
        help="Optional path for saving the evaluation summary JSON.",
    )
    parser.add_argument(
        "--diameter-json",
        help="Optional object diameter JSON. Defaults to notebook constants.",
    )
    parser.add_argument(
        "--symmetric-objects",
        nargs="*",
        help="Object ids that should use ADD-S.",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        help="ADD threshold ratio relative to object diameter.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_json_config(args.config)

    from heatnet.evaluation.add import evaluate_pose_estimation
    from heatnet.evaluation.pnp import run_pnp

    kp2d_json = get_value(args.kp2d_json, config, "kp2d_json")
    kp3d_json = get_value(args.kp3d_json, config, "kp3d_json")
    gt_json = get_value(args.gt_json, config, "gt_json")
    output_json = get_value(args.output_json, config, "output_json")
    diameter_json = get_value(args.diameter_json, config, "diameter_json")
    symmetric_objects = get_value(
        args.symmetric_objects,
        config,
        "symmetric_objects",
        sorted(DEFAULT_SYMMETRIC_OBJECTS),
    )
    threshold_ratio = get_value(args.threshold_ratio, config, "threshold_ratio", 0.1)

    missing = [
        key
        for key, value in {
            "kp2d_json": kp2d_json,
            "kp3d_json": kp3d_json,
            "gt_json": gt_json,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    kp2d = load_keypoint_predictions(kp2d_json)
    kp3d = load_json(kp3d_json)
    gt_data = load_json(gt_json)
    diameter_map = load_json(diameter_json) if diameter_json else DEFAULT_DIAMETER_MAP

    pnp_results = {}
    skipped = []

    for image_id, keypoints_2d in kp2d.items():
        result = run_pnp(image_id, keypoints_2d, kp3d)
        if result is None:
            skipped.append(image_id)
            continue

        _, pose_result, inliers = result
        pnp_results[image_id] = (pose_result, inliers)

    accuracy_results, results_distribution, high_error_samples = evaluate_pose_estimation(
        pnp_results=pnp_results,
        kp3d=kp3d,
        gt_data=gt_data,
        diameter_map=diameter_map,
        symmetric_objects=set(symmetric_objects),
        threshold_ratio=threshold_ratio,
    )

    summary = {
        "counts": {
            "predictions": len(kp2d),
            "successful_pnp": len(pnp_results),
            "skipped": len(skipped),
        },
        "accuracy_results": accuracy_results,
        "results_distribution_class": to_builtin(results_distribution),
        "high_error_samples": to_builtin(high_error_samples),
        "skipped_images": skipped,
    }

    print(f"Predictions: {len(kp2d)}")
    print(f"Successful PnP: {len(pnp_results)}")
    print(f"Skipped: {len(skipped)}")
    print("Accuracy by class:")
    for cls in sorted(accuracy_results):
        print(f"  {cls}: {accuracy_results[cls]:.2f}%")

    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(to_builtin(summary), indent=2))
        print(f"Saved evaluation summary to {output_path}")


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def load_keypoint_predictions(path):
    data = load_json(path)
    if isinstance(data, dict) and "keypoints_2d" in data:
        return data["keypoints_2d"]
    return data


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
