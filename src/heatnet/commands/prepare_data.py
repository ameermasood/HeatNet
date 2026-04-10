"""Unified data preparation entrypoint for HeatNet."""

from __future__ import annotations

import argparse

from heatnet.config import get_value
from heatnet.config import load_json_config


def build_parser():
    parser = argparse.ArgumentParser(
        prog="heatnet prepare-data",
        description="Prepare HeatNet data assets.",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file. The selected subcommand can read values from it.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    full_data = subparsers.add_parser(
        "full-data",
        help="Build full_data train/test splits from raw LineMOD data.",
    )
    full_data.add_argument("--raw-data-dir")
    full_data.add_argument("--output-dir")
    full_data.add_argument("--test-size", type=float, default=0.1)
    full_data.add_argument("--seed", type=int, default=42)

    yolo = subparsers.add_parser(
        "yolo-data",
        help="Prepare YOLO train/val data from full_data/train.",
    )
    yolo.add_argument("--train-image-dir")
    yolo.add_argument("--train-gt-json")
    yolo.add_argument("--output-dir")
    yolo.add_argument("--val-size", type=float, default=0.2)
    yolo.add_argument("--seed", type=int, default=8)

    crop_rgb = subparsers.add_parser(
        "crop-rgb",
        help="Crop and resize RGB images from YOLO labels.",
    )
    crop_rgb.add_argument("--label-dir")
    crop_rgb.add_argument("--image-dir")
    crop_rgb.add_argument("--output-dir")
    crop_rgb.add_argument("--max-workers", type=int, default=8)

    crop_depth = subparsers.add_parser(
        "crop-depth",
        help="Crop and resize depth images from YOLO labels.",
    )
    crop_depth.add_argument("--label-dir")
    crop_depth.add_argument("--image-dir")
    crop_depth.add_argument("--output-dir")
    crop_depth.add_argument("--max-workers", type=int, default=8)

    bbox_predict = subparsers.add_parser(
        "bbox-predict",
        help="Generate YOLO bbox label predictions for a directory of images.",
    )
    bbox_predict.add_argument("--model-path")
    bbox_predict.add_argument("--image-dir")
    bbox_predict.add_argument("--output-dir")
    bbox_predict.add_argument("--conf-threshold", type=float, default=0.25)
    bbox_predict.add_argument("--image-size", type=int, default=640)
    bbox_predict.add_argument("--max-det", type=int, default=1)

    sample = subparsers.add_parser(
        "sample-3d",
        help="Sample 3D keypoints from CAD models.",
    )
    sample.add_argument("--cad-model-dir")
    sample.add_argument("--output-json")
    sample.add_argument("--method", choices=["fps", "cps"])
    sample.add_argument("--k-points", type=int, default=50)
    sample.add_argument(
        "--skip-classes",
        nargs="*",
        default=["obj_03.ply", "obj_07.ply", "models_info.yml"],
    )

    project = subparsers.add_parser(
        "project-2d",
        help="Project 3D keypoints into YOLO crop space.",
    )
    project.add_argument("--gt-json")
    project.add_argument("--kp3d-json")
    project.add_argument("--yolo-label-dir")
    project.add_argument("--output-json")

    heatmaps = subparsers.add_parser(
        "heatmaps",
        help="Generate Gaussian heatmaps from 2D keypoint labels.",
    )
    heatmaps.add_argument("--input-json")
    heatmaps.add_argument("--output-dir")
    heatmaps.add_argument("--sigma", type=float, default=2.0)

    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_json_config(args.config)

    from heatnet.data.preparation import create_full_data_split
    from heatnet.data.preparation import crop_from_yolo_labels
    from heatnet.data.preparation import generate_yolo_bbox_predictions
    from heatnet.data.preparation import generate_heatmaps_from_keypoints
    from heatnet.data.preparation import prepare_yolo_data
    from heatnet.data.preparation import project_keypoints_yolo_scaled
    from heatnet.data.preparation import sample_3d_keypoints

    command_config_key = args.command.replace("-", "_")
    command_config = config.get(command_config_key, {})
    if command_config and not isinstance(command_config, dict):
        raise ValueError(
            f"Config section '{command_config_key}' must be a JSON object."
        )

    if args.command == "full-data":
        raw_data_dir = get_value(args.raw_data_dir, command_config, "raw_data_dir")
        output_dir = get_value(args.output_dir, command_config, "output_dir")
        _require_values(raw_data_dir=raw_data_dir, output_dir=output_dir)
        create_full_data_split(
            raw_data_dir=raw_data_dir,
            output_dir=output_dir,
            test_size=get_value(args.test_size, command_config, "test_size", 0.1),
            seed=get_value(args.seed, command_config, "seed", 42),
        )
    elif args.command == "yolo-data":
        train_image_dir = get_value(
            args.train_image_dir, command_config, "train_image_dir"
        )
        train_gt_json = get_value(args.train_gt_json, command_config, "train_gt_json")
        output_dir = get_value(args.output_dir, command_config, "output_dir")
        _require_values(
            train_image_dir=train_image_dir,
            train_gt_json=train_gt_json,
            output_dir=output_dir,
        )
        prepare_yolo_data(
            train_image_dir=train_image_dir,
            train_gt_json=train_gt_json,
            output_dir=output_dir,
            val_size=get_value(args.val_size, command_config, "val_size", 0.2),
            seed=get_value(args.seed, command_config, "seed", 8),
        )
    elif args.command == "crop-rgb":
        label_dir = get_value(args.label_dir, command_config, "label_dir")
        image_dir = get_value(args.image_dir, command_config, "image_dir")
        output_dir = get_value(args.output_dir, command_config, "output_dir")
        _require_values(label_dir=label_dir, image_dir=image_dir, output_dir=output_dir)
        crop_from_yolo_labels(
            label_dir=label_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            is_depth=False,
            max_workers=get_value(args.max_workers, command_config, "max_workers", 8),
        )
    elif args.command == "crop-depth":
        label_dir = get_value(args.label_dir, command_config, "label_dir")
        image_dir = get_value(args.image_dir, command_config, "image_dir")
        output_dir = get_value(args.output_dir, command_config, "output_dir")
        _require_values(label_dir=label_dir, image_dir=image_dir, output_dir=output_dir)
        crop_from_yolo_labels(
            label_dir=label_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            is_depth=True,
            max_workers=get_value(args.max_workers, command_config, "max_workers", 8),
        )
    elif args.command == "bbox-predict":
        model_path = get_value(args.model_path, command_config, "model_path")
        image_dir = get_value(args.image_dir, command_config, "image_dir")
        output_dir = get_value(args.output_dir, command_config, "output_dir")
        _require_values(model_path=model_path, image_dir=image_dir, output_dir=output_dir)
        generate_yolo_bbox_predictions(
            model_path=model_path,
            image_dir=image_dir,
            output_dir=output_dir,
            conf_threshold=get_value(
                args.conf_threshold, command_config, "conf_threshold", 0.25
            ),
            image_size=get_value(args.image_size, command_config, "image_size", 640),
            max_det=get_value(args.max_det, command_config, "max_det", 1),
        )
    elif args.command == "sample-3d":
        cad_model_dir = get_value(args.cad_model_dir, command_config, "cad_model_dir")
        output_json = get_value(args.output_json, command_config, "output_json")
        method = get_value(args.method, command_config, "method")
        _require_values(
            cad_model_dir=cad_model_dir,
            output_json=output_json,
            method=method,
        )
        sample_3d_keypoints(
            cad_model_dir=cad_model_dir,
            output_json_path=output_json,
            method=method,
            skip_classes=get_value(
                args.skip_classes,
                command_config,
                "skip_classes",
                ["obj_03.ply", "obj_07.ply", "models_info.yml"],
            ),
            k_points=get_value(args.k_points, command_config, "k_points", 50),
        )
    elif args.command == "project-2d":
        gt_json = get_value(args.gt_json, command_config, "gt_json")
        kp3d_json = get_value(args.kp3d_json, command_config, "kp3d_json")
        yolo_label_dir = get_value(
            args.yolo_label_dir, command_config, "yolo_label_dir"
        )
        output_json = get_value(args.output_json, command_config, "output_json")
        _require_values(
            gt_json=gt_json,
            kp3d_json=kp3d_json,
            yolo_label_dir=yolo_label_dir,
            output_json=output_json,
        )
        project_keypoints_yolo_scaled(
            gt_path=gt_json,
            keypoint_json_path=kp3d_json,
            yolo_label_dir=yolo_label_dir,
            output_json_path=output_json,
        )
    elif args.command == "heatmaps":
        input_json = get_value(args.input_json, command_config, "input_json")
        output_dir = get_value(args.output_dir, command_config, "output_dir")
        _require_values(input_json=input_json, output_dir=output_dir)
        generate_heatmaps_from_keypoints(
            input_json_path=input_json,
            output_dir=output_dir,
            sigma=get_value(args.sigma, command_config, "sigma", 2.0),
        )


def _require_values(**kwargs):
    missing = [key for key, value in kwargs.items() if value is None]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
