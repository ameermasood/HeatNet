"""Unified data preparation helpers derived from the project notebooks."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DEFAULT_CAMERA_MATRIX = np.array(
    [
        [572.4114, 0.0, 325.2611],
        [0.0, 573.57043, 242.04899],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DEFAULT_CLASS_NAMES = [
    "ape",
    "benchvise",
    "camera",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
    "iron",
    "lamp",
    "phone",
]


def get_valid_object_ids(exclude_ids=("03", "07")):
    return [f"{index:02d}" for index in range(1, 16) if f"{index:02d}" not in exclude_ids]


def get_yolo_class_map(valid_object_ids=None):
    valid_object_ids = valid_object_ids or get_valid_object_ids()
    return {object_id: idx for idx, object_id in enumerate(valid_object_ids)}


def create_full_data_split(raw_data_dir, output_dir, exclude_ids=("03", "07"), test_size=0.1, seed=42):
    import yaml
    from sklearn.model_selection import train_test_split

    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    valid_ids = get_valid_object_ids(exclude_ids)

    images_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    train_images_dir = output_dir / "train" / "images"
    train_depth_dir = output_dir / "train" / "depth"
    test_images_dir = output_dir / "test" / "images"
    test_depth_dir = output_dir / "test" / "depth"
    for path in [
        images_dir,
        depth_dir,
        train_images_dir,
        train_depth_dir,
        test_images_dir,
        test_depth_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    gt_data = {}
    all_image_names = []
    labels = []

    for object_id in valid_ids:
        object_root = raw_data_dir / object_id
        rgb_dir = object_root / "rgb"
        object_depth_dir = object_root / "depth"
        gt_path = object_root / "gt.yml"

        with open(gt_path, "r") as handle:
            gt_yml = yaml.safe_load(handle)

        for filename in sorted(os.listdir(rgb_dir)):
            if not filename.endswith(".png"):
                continue

            stem = Path(filename).stem
            combined_name = f"{object_id}_{filename}"
            shutil.copy2(rgb_dir / filename, images_dir / combined_name)
            shutil.copy2(object_depth_dir / filename, depth_dir / combined_name)
            all_image_names.append(combined_name)
            labels.append(object_id)

            frame_key = int(stem)
            if frame_key not in gt_yml:
                continue
            entry = gt_yml[frame_key][0]
            gt_data[f"{object_id}_{stem}"] = [
                entry["cam_R_m2c"],
                entry["cam_t_m2c"],
                entry["obj_bb"],
            ]

    with open(output_dir / "gt.json", "w") as handle:
        json.dump(gt_data, handle, indent=2)

    train_images, test_images = train_test_split(
        all_image_names,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    _move_named_files(train_images, images_dir, train_images_dir)
    _move_named_files(train_images, depth_dir, train_depth_dir)
    _move_named_files(test_images, images_dir, test_images_dir)
    _move_named_files(test_images, depth_dir, test_depth_dir)

    _write_split_gt(gt_data, train_images, output_dir / "train" / "gt.json")
    _write_split_gt(gt_data, test_images, output_dir / "test" / "gt.json")


def prepare_yolo_data(
    train_image_dir,
    train_gt_json,
    output_dir,
    val_size=0.2,
    seed=8,
    class_map=None,
    class_names=None,
):
    import cv2
    import yaml
    from sklearn.model_selection import train_test_split

    train_image_dir = Path(train_image_dir)
    output_dir = Path(output_dir)
    class_map = class_map or get_yolo_class_map()
    class_names = class_names or DEFAULT_CLASS_NAMES

    with open(train_gt_json, "r") as handle:
        gt_data = json.load(handle)

    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    image_names = sorted(path.name for path in train_image_dir.glob("*.png"))
    available = [name for name in image_names if name.replace(".png", "") in gt_data]
    train_imgs, val_imgs = train_test_split(available, test_size=val_size, random_state=seed)

    for split, split_images in [("train", train_imgs), ("val", val_imgs)]:
        for name in split_images:
            key = name.replace(".png", "")
            bbox = gt_data[key][2]
            object_id = key.split("_")[0]
            if object_id not in class_map:
                continue

            image_path = train_image_dir / name
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]

            x_center, y_center, norm_w, norm_h = _convert_bbox_to_yolo(bbox, width, height)
            label_path = output_dir / "labels" / split / name.replace(".png", ".txt")
            with open(label_path, "w") as handle:
                handle.write(
                    f"{class_map[object_id]} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                )

            shutil.copyfile(image_path, output_dir / "images" / split / name)

    data_yaml = {
        "train": str(output_dir / "images" / "train"),
        "val": str(output_dir / "images" / "val"),
        "nc": len(class_names),
        "names": class_names,
    }
    with open(output_dir / "data.yaml", "w") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)


def crop_from_yolo_labels(
    label_dir,
    image_dir,
    output_dir,
    image_width=640,
    image_height=480,
    resize_shape=(256, 256),
    is_depth=False,
    max_workers=8,
):
    import cv2

    label_dir = Path(label_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(path for path in label_dir.glob("*.txt"))

    def process(label_path):
        image_key = label_path.stem
        image_path = image_dir / f"{image_key}.png"
        try:
            line = label_path.read_text().strip()
            if not line:
                return ("skip", image_key, "empty label")

            _, xc, yc, width, height = map(float, line.split())
            x = (xc - width / 2) * image_width
            y = (yc - height / 2) * image_height
            width *= image_width
            height *= image_height

            if not _is_valid_bbox(x, y, width, height, image_width, image_height):
                return ("skip", image_key, "invalid bbox")

            read_flag = cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR
            image = cv2.imread(str(image_path), read_flag)
            if image is None:
                return ("skip", image_key, "image not found")

            x1 = int(max(0, x))
            y1 = int(max(0, y))
            x2 = int(min(image_width, x + width))
            y2 = int(min(image_height, y + height))

            cropped = image[y1:y2, x1:x2]
            interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
            resized = cv2.resize(cropped, resize_shape, interpolation=interpolation)
            success = cv2.imwrite(str(output_dir / f"{image_key}.png"), resized)
            return ("saved", image_key, None) if success else ("skip", image_key, "failed to save")
        except Exception as exc:
            return ("skip", image_key, str(exc))

    return _run_parallel(process, label_files, output_dir, "skipped_depth.txt" if is_depth else "skipped_images.txt", max_workers=max_workers)


def generate_yolo_bbox_predictions(
    model_path,
    image_dir,
    output_dir,
    conf_threshold=0.25,
    image_size=640,
    max_det=1,
):
    from ultralytics import YOLO

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    saved = 0
    skipped = []

    for image_path in image_paths:
        try:
            results = model(
                str(image_path),
                conf=conf_threshold,
                imgsz=image_size,
                max_det=max_det,
                verbose=False,
            )

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                skipped.append((image_path.name, "no detections"))
                continue

            box = result.boxes[0]
            cls_id = int(box.cls.item())
            x_center, y_center, width, height = box.xywhn[0].tolist()

            label_path = output_dir / f"{image_path.stem}.txt"
            with open(label_path, "w") as handle:
                handle.write(
                    f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )
            saved += 1
        except Exception as exc:
            skipped.append((image_path.name, str(exc)))

    with open(output_dir / "skipped_detections.txt", "w") as handle:
        for image_name, reason in skipped:
            handle.write(f"{image_name}: {reason}\n")

    return {"saved": saved, "skipped": len(skipped)}


def sample_3d_keypoints(cad_model_dir, output_json_path, method="fps", skip_classes=None, k_points=50, curvature_k=30):
    import open3d as o3d

    cad_model_dir = Path(cad_model_dir)
    output_json_path = Path(output_json_path)
    skip_classes = set(skip_classes or [])
    keypoints_per_class = {}

    for model_file in sorted(cad_model_dir.iterdir()):
        if model_file.suffix != ".ply" or model_file.name in skip_classes:
            continue

        class_name = model_file.stem.replace("obj_", "")
        point_cloud = o3d.io.read_point_cloud(str(model_file))
        points = np.asarray(point_cloud.points)

        if method == "fps":
            keypoints = _fps_sampling(points, k_points)
        elif method == "cps":
            curvatures = _estimate_curvature(points, curvature_k)
            keypoints = _cps_sampling(points, curvatures, k_points)
        else:
            raise ValueError(f"Unsupported sampling method: {method}")

        keypoints_per_class[class_name] = keypoints.tolist()

    with open(output_json_path, "w") as handle:
        json.dump(keypoints_per_class, handle, indent=4)


def project_keypoints_yolo_scaled(
    gt_path,
    keypoint_json_path,
    yolo_label_dir,
    output_json_path,
    camera_matrix=None,
    image_size=(256, 256),
):
    import cv2

    camera_matrix = np.array(camera_matrix) if camera_matrix is not None else DEFAULT_CAMERA_MATRIX

    with open(gt_path, "r") as handle:
        gt_data = json.load(handle)
    with open(keypoint_json_path, "r") as handle:
        keypoints_3d = json.load(handle)

    all_keypoints_2d = {}

    for image_key, (rotation_list, translation, _) in gt_data.items():
        object_id = image_key.split("_")[0]
        if object_id not in keypoints_3d:
            continue

        label_path = Path(yolo_label_dir) / f"{image_key}.txt"
        if not label_path.exists():
            continue

        line = label_path.read_text().strip()
        if not line:
            continue
        _, xc, yc, width, height = map(float, line.split())
        x = (xc - width / 2) * 640
        y = (yc - height / 2) * 480
        width *= 640
        height *= 480

        object_points = np.array(keypoints_3d[object_id], dtype=np.float32)
        rotation = np.array(rotation_list).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(rotation)
        translation = np.array(translation).reshape(3, 1)
        keypoints_2d, _ = cv2.projectPoints(object_points, rvec, translation, camera_matrix, None)
        keypoints_2d = keypoints_2d.squeeze()

        scale_x = image_size[0] / width
        scale_y = image_size[1] / height
        scaled = [[float((u - x) * scale_x), float((v - y) * scale_y)] for u, v in keypoints_2d]
        all_keypoints_2d[image_key] = scaled

    with open(output_json_path, "w") as handle:
        json.dump(all_keypoints_2d, handle, indent=2)


def generate_heatmaps_from_keypoints(
    input_json_path,
    output_dir,
    image_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2.0,
):
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(input_json_path, "r") as handle:
        keypoints_2d = json.load(handle)

    for image_id, coords in keypoints_2d.items():
        if not coords:
            continue

        scaled_keypoints = _scale_coords(coords, orig_size=image_size, target_size=heatmap_size)
        heatmaps = torch.stack(
            [
                _generate_heatmap(keypoint, shape=heatmap_size, sigma=sigma)
                for keypoint in scaled_keypoints
            ]
        )
        torch.save(heatmaps, output_dir / f"{image_id}.pt")


def _convert_bbox_to_yolo(bbox, image_width, image_height):
    x, y, width, height = bbox
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    return x_center, y_center, width / image_width, height / image_height


def _move_named_files(names, source_dir, destination_dir):
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        source_path = source_dir / name
        destination_path = destination_dir / name
        if source_path.exists():
            shutil.move(source_path, destination_path)


def _write_split_gt(gt_data, image_names, output_json_path):
    split_gt = {
        Path(name).stem: gt_data[Path(name).stem]
        for name in image_names
        if Path(name).stem in gt_data
    }
    with open(output_json_path, "w") as handle:
        json.dump(split_gt, handle, indent=4)


def _is_valid_bbox(x, y, width, height, image_width=640, image_height=480):
    return (
        width > 0
        and height > 0
        and x >= 0
        and y >= 0
        and x + width <= image_width
        and y + height <= image_height
    )


def _run_parallel(process_fn, items, output_dir, skipped_log_name, max_workers=8):
    saved = 0
    skipped = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_fn, item) for item in items]
        for future in futures:
            status, image_key, reason = future.result()
            if status == "saved":
                saved += 1
            else:
                skipped.append((image_key, reason))

    with open(Path(output_dir) / skipped_log_name, "w") as handle:
        for image_key, reason in skipped:
            handle.write(f"{image_key}: {reason}\n")

    return {"saved": saved, "skipped": len(skipped)}


def _estimate_curvature(points, k=30):
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)

    curvatures = []
    point_array = np.asarray(point_cloud.points)
    for point in point_array:
        _, idx, _ = kd_tree.search_knn_vector_3d(point, k)
        if len(idx) < 3:
            curvatures.append(0.0)
            continue
        neighbors = point_array[idx, :]
        covariance = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        curvature = eigenvalues[0] / (sum(eigenvalues) + 1e-8)
        curvatures.append(curvature)
    return np.array(curvatures)


def _cps_sampling(points, curvatures, k_points):
    curvatures = np.array(curvatures)
    curvatures = (curvatures - curvatures.min()) / (curvatures.max() - curvatures.min() + 1e-8)
    hist, bins = np.histogram(curvatures, bins=256, density=True)
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8)
    equalized = np.interp(curvatures, bins[:-1], cdf)

    selected_indices = []
    weights = equalized
    centroid = np.mean(points, axis=0)
    distance_to_centroid = np.linalg.norm(points - centroid, axis=1)
    score = distance_to_centroid * weights

    for _ in range(k_points):
        score[selected_indices] = -np.inf
        idx = np.argmax(score)
        selected_indices.append(idx)
        reference_point = points[idx]
        distance_to_reference = np.linalg.norm(points - reference_point, axis=1)
        score = np.minimum(score, distance_to_reference * weights)

    return points[selected_indices]


def _fps_sampling(points, k_points):
    sampled = [np.random.randint(len(points))]
    distances = np.full(len(points), np.inf)

    for _ in range(1, k_points):
        last_point = points[sampled[-1]]
        dist = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dist)
        next_idx = np.argmax(distances)
        sampled.append(next_idx)

    return points[sampled]


def _scale_coords(coords, orig_size=(256, 256), target_size=(64, 64)):
    scale_x = target_size[1] / orig_size[1]
    scale_y = target_size[0] / orig_size[0]
    return [(x * scale_x, y * scale_y) for x, y in coords]


def _generate_heatmap(center, shape, sigma):
    import torch

    x_grid = torch.arange(shape[1]).view(1, -1).expand(shape)
    y_grid = torch.arange(shape[0]).view(-1, 1).expand(shape)
    distance_squared = (x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2
    return torch.exp(-distance_squared / (2 * sigma**2))
