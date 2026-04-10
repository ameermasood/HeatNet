"""Prediction helpers built from the end-to-end notebook logic."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from heatnet.evaluation.keypoints import extract_keypoints_to_original_image_space
from heatnet.evaluation.pnp import run_pnp
from heatnet.models.baseline import KeypointHeatmapNet
from heatnet.models.cross_fusion import CrossFuNet


class RGBPosePredictor:
    uses_depth = False

    def __init__(self, yolo_model_path, kpd_model_path, kp3d_path, num_keypoints=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(yolo_model_path)
        self.kpd = self._load_kpd(kpd_model_path, num_keypoints)
        self.kpd.to(self.device).eval()
        self.kp3d_dict = self._load_3d_points(kp3d_path)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _load_kpd(self, path, num_keypoints):
        model = KeypointHeatmapNet(num_keypoints=num_keypoints)
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model

    def _load_3d_points(self, kp3d_path):
        with open(kp3d_path, "r") as handle:
            return json.load(handle)

    def detect_object(self, image_path):
        return self.yolo(image_path)

    def get_bbox(self, result, index=0):
        box = result[0].boxes[index].xyxy[0].tolist()
        return list(map(int, box))

    def preprocess_image(self, image_path, bbox, size=(256, 256)):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.to_tensor(image)
        x1, y1, x2, y2 = bbox
        crop = image_tensor[:, y1:y2, x1:x2]
        resized = torch.nn.functional.interpolate(
            crop.unsqueeze(0), size=size, mode="bilinear", align_corners=False
        )
        normalized = self.normalize(resized.squeeze(0)).unsqueeze(0).to(self.device)
        return normalized

    def detect_keypoints(self, image_path, bbox):
        image_tensor = self.preprocess_image(image_path, bbox)
        return self.kpd(image_tensor)

    def estimate_pose(self, image_path):
        result = self.detect_object(image_path)
        bbox = self.get_bbox(result)
        heatmaps = self.detect_keypoints(image_path, bbox)
        keypoints_2d = extract_keypoints_to_original_image_space(heatmaps, bbox)
        image_name = Path(image_path).name
        pnp_result = run_pnp(image_name, keypoints_2d.tolist(), self.kp3d_dict)
        if pnp_result is None:
            return None

        _, pose_result, inliers = pnp_result
        return Path(image_path).stem, keypoints_2d.tolist(), pose_result, inliers

    def predict_many(self, image_paths, max_workers=4):
        def process(image_path):
            try:
                return self.estimate_pose(image_path)
            except Exception as exc:
                print(f"[ERROR] {image_path}: {exc}")
                return None

        return _run_parallel(process, image_paths, max_workers=max_workers)


class RGBDPosePredictor(RGBPosePredictor):
    uses_depth = True

    def __init__(
        self,
        yolo_model_path,
        kpd_model_path,
        kp3d_path,
        act_layer=None,
        num_keypoints=50,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(yolo_model_path)
        self.kpd = self._load_kpd(kpd_model_path, act_layer, num_keypoints)
        self.kpd.to(self.device).eval()
        self.kp3d_dict = self._load_3d_points(kp3d_path)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _load_kpd(self, path, act_layer, num_keypoints):
        if act_layer is None:
            act_layer = nn.Mish(inplace=True)
        model = CrossFuNet(num_keypoints=num_keypoints, act_layer=act_layer)
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        return model

    def preprocess_depth_image(self, depth_path, bbox, size=(256, 256)):
        depth_img = Image.open(depth_path)
        depth_np = np.array(depth_img).astype(np.float32)
        depth_np = (depth_np - 500.0) / 1000.0
        depth_np = np.clip(depth_np, 0.0, 1.0)
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)

        x1, y1, x2, y2 = bbox
        depth_crop = depth_tensor[:, y1:y2, x1:x2]
        resized = torch.nn.functional.interpolate(
            depth_crop.unsqueeze(0), size=size, mode="bilinear", align_corners=False
        )
        return resized.to(self.device)

    def detect_keypoints(self, image_path, depth_path, bbox):
        image_tensor = self.preprocess_image(image_path, bbox)
        depth_tensor = self.preprocess_depth_image(depth_path, bbox)
        return self.kpd(image_tensor, depth_tensor)

    def estimate_pose(self, image_path, depth_path):
        result = self.detect_object(image_path)
        bbox = self.get_bbox(result)
        heatmaps = self.detect_keypoints(image_path, depth_path, bbox)
        keypoints_2d = extract_keypoints_to_original_image_space(heatmaps, bbox)
        image_name = Path(image_path).name
        pnp_result = run_pnp(image_name, keypoints_2d.tolist(), self.kp3d_dict)
        if pnp_result is None:
            return None

        _, pose_result, inliers = pnp_result
        return Path(image_path).stem, keypoints_2d.tolist(), pose_result, inliers

    def predict_many(self, image_paths, depth_paths, max_workers=4):
        def process(pair):
            image_path, depth_path = pair
            try:
                return self.estimate_pose(image_path, depth_path)
            except Exception as exc:
                print(f"[ERROR] {image_path}: {exc}")
                return None

        pairs = list(zip(image_paths, depth_paths))
        return _run_parallel(process, pairs, max_workers=max_workers)


def build_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "mish":
        return nn.Mish(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _run_parallel(process_fn, items, max_workers=4):
    pnp_results = {}
    keypoint_results = {}
    skipped = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_fn, item): item for item in items}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                item = futures[future]
                skipped.append(item[0] if isinstance(item, tuple) else item)
                continue
            image_id, keypoints_2d, pose_result, inliers = result
            keypoint_results[image_id] = keypoints_2d
            pnp_results[image_id] = (pose_result, inliers)

    return keypoint_results, pnp_results, skipped
