"""ADD and ADD-S evaluation utilities."""

from collections import defaultdict

import numpy as np


def get_gt_pose(img_id, gt_data):
    entry = gt_data[img_id]
    rotation_list, translation_list = entry[0], entry[1]
    rotation = np.array(rotation_list, float).reshape(3, 3)
    translation = np.array(translation_list, float).flatten().reshape(3, 1)
    return rotation, translation


def evaluate_pose_estimation(
    pnp_results,
    kp3d,
    gt_data,
    diameter_map,
    symmetric_objects,
    threshold_ratio=0.1,
):
    correct_by_class = defaultdict(int)
    total_by_class = defaultdict(int)
    results_distribution_class = defaultdict(list)
    high_error_samples = defaultdict(list)

    for img_id, (est, _) in pnp_results.items():
        cls = img_id.split("_")[0]

        try:
            gt_rotation, gt_translation = get_gt_pose(img_id, gt_data)
        except KeyError:
            continue

        pred_rotation = np.array(est["R"], dtype=float)
        pred_translation = np.array(est["t"], dtype=float).reshape(3, 1)

        pts3d = np.array(kp3d[cls], dtype=np.float32)
        gt_transformed = (gt_rotation @ pts3d.T + gt_translation).T
        pred_transformed = (pred_rotation @ pts3d.T + pred_translation).T

        if cls in symmetric_objects:
            add_error = np.mean(
                [
                    np.min(np.linalg.norm(gt_point - pred_transformed, axis=1))
                    for gt_point in gt_transformed
                ]
            )
        else:
            add_error = np.mean(np.linalg.norm(gt_transformed - pred_transformed, axis=1))

        total_by_class[cls] += 1
        results_distribution_class[cls].append(add_error / diameter_map[cls])

        if add_error < threshold_ratio * diameter_map[cls]:
            correct_by_class[cls] += 1
        else:
            high_error_samples[cls].append((img_id, add_error))

    accuracy_results = {
        cls: 100.0 * correct_by_class[cls] / total_by_class[cls]
        for cls in total_by_class
    }

    for cls in high_error_samples:
        high_error_samples[cls].sort(key=lambda x: x[1], reverse=True)

    return accuracy_results, results_distribution_class, high_error_samples
