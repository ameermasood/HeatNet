"""PnP utilities for pose recovery."""

import cv2
import numpy as np

K = np.array(
    [
        [572.4114, 0.0, 325.2611],
        [0.0, 573.57043, 242.04899],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def eliminate_duplicate_pairs(pts2d, pts3d):
    pairs = list(zip(pts3d, pts2d))
    unique_pairs = []
    seen_2d = set()

    for p3d, p2d in pairs:
        key = tuple(np.round(p2d))

        if key in seen_2d:
            continue

        seen_2d.add(key)
        unique_pairs.append((p3d, p2d))

    if len(unique_pairs) < 4:
        print("[WARNING] Not enough unique 2D keypoints after deduplication")
        return None

    pts3d_clean = np.array([p3d for p3d, _ in unique_pairs], dtype=np.float64)
    pts2d_clean = np.array([p2d for _, p2d in unique_pairs], dtype=np.float64)
    return pts2d_clean, pts3d_clean


def run_pnp(img_id, keypoints_2d, kp3d_dict, camera_matrix=K):
    obj_key = img_id.split("_")[0]

    pts3d = np.array(kp3d_dict[obj_key])
    pts2d = np.array(keypoints_2d, np.float64)

    cleaned = eliminate_duplicate_pairs(pts2d, pts3d)
    if cleaned is None:
        return None

    pts2d, pts3d = cleaned

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=pts3d,
        imagePoints=pts2d,
        cameraMatrix=camera_matrix,
        distCoeffs=None,
        reprojectionError=4,
        confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return obj_key, {"R": rotation_matrix.tolist(), "t": tvec.flatten().tolist()}, inliers
