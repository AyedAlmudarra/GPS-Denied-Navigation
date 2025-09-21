import cv2
import numpy as np
from collections import deque
import logging

class VPSTransformer:
    """
    Enhanced Visual Place Recognition using SIFT (or optionally deep features),
    with temporal smoothing, confidence modeling, and altitude fusion.
    """

    def __init__(self, map_path, nfeatures=500, min_matches=5, focal_px=525.0, smoothing_window=5, pixels_per_meter=None, camera_K=None, camera_dist=None, yaw_sign=1, scale_levels=None, tile_rows=0, tile_cols=0, features_per_tile=0, ratio_thresh=0.7):
        # Load grayscale map image
        img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load map image from {map_path}")

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)

        # Detect keypoints and descriptors in map image
        self.kp_map, self.des_map = self.sift.detectAndCompute(img, None)
        if self.des_map is None or len(self.kp_map) < min_matches:
            raise ValueError("Not enough features in map image.")

        self.min_matches = min_matches
        self.focal = focal_px
        self.map_shape = img.shape  # (height, width)
        self.pixels_per_meter = pixels_per_meter

        # Camera intrinsics for undistortion (optional)
        self.K = camera_K
        self.dist = camera_dist
        self._ud_shape = None
        self._ud_map1 = None
        self._ud_map2 = None

        # Yaw sign config (+1 or -1)
        self.yaw_sign = yaw_sign

        # For temporal smoothing
        self.pose_buffer = deque(maxlen=smoothing_window)  # Stores dicts with pose and confidence

        # FLANN matcher setup
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )

        # Coarse-to-fine and tiling settings
        self.scale_levels = scale_levels if scale_levels is not None else [1.0]
        self.tile_rows = int(tile_rows)
        self.tile_cols = int(tile_cols)
        self.features_per_tile = int(features_per_tile)
        self.ratio_thresh = float(ratio_thresh)
        # Early-exit score threshold (optional). If >0, stop searching pyramid once exceeded.
        self.early_score_threshold = float(0.0)

        logging.info(f"[VPS INIT] Loaded map {map_path} with {len(self.kp_map)} keypoints")

    def _ensure_ud_maps(self, img):
        if self.K is None or self.dist is None:
            return False
        h, w = img.shape[:2]
        if self._ud_shape == (h, w) and self._ud_map1 is not None and self._ud_map2 is not None:
            return True
        newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 0)
        self._ud_map1, self._ud_map2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, newK, (w, h), cv2.CV_16SC2)
        self._ud_shape = (h, w)
        return True

    def preprocess_frame(self, frame):
        """Convert to grayscale and apply histogram equalization to normalize illumination."""
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        if self._ensure_ud_maps(equalized):
            equalized = cv2.remap(equalized, self._ud_map1, self._ud_map2, interpolation=cv2.INTER_LINEAR)
        return equalized

    def _build_pyramid(self, img, scales):
        pyr = []
        for s in scales:
            if abs(s - 1.0) < 1e-6:
                pyr.append(img)
            else:
                new_size = (int(img.shape[1] * s), int(img.shape[0] * s))
                if new_size[0] <= 10 or new_size[1] <= 10:
                    continue
                pyr.append(cv2.resize(img, new_size, interpolation=cv2.INTER_AREA))
        return pyr

    def _resize_mask(self, mask, target_shape):
        if mask is None:
            return None
        return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    def _detect_tiled(self, img, mask):
        # If no tiling configured, run once
        if self.tile_rows <= 0 or self.tile_cols <= 0 or self.features_per_tile <= 0:
            kp, des = self.sift.detectAndCompute(img, mask)
            return kp, des
        h, w = img.shape[:2]
        cell_h = h // self.tile_rows
        cell_w = w // self.tile_cols
        all_kp = []
        all_desc = []
        for r in range(self.tile_rows):
            for c in range(self.tile_cols):
                y0 = r * cell_h
                x0 = c * cell_w
                y1 = h if r == self.tile_rows - 1 else (y0 + cell_h)
                x1 = w if c == self.tile_cols - 1 else (x0 + cell_w)
                tile_mask = np.zeros((h, w), dtype=np.uint8)
                tile_mask[y0:y1, x0:x1] = 255
                if mask is not None:
                    tile_mask = cv2.bitwise_and(tile_mask, mask)
                kp_tile, des_tile = self.sift.detectAndCompute(img, tile_mask)
                if kp_tile is None or len(kp_tile) == 0:
                    continue
                # Sort by response
                kp_sorted = sorted(zip(kp_tile, ([] if des_tile is None else list(des_tile))), key=lambda x: x[0].response, reverse=True)
                kp_top = kp_sorted[:self.features_per_tile]
                if len(kp_top) == 0:
                    continue
                kps, descs = zip(*kp_top)
                all_kp.extend(kps)
                if len(descs) > 0 and descs[0] is not None:
                    all_desc.extend(descs)
        if len(all_kp) == 0:
            return [], None
        des = np.array(all_desc) if len(all_desc) > 0 else None
        return all_kp, des

    def match(self, frame, motion_mask=None, altitude=0.0):
        """
        Match current camera frame to map and estimate pose with smoothing and confidence.

        Args:
            frame (np.array): Current camera frame (BGR or grayscale)
            motion_mask (np.array|None): Optional 8-bit mask where 0 marks dynamic regions to ignore
            altitude (float): Current altitude in meters

        Returns:
            dict or None: {'x', 'y', 'z', 'yaw', 'confidence'} or None if no reliable match
        """
        # Preprocess
        base = self.preprocess_frame(frame)

        # Build pyramid
        pyr = self._build_pyramid(base, self.scale_levels)
        mask_pyr = [self._resize_mask(motion_mask, lvl.shape) for lvl in pyr]

        best = None
        best_score = -1.0

        for lvl_img, lvl_mask in zip(pyr, mask_pyr):
            # Detect features (tiled if configured)
            kp_frame, des_frame = self._detect_tiled(lvl_img, lvl_mask)
            if des_frame is None or len(kp_frame) < self.min_matches:
                continue

            # Match descriptors
            matches = self.flann.knnMatch(self.des_map, des_frame, k=2)
            good_matches = [m for m, n in matches if m.distance < self.ratio_thresh * n.distance] if matches else []
            if len(good_matches) < self.min_matches:
                continue

            # Extract matched keypoints
            src_pts = np.float32([self.kp_map[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography with RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None or mask is None:
                continue

            inliers = mask.flatten().astype(bool)
            inlier_ratio = float(np.count_nonzero(inliers)) / float(len(good_matches))

            # Degeneracy checks on the homography (rotation/scaling conditioning)
            try:
                rot_mat = M[0:2, 0:2]
                s = np.linalg.svd(rot_mat, compute_uv=False)
                if s.min() <= 1e-6 or (s.max() / s.min()) > 1000.0:
                    continue
                if not np.isfinite(s).all():
                    continue
                # Must be invertible for center transform
                if abs(np.linalg.det(M)) < 1e-9:
                    continue
            except Exception:
                continue

            # Reprojection error for inliers
            inlier_src = src_pts[inliers]
            inlier_dst = dst_pts[inliers]
            reproj = cv2.perspectiveTransform(inlier_src, M)
            reproj_err = np.linalg.norm(reproj - inlier_dst, axis=2).flatten()
            mean_reproj = float(np.mean(reproj_err)) if reproj_err.size > 0 else 1e6

            # Score combines inlier ratio and reprojection quality
            score = inlier_ratio * np.exp(-mean_reproj / 5.0)
            if score > best_score:
                best_score = score
                best = (M, lvl_img.shape)
            # Early exit if a sufficiently good match is found
            if self.early_score_threshold > 0.0 and score >= self.early_score_threshold:
                break

        if best is None:
            logging.debug("[VPS] No valid homography across pyramid levels.")
            return None

        M, shape_used = best

        # Extract yaw from homography
        rot_mat = M[0:2, 0:2]
        U, _, Vt = np.linalg.svd(rot_mat)
        R_mat = U @ Vt
        yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0]) * float(self.yaw_sign)

        # Invert homography to find frame center in map coords
        invM = np.linalg.inv(M)
        h, w = shape_used
        center_frame = np.array([[[w / 2, h / 2]]], dtype=np.float32)
        center_map = cv2.perspectiveTransform(center_frame, invM).flatten()

        # Map center in pixels
        map_center = np.array([self.map_shape[1] / 2, self.map_shape[0] / 2])

        # Pixel offset from map center
        delta_px = center_map - map_center

        # Convert pixel delta to meters (prefer ppm if available)
        if self.pixels_per_meter and self.pixels_per_meter > 0:
            dx = delta_px[0] / self.pixels_per_meter
            dy = delta_px[1] / self.pixels_per_meter
        else:
            dx = delta_px[0] * altitude / self.focal
            dy = delta_px[1] * altitude / self.focal

        # Confidence from best_score (already weighted)
        confidence = max(0.0, min(1.0, float(best_score)))

        # Save to buffer for smoothing
        self.pose_buffer.append({
            "x": dx,
            "y": dy,
            "z": altitude,
            "yaw": yaw,
            "confidence": confidence
        })

        # Temporal smoothing weighted by confidence
        if len(self.pose_buffer) == 0:
            return None

        sum_conf = sum(p["confidence"] for p in self.pose_buffer)
        if sum_conf < 1e-6:
            return None

        # Circular mean for yaw to avoid wraparound bias
        sum_c = sum(np.cos(p["yaw"]) * p["confidence"] for p in self.pose_buffer)
        sum_s = sum(np.sin(p["yaw"]) * p["confidence"] for p in self.pose_buffer)
        yaw_avg = np.arctan2(sum_s, sum_c)

        smoothed_pose = {
            "x": sum(p["x"] * p["confidence"] for p in self.pose_buffer) / sum_conf,
            "y": sum(p["y"] * p["confidence"] for p in self.pose_buffer) / sum_conf,
            "z": altitude,  # Use latest altitude directly
            "yaw": yaw_avg,
            "confidence": sum_conf / len(self.pose_buffer)
        }

        logging.info(f"[VPS] Smoothed pose: x={smoothed_pose['x']:.2f} m, y={smoothed_pose['y']:.2f} m, yaw={np.degrees(smoothed_pose['yaw']):.2f} deg, confidence={smoothed_pose['confidence']:.2f}")

        return smoothed_pose

