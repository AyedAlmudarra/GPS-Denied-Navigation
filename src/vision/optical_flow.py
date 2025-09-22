import cv2
import numpy as np
from collections import deque
import logging

"""
OpticalFlowTracker
------------------
Dense-enough sparse-tracking pipeline for planar translational flow using ORB seeding
and Lucas–Kanade (LK) pyramidal optical flow with robust estimation and gating.

Key ideas
- Image preprocessing: grayscale + histogram equalization; optional undistortion.
- Optional image downscaling for performance; all flow vectors are rescaled back.
- Feature seeding: grid-based ORB seeding (optional) or ROI-based detection.
- Tracking: LK forward flow, optional forward–backward check to reject outliers.
- Robust motion model: affine model estimated with RANSAC; residual-based gating.
- Motion mask: dynamic regions (large residuals) are suppressed for VPS.
- Velocity estimation: median flow → meters using altitude / focal length.
- ZUPT: zero-velocity update when flow magnitude is below adaptive threshold.

Return semantics of update(frame, altitude, dt)
- Returns (vx_m, vy_m, confidence), where vx/vy are planar translational flow magnitudes
  in meters per frame converted to meters per second if dt is provided; internally we
  compute flow in meters and divide by dt only for confidence/velocity smoothing, while
  the function returns the planar displacement per frame in meters (flow_m components).
- Confidence in [0,1] combines inlier ratio and residual quality.

Requirements
- `focal_length_px` must be set (pixels). Altitude in meters is passed per frame.
- For undistortion, provide `camera` intrinsics in the config.
"""

class OpticalFlowTracker:
    def __init__(self, config):
        cfg = config["vision"]["flow"]
        # Detection and LK parameters
        self.max_corners = cfg.get("max_corners", 200)
        self.quality_level = cfg.get("quality_level", 0.01)
        self.min_distance = cfg.get("min_distance", 7)
        self.block_size = cfg.get("block_size", 7)
        self.win_size = cfg.get("win_size", 21)
        self.max_level = cfg.get("max_level", 3)
        self.min_texture_var = cfg.get("min_texture_var", 100.0)
        self.zupt_threshold = cfg.get("zupt_threshold", 0.01)
        self.flow_reset_threshold = cfg.get("flow_reset_threshold", 0.5)
        self.focal_length_px = cfg.get("focal_length_px", 525.0)
        self.altitude = 1.0  # default initial altitude in meters
        # Optional downscale factor for performance (process smaller image)
        self.downscale = float(cfg.get("downscale", 1.0))

        # Camera intrinsics (optional) for undistortion
        cam_cfg = config.get("camera", {})
        self.fx = cam_cfg.get("fx", None)
        self.fy = cam_cfg.get("fy", None)
        self.cx = cam_cfg.get("cx", None)
        self.cy = cam_cfg.get("cy", None)
        self.dist = cam_cfg.get("distortion", None)
        self.K = None
        if all(v is not None for v in [self.fx, self.fy, self.cx, self.cy]):
            self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)
            if self.dist is not None:
                self.dist = np.array(self.dist, dtype=np.float32)
        # Undistort map cache
        self._ud_shape = None
        self._ud_map1 = None
        self._ud_map2 = None

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(self.win_size, self.win_size),
            maxLevel=self.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # ORB used for seeding points; we keep ORB here even if VPS uses SIFT
        self.orb = cv2.ORB_create(nfeatures=self.max_corners)
        self.prev_gray = None
        self.prev_kps = None
        self.prev_ids = []
        self.feature_id_counter = 0

        # Store points with IDs for tracking
        self.prev_pts = None  # Nx2 array of points
        self.prev_pts_ids = []  # List of feature IDs

        # For visualization overlays (consumed by UI)
        self.prev_points = None  # inlier previous points (Nx2)
        self.last_points = None  # inlier current points (Nx2)

        # For smoothing velocity & ZUPT
        self.vel_buffer = deque(maxlen=5)
        self.last_vel = np.array([0.0, 0.0])
        self.last_texture_var = 0.0
        self.last_flow_mag = 0.0
        self.motion_mask = None

        # Tunables for robust estimation
        self.ransac_reproj_thresh = cfg.get("ransac_reproj_thresh", 3.0)
        self.residual_scale = cfg.get("residual_scale", 3.0)
        self.lk_error_thresh = cfg.get("lk_error_thresh", 20.0)

        # ROI parameters
        self.roi_pad_px = int(cfg.get("roi_pad_px", 40))

        # FB-LK and grid flags
        self.fb_check_enabled = bool(cfg.get("fb_check_enabled", False))
        self.fb_error_thresh = float(cfg.get("fb_error_thresh", 2.0))
        self.grid_rows = int(cfg.get("grid_rows", 0))
        self.grid_cols = int(cfg.get("grid_cols", 0))
        self.features_per_cell = int(cfg.get("features_per_cell", 0))

    def reset(self):
        """Reset internal tracking state for watchdog recovery."""
        self.prev_gray = None
        self.prev_pts = None
        self.prev_pts_ids = []
        self.prev_points = None
        self.last_points = None
        self.vel_buffer.clear()
        self.last_vel = np.array([0.0, 0.0])
        self.motion_mask = None

    def _ensure_ud_maps(self, gray):
        """Ensure undistortion maps are available for the given image size.

        Returns True if maps are prepared and undistortion should be applied.
        """
        if self.K is None or self.dist is None:
            return False
        h, w = gray.shape[:2]
        if self._ud_shape == (h, w) and self._ud_map1 is not None and self._ud_map2 is not None:
            return True
        newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 0)
        self._ud_map1, self._ud_map2 = cv2.initUndistortRectifyMap(self.K, self.dist, None, newK, (w, h), cv2.CV_16SC2)
        self._ud_shape = (h, w)
        return True

    def _undistort(self, gray):
        """Undistort grayscale image if intrinsics are available."""
        if self.K is None or self.dist is None:
            return gray
        if self._ensure_ud_maps(gray):
            return cv2.remap(gray, self._ud_map1, self._ud_map2, interpolation=cv2.INTER_LINEAR)
        return gray

    def _detect_in_roi(self, gray, pts):
        """Detect ORB keypoints within a padded ROI around prior inliers."""
        h, w = gray.shape[:2]
        if pts is None or len(pts) == 0:
            kps = self.orb.detect(gray, None)
            return np.array([kp.pt for kp in kps], dtype=np.float32)
        x_min = max(0, int(np.min(pts[:,0]) - self.roi_pad_px))
        y_min = max(0, int(np.min(pts[:,1]) - self.roi_pad_px))
        x_max = min(w-1, int(np.max(pts[:,0]) + self.roi_pad_px))
        y_max = min(h-1, int(np.max(pts[:,1]) + self.roi_pad_px))
        roi = gray[y_min:y_max+1, x_min:x_max+1]
        kps = self.orb.detect(roi, None)
        if not kps:
            kps = self.orb.detect(gray, None)
            return np.array([kp.pt for kp in kps], dtype=np.float32)
        pts_roi = np.array([kp.pt for kp in kps], dtype=np.float32)
        pts_roi[:,0] += x_min
        pts_roi[:,1] += y_min
        return pts_roi

    def _detect_grid(self, gray):
        """Detect ORB features uniformly across a grid to ensure coverage."""
        if self.grid_rows <= 0 or self.grid_cols <= 0 or self.features_per_cell <= 0:
            kps = self.orb.detect(gray, None)
            return np.array([kp.pt for kp in kps], dtype=np.float32)
        h, w = gray.shape[:2]
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols
        pts_list = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y0 = r * cell_h
                x0 = c * cell_w
                y1 = min(h, y0 + cell_h)
                x1 = min(w, x0 + cell_w)
                mask = np.zeros_like(gray, dtype=np.uint8)
                mask[y0:y1, x0:x1] = 255
                kps = self.orb.detect(gray, mask)
                if not kps:
                    continue
                # Sort by response and take top K
                kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:self.features_per_cell]
                for kp in kps:
                    pts_list.append(kp.pt)
        if not pts_list:
            kps = self.orb.detect(gray, None)
            return np.array([kp.pt for kp in kps], dtype=np.float32)
        return np.array(pts_list, dtype=np.float32)

    def initialize(self, first_frame):
        """Initialize tracking on the first frame by detecting seed features."""
        frame_gray_full = self._illumination_normalize(first_frame)
        frame_gray_full = self._undistort(frame_gray_full)
        if self.downscale != 1.0:
            new_w = max(1, int(round(frame_gray_full.shape[1] * self.downscale)))
            new_h = max(1, int(round(frame_gray_full.shape[0] * self.downscale)))
            frame_gray = cv2.resize(frame_gray_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame_gray = frame_gray_full
        # Detect initial ORB keypoints (grid if configured)
        if self.grid_rows > 0 and self.grid_cols > 0 and self.features_per_cell > 0:
            pts = self._detect_grid(frame_gray)
        else:
            kps = self.orb.detect(frame_gray, None)
            pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        self.prev_pts = pts
        self.prev_pts_ids = self._assign_feature_ids(pts, None, None)
        self.prev_gray = frame_gray
        self.vel_buffer.clear()
        self.last_vel = np.array([0.0, 0.0])
        self.motion_mask = None

    def _has_sufficient_texture(self, gray):
        """Return whether image has enough texture using variance of Laplacian."""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var_lap = lap.var()
        self.last_texture_var = float(var_lap)
        return var_lap >= self.min_texture_var, var_lap

    def _illumination_normalize(self, img):
        """Convert to grayscale and apply histogram equalization."""
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    def _assign_feature_ids(self, pts, prev_pts, prev_ids, max_distance=5.0):
        """Assign stable IDs to features by nearest neighbor association."""
        new_ids = []
        if prev_pts is None or len(prev_pts) == 0:
            # Assign new IDs to all
            for _ in range(len(pts)):
                new_ids.append(self.feature_id_counter)
                self.feature_id_counter += 1
            return new_ids

        prev_pts_np = np.array(prev_pts)
        for pt in pts:
            dists = np.linalg.norm(prev_pts_np - pt, axis=1)
            min_idx = np.argmin(dists)
            if dists[min_idx] < max_distance:
                # Reuse ID
                new_ids.append(prev_ids[min_idx])
            else:
                # New ID
                new_ids.append(self.feature_id_counter)
                self.feature_id_counter += 1
        return new_ids

    def update(self, frame, altitude, dt=None):
        """Track optical flow and estimate planar motion.

        Parameters
        ----------
        frame : np.ndarray (H,W,3) BGR
            Current frame.
        altitude : float
            Altitude in meters used for pixel→meter conversion.
        dt : float | None
            Time delta between frames in seconds. If None, a 30 FPS fallback is used
            for internal velocity smoothing only; return values remain in meters per frame.

        Returns
        -------
        dx_m : float
            Planar displacement in meters along x per frame (camera frame → map frame handled upstream).
        dy_m : float
            Planar displacement in meters along y per frame.
        confidence : float in [0,1]
            Confidence of this update based on inlier ratio and residuals.
        """
        self.altitude = altitude
        frame_gray_full = self._illumination_normalize(frame)
        frame_gray_full = self._undistort(frame_gray_full)
        # Prepare processing image (possibly downscaled)
        if self.downscale != 1.0:
            new_w = max(1, int(round(frame_gray_full.shape[1] * self.downscale)))
            new_h = max(1, int(round(frame_gray_full.shape[0] * self.downscale)))
            frame_gray = cv2.resize(frame_gray_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_px = self.downscale
        else:
            frame_gray = frame_gray_full
            scale_px = 1.0

        has_texture, var_lap = self._has_sufficient_texture(frame_gray)
        if not has_texture:
            # No sufficient texture, reset state
            self.prev_gray = frame_gray
            self.prev_pts = None
            self.prev_pts_ids = []
            self.prev_points = None
            self.last_points = None
            self.vel_buffer.clear()
            self.last_vel = np.array([0.0, 0.0])
            self.motion_mask = None
            return 0.0, 0.0, 1e-6  # failure: low texture

        if self.prev_gray is None:
            # First frame initialization
            if self.grid_rows > 0 and self.grid_cols > 0 and self.features_per_cell > 0:
                pts = self._detect_grid(frame_gray)
            else:
                kps = self.orb.detect(frame_gray, None)
                pts = np.array([kp.pt for kp in kps], dtype=np.float32)
            self.prev_pts = pts
            self.prev_pts_ids = self._assign_feature_ids(pts, None, None)
            self.prev_gray = frame_gray
            self.prev_points = None
            self.last_points = None
            self.motion_mask = None
            return 0.0, 0.0, 1e-6

        # Calculate optical flow from prev to current frame
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            frame_gray,
            self.prev_pts,
            None,
            **self.lk_params
        )

        if next_pts is None or status is None:
            self.motion_mask = None
            return 0.0, 0.0, 1e-6

        # Filter good points with LK status and error threshold
        status_flat = status.flatten()
        valid = (status_flat == 1)
        if error is not None:
            valid = valid & (error.flatten() < self.lk_error_thresh)

        good_prev = self.prev_pts[valid]
        good_next = next_pts[valid]
        good_ids = [fid for fid, st, e in zip(self.prev_pts_ids, status_flat, (error.flatten() if error is not None else np.zeros_like(status_flat))) if (st == 1 and (e < self.lk_error_thresh))]

        # Forward–backward LK gating
        if self.fb_check_enabled and len(good_next) > 0:
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                frame_gray,
                self.prev_gray,
                good_next.reshape(-1,1,2),
                None,
                **self.lk_params
            )
            if back_pts is not None and back_status is not None:
                back_valid = back_status.flatten() == 1
                fb_err = np.linalg.norm(good_prev - back_pts.reshape(-1,2), axis=1)
                keep_fb = (back_valid) & (fb_err < self.fb_error_thresh)
                good_prev = good_prev[keep_fb]
                good_next = good_next[keep_fb]
                good_ids = [gid for gid, k in zip(good_ids, keep_fb) if k]

        if len(good_prev) < 4:
            # Not enough points, re-detect (prefer grid or ROI)
            if self.grid_rows > 0 and self.grid_cols > 0 and self.features_per_cell > 0:
                pts = self._detect_grid(frame_gray)
            else:
                pts = self._detect_in_roi(frame_gray, self.prev_points)
            self.prev_pts = pts
            self.prev_pts_ids = self._assign_feature_ids(pts, None, None)
            self.prev_gray = frame_gray
            self.prev_points = None
            self.last_points = None
            self.vel_buffer.clear()
            self.last_vel = np.array([0.0, 0.0])
            self.motion_mask = None
            return 0.0, 0.0, 1e-6

        # Robust estimation using affine model (RANSAC)
        M, inlier_mask = cv2.estimateAffine2D(
            good_prev,
            good_next,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_thresh,
            maxIters=2000,
            confidence=0.99,
            refineIters=10
        )
        if M is None or inlier_mask is None:
            # Reset on failure (prefer grid or ROI)
            if self.grid_rows > 0 and self.grid_cols > 0 and self.features_per_cell > 0:
                pts = self._detect_grid(frame_gray)
            else:
                pts = self._detect_in_roi(frame_gray, self.prev_points)
            self.prev_pts = pts
            self.prev_pts_ids = self._assign_feature_ids(pts, None, None)
            self.prev_gray = frame_gray
            self.prev_points = None
            self.last_points = None
            self.vel_buffer.clear()
            self.last_vel = np.array([0.0, 0.0])
            self.motion_mask = None
            return 0.0, 0.0, 1e-6

        inliers = inlier_mask.flatten().astype(bool)
        if np.count_nonzero(inliers) < 4:
            # Too few inliers
            if self.grid_rows > 0 and self.grid_cols > 0 and self.features_per_cell > 0:
                pts = self._detect_grid(frame_gray)
            else:
                pts = self._detect_in_roi(frame_gray, self.prev_points)
            self.prev_pts = pts
            self.prev_pts_ids = self._assign_feature_ids(pts, None, None)
            self.prev_gray = frame_gray
            self.prev_points = None
            self.last_points = None
            self.vel_buffer.clear()
            self.last_vel = np.array([0.0, 0.0])
            self.motion_mask = None
            return 0.0, 0.0, 1e-6

        inlier_prev = good_prev[inliers]
        inlier_next = good_next[inliers]
        inlier_ids = [gid for gid, keep in zip(good_ids, inliers) if keep]

        # Residuals and gating (median + MAD for robust scale)
        pred = (inlier_prev @ M[:,:2].T) + M[:,2]
        residuals = np.linalg.norm(pred - inlier_next, axis=1)
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med)) + 1e-6
        # Adapt threshold within bounds
        thresh = self.ransac_reproj_thresh
        gate = med + self.residual_scale * mad
        keep = residuals < max(thresh, gate)

        # Build motion mask: suppress outliers/high-residual regions
        mask_dyn = np.full(frame_gray.shape, 255, dtype=np.uint8)
        # Points rejected from inliers by residual gate
        bad_inlier_points = inlier_next[~keep] if np.any(~keep) else np.empty((0,2))
        # Points not inliers at all
        not_inlier_points = good_next[~inliers] if np.any(~inliers) else np.empty((0,2))
        for arr in (bad_inlier_points, not_inlier_points):
            for pt in arr:
                x = int(round(pt[0])); y = int(round(pt[1]))
                if 0 <= x < mask_dyn.shape[1] and 0 <= y < mask_dyn.shape[0]:
                    cv2.circle(mask_dyn, (x, y), 12, 0, -1)
        # Upsample motion mask to full resolution if needed
        if scale_px != 1.0:
            self.motion_mask = cv2.resize(mask_dyn, (frame_gray_full.shape[1], frame_gray_full.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            self.motion_mask = mask_dyn

        inlier_prev = inlier_prev[keep]
        inlier_next = inlier_next[keep]
        inlier_ids = [iid for iid, k in zip(inlier_ids, keep) if k]

        if len(inlier_prev) < 3:
            # Not enough after gating
            self.prev_gray = frame_gray
            self.prev_points = None
            self.last_points = None
            return 0.0, 0.0, 1e-6

        # Expose for overlay
        self.prev_points = inlier_prev.copy()
        self.last_points = inlier_next.copy()

        # Median flow for robustness
        flow_vectors = inlier_next - inlier_prev
        median_flow = np.median(flow_vectors, axis=0)
        # Rescale pixel flow back to original resolution if downscaled
        median_flow = median_flow / scale_px

        # Convert pixel flow to meters using altitude and focal length
        flow_m = median_flow * self.altitude / self.focal_length_px

        # Adaptive ZUPT based on texture
        adaptive_zupt = self.zupt_threshold * (1.5 if var_lap < (2.0 * self.min_texture_var) else 1.0)

        flow_mag = np.linalg.norm(flow_m)
        self.last_flow_mag = float(flow_mag)
        if flow_mag < adaptive_zupt:
            velocity = np.array([0.0, 0.0])
        else:
            if dt is not None and dt > 1e-6:
                velocity = flow_m / dt
            else:
                velocity = flow_m / (1.0 / 30.0)  # fallback to 30 FPS assumption

        # Smooth velocity (used for diagnostics/UI)
        self.vel_buffer.append(velocity)
        smooth_velocity = np.mean(np.array(self.vel_buffer), axis=0)
        self.last_vel = smooth_velocity

        # Confidence from inlier ratio and residuals
        inlier_ratio = float(len(inlier_prev)) / float(len(good_prev) + 1e-6)
        mean_res = float(np.mean(residuals[keep])) if np.any(keep) else 1e6
        conf = inlier_ratio * np.exp(-mean_res / max(1.0, self.ransac_reproj_thresh))
        confidence = float(max(0.0, min(1.0, conf)))

        # Prepare for next frame
        self.prev_gray = frame_gray
        self.prev_pts = inlier_next.reshape(-1, 2)
        self.prev_pts_ids = inlier_ids

        # If too few features left, re-detect to maintain feature set
        if len(self.prev_pts) < self.max_corners * 0.3:
            if self.grid_rows > 0 and self.grid_cols > 0 and self.features_per_cell > 0:
                new_pts = self._detect_grid(frame_gray)
            else:
                new_pts = self._detect_in_roi(frame_gray, self.prev_points)
            new_ids = self._assign_feature_ids(new_pts, self.prev_pts, self.prev_pts_ids)
            self.prev_pts = new_pts
            self.prev_pts_ids = new_ids

        return flow_m[0], flow_m[1], confidence

