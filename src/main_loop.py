#!/usr/bin/env python3
"""
Enhanced GPS-denied navigation loop with:
• Gazebo camera input
• Optical-flow Visual-Inertial Odometry (VIO)
• SIFT-based Visual Place Recognition (VPS)
• IMU-based fallback
• Factor-graph fusion
• MAVLink VISION_POSITION_ESTIMATE output
• Dual-screen visualization (camera + map estimate)
• Debug logs, timestamps, and adaptive fusion

High-level flow
- Initialize modules from config: camera, MAVLink, OpticalFlow, VIO EKF, VPS, factor graph
- Wait for takeoff altitude, bootstrap initial pose from VPS if available
- In a loop:
  * Read frame and IMU, compute altitude
  * Run OF/VIO at configured/adaptive rate; optionally fuse ATTITUDE yaw
  * Run VPS at configured/adaptive rate; maintain smoothed fixes buffer
  * Add relative+absolute+yaw factors to optimizer; run optimization
  * Publish fused vision pose to MAVLink with adaptive covariance
  * Optional UI overlays for diagnostics; periodic CSV summaries
"""

import os
import sys
import time
import yaml
import cv2
import numpy as np
import logging
import csv
import bisect
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

from src.sensors.gazebo_camera import GazeboCamera
from src.vision.optical_flow import OpticalFlowTracker
from src.vision.vio_tracker import VIOTracker
from src.vision.vps_transformer import VPSTransformer
from src.fusion.factor_graph import FactorGraphOptimizer
from src.mavlink_sender import MAVLinkSender

# Set up logging
logging.basicConfig(
    filename='navigation_debug.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

@dataclass
class IMUData:
    """Simple IMU container with `timestamp`, `accel` (3,), `gyro` (3,)."""
    timestamp: float
    accel: Optional[np.ndarray] = None
    gyro: Optional[np.ndarray] = None

@dataclass
class FrameData:
    """Camera frame with `timestamp` and BGR image array."""
    timestamp: float
    image: np.ndarray

@dataclass
class OpticalFlowOutput:
    """Optical flow output packaged for the main loop."""
    dx: float
    dy: float
    confidence: float

@dataclass
class VPSFix:
    """A smoothed VPS fix with position, yaw, and confidence."""
    x: float
    y: float
    z: float
    yaw: float
    confidence: float

@dataclass
class PoseEstimate:
    """Fused pose estimate fields used by overlays and logging."""
    x: float
    y: float
    z: float
    yaw: float


class NavigationPipeline:
    """Top-level orchestrator for the GPS-denied navigation runtime."""
    def __init__(self, cfg_path=None):
        """Load configuration, initialize subsystems, and prepare runtime state."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_file = cfg_path or os.path.join(base_dir, "config.yaml")

        with open(cfg_file, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Validate configuration early
        self._validate_config()

        # Monotonic clock for scheduling and watchdogs
        self._mono = time.monotonic

        # OpenCV optimization settings
        rt_cv = self.cfg.get("runtime", {}).get("opencv", {})
        try:
            cv2.setUseOptimized(bool(rt_cv.get("use_optimized", True)))
        except Exception:
            pass
        try:
            cv2.setNumThreads(int(rt_cv.get("num_threads", 0)))
        except Exception:
            pass

        # MAVLink connection to ArduPilot (request IMU/ATTITUDE/GPS streams)
        self.mav = MAVLinkSender(
            connection_str=self.cfg["mavlink"].get("connection_str", "udpout:localhost:14550"),
            vision_rate_hz=self.cfg["mavlink"].get("vision_rate_hz", 10)
        )
        self.mav.start()

        # Gazebo H.264 UDP camera via GStreamer backend
        self.camera = GazeboCamera(udp_port=self.cfg["camera"].get("udp_port", 5600))
        self.camera.start()

        # Visual modules
        vio_cfg = self.cfg["vision"]["vio"]
        self.optical_vio = OpticalFlowTracker(self.cfg)   # image-space OF → planar meters
        
        self.imu_vio = VIOTracker(self.cfg)              # EKF on IMU + OF/VPS updates

        # VPSTransformer with optional pixels_per_meter and camera intrinsics from config
        ppm = self.cfg.get("map", {}).get("pixels_per_meter", None)
        cam = self.cfg.get("camera", {})
        yaw_sign = int(self.cfg.get("fusion", {}).get("yaw_sign", 1))
        K = None
        dist = None
        if all(k in cam for k in ("fx","fy","cx","cy")):
            K = np.array([[float(cam["fx"]), 0.0, float(cam["cx"])], [0.0, float(cam["fy"]), float(cam["cy"])], [0.0, 0.0, 1.0]], dtype=np.float32)
            if "distortion" in cam:
                dist = np.array(cam["distortion"], dtype=np.float32)
        vps_cfg = self.cfg["vision"]["vps"]
        self.vps_transform = VPSTransformer(
            map_path=vps_cfg["map_path"],
            focal_px=vps_cfg.get("focal_px", 525.0),
            pixels_per_meter=ppm,
            camera_K=K,
            camera_dist=dist,
            yaw_sign=yaw_sign,
            scale_levels=vps_cfg.get("scale_levels", [1.0]),
            tile_rows=vps_cfg.get("tile_rows", 0),
            tile_cols=vps_cfg.get("tile_cols", 0),
            features_per_tile=vps_cfg.get("features_per_tile", 0),
            ratio_thresh=vps_cfg.get("ratio_thresh", 0.7)
        )
        # Apply early-exit score threshold if provided in config
        try:
            self.vps_transform.early_score_threshold = float(vps_cfg.get("early_score_threshold", 0.0))
        except Exception:
            pass

        # Factor-graph optimizer wrapper (GTSAM if available; simple fallback otherwise)
        self.fuser = FactorGraphOptimizer(
            window_size=self.cfg["fusion"].get("window_size", 10),
            config=self.cfg
        )

        # Map image for overlays
        self.map_image = cv2.imread(self.cfg["vision"]["vps"]["map_path"])
        self.x = self.y = self.z = self.yaw = 0.0
        # Wall-clock used for logging/CSV; monotonic for scheduling
        self._last_vio_time_mono = self._mono()
        self._last_vps_time_mono = self._mono()

        # Visualization and confidence/FPS tracking
        self.trail = []
        self.last_confidence = 0.0
        self.last_fps = 0.0
        self.last_frame_time = time.time()

        # Buffers for timestamp synchronization
        self.imu_buffer: deque[IMUData] = deque(maxlen=200)
        self.imu_time_buffer: deque[float] = deque(maxlen=200)
        self.frame_buffer: deque[FrameData] = deque(maxlen=50)
        self.vps_buffer: deque[VPSFix] = deque(maxlen=self.cfg["vision"]["vps"].get("smoothing_window", 5))

        # For adaptive fusion fallback
        self.low_conf_counter = 0
        self.low_conf_threshold = self.cfg["vision"]["vps"].get("low_conf_threshold", 0.3)
        self.low_conf_max_count = self.cfg["vision"]["vps"].get("low_conf_max_count", 10)
        self.max_cov = self.cfg["vision"]["vio"].get("max_cov", 1.0)

        # Yaw prior sigma (radians)
        self.yaw_prior_sigma = self.cfg["fusion"].get("yaw_prior_sigma", 0.05)
        # EKF yaw fuse from ATTITUDE flag
        self.yaw_update_from_attitude = bool(self.cfg.get("vision", {}).get("vio", {}).get("yaw_update_from_attitude", False))

        # Headless flag for deployment
        self.headless = bool(self.cfg.get("ui", {}).get("headless", False))

        # For periodic summaries
        self._last_summary_time = time.time()
        self._summary_interval = float(self.cfg.get("logging", {}).get("summary_interval_sec", 1.0))

        # Behavior flags
        self.use_flow_as_velocity = bool(self.cfg["vision"].get("use_flow_as_velocity", False))
        self.use_motion_mask = bool(self.cfg["vision"]["vps"].get("use_motion_mask", False))
        # Avoid double-smoothing: by default rely on VPS internal smoothing
        self.main_loop_vps_smoothing = bool(self.cfg["vision"]["vps"].get("main_loop_smoothing", False))

        # UI toggles
        self.ui_show_tracks = True
        self.ui_show_mask = False
        self.ui_show_bars = True
        self.ui_show_scale = True
        self._ppm_overlay = float(self.cfg.get("map", {}).get("pixels_per_meter", 20) or 20)

        # Watchdogs
        rt = self.cfg.get("runtime", {})
        wd = rt.get("watchdogs", {})
        self.of_timeout = float(wd.get("of_timeout_sec", 1.0))
        self.vps_timeout = float(wd.get("vps_timeout_sec", 3.0))
        self.imu_timeout = float(wd.get("imu_timeout_sec", 1.0))
        self._last_of_ok_mono = self._mono()
        self._last_vps_ok_mono = self._mono()
        self._last_imu_ok_mono = self._mono()
        # Distinguish failure vs stillness for OF
        self.of_fail_count = 0
        self.of_fail_threshold = int(wd.get("of_fail_threshold", 5))

        # Adaptive rates configuration
        ar = rt.get("adaptive_rates", {})
        self.adaptive_enabled = bool(ar.get("enabled", False))
        self.vio_rate_min = float(ar.get("vio_rate_min_hz", 10))
        self.vio_rate_max = float(ar.get("vio_rate_max_hz", 30))
        self.vps_rate_min = float(ar.get("vps_rate_min_hz", 0.5))
        self.vps_rate_max = float(ar.get("vps_rate_max_hz", 2.0))
        self.of_conf_low = float(ar.get("of_conf_low", 0.25))
        self.of_conf_high = float(ar.get("of_conf_high", 0.7))
        self.vps_conf_low = float(ar.get("vps_conf_low", 0.25))
        self.vps_conf_high = float(ar.get("vps_conf_high", 0.7))

        # Dynamic rates (start at configured)
        self._vio_rate_hz = float(self.cfg["vision"]["vio"].get("flow_rate_hz", 30))
        self._vps_rate_hz = float(self.cfg["vision"]["vps"].get("rate_hz", 1))
        self._vio_rate_target = self._vio_rate_hz
        self._vps_rate_target = self._vps_rate_hz
        # Adaptive rate smoothing and hysteresis
        self._rate_alpha = 0.2
        self._rate_hyst = 0.05

        # Evaluation CSV logging
        log_cfg = self.cfg.get("logging", {})
        self.eval_csv_path = log_cfg.get("eval_csv_path", None)
        self.enable_truth = bool(log_cfg.get("enable_truth", False))
        self._csv_file = None
        self._csv_writer = None
        if self.eval_csv_path:
            header = [
                "t", "x", "y", "z", "yaw",
                "of_conf", "vps_conf", "vio_rate", "vps_rate",
                "truth_x", "truth_y", "truth_z", "truth_yaw"
            ]
            try:
                self._csv_file = open(self.eval_csv_path, mode='w', newline='')
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(header)
            except Exception as e:
                logging.error(f"Failed to open eval CSV: {e}")
                self._csv_file = None
                self._csv_writer = None

        # Recording (frames, IMU, optional video)
        rec_cfg = rt.get("recording", {})
        self.rec_enabled = bool(rec_cfg.get("enabled", False))
        self.rec_dir = rec_cfg.get("dir", "recordings/run_01")
        self.rec_save_frames = bool(rec_cfg.get("save_frames", True))
        self.rec_save_imu = bool(rec_cfg.get("save_imu", True))
        self.rec_save_video = bool(rec_cfg.get("save_video", False))
        self.rec_video_fps = int(rec_cfg.get("video_fps", 20))
        self.rec_image_format = rec_cfg.get("image_format", "png")
        self._rec_imu_file = None
        self._rec_vid = None
        if self.rec_enabled:
            os.makedirs(self.rec_dir, exist_ok=True)
            if self.rec_save_imu:
                try:
                    self._rec_imu_file = open(os.path.join(self.rec_dir, 'imu.csv'), 'w', newline='')
                    imu_writer = csv.writer(self._rec_imu_file)
                    imu_writer.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz"]) 
                    self._rec_imu_writer = imu_writer
                except Exception as e:
                    logging.error(f"Failed to open IMU CSV: {e}")
                    self._rec_imu_file = None
                    self._rec_imu_writer = None
            else:
                self._rec_imu_writer = None
            if self.rec_save_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # Auto-detect frame size from camera if available
                size = self.camera.get_frame_size() if hasattr(self.camera, 'get_frame_size') else None
                if size is None:
                    size = (640, 480)
                self._rec_vid = cv2.VideoWriter(os.path.join(self.rec_dir, 'video.mp4'), fourcc, self.rec_video_fps, size)

    def _get_altitude(self):
        """Return current relative altitude in meters from GLOBAL_POSITION_INT if available."""
        msg = self.mav.conn.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        return msg.relative_alt / 1000.0 if msg else self.z

    def _get_attitude_yaw(self):
        """Return ATTITUDE yaw (radians) if available, applying configured yaw_sign."""
        att = None
        try:
            att = self.mav.latest_attitude()
        except Exception:
            logging.debug("[ATT] latest_attitude() raised; continuing without yaw prior")
        if att and "yaw" in att:
            return att["yaw"] * float(self.cfg["fusion"].get("yaw_sign", 1))
        return None

    def _interpolate_imu_to(self, ts):
        """Linear interpolate IMU accel/gyro to the given timestamp ts (seconds)."""
        if len(self.imu_buffer) < 2:
            return None
        # Use bisect over timestamp buffer to locate neighbors
        times = self.imu_time_buffer
        if not times or ts < times[0] or ts > times[-1]:
            return None
        idx = bisect.bisect_right(list(times), ts)
        if idx == 0 or idx >= len(times):
            return None
        a = self.imu_buffer[idx-1]
        b = self.imu_buffer[idx]
        t0, t1 = a.timestamp, b.timestamp
        if t1 <= t0:
            return a
        alpha = (ts - t0) / (t1 - t0)
        accel = a.accel * (1 - alpha) + b.accel * alpha
        gyro = a.gyro * (1 - alpha) + b.gyro * alpha
        return IMUData(timestamp=ts, accel=accel, gyro=gyro)

    def _log_data(self, **kwargs):
        """Placeholder for per-frame detailed logging; summaries handled separately."""
        pass

    def _log_summary(self, **kwargs):
        """Emit a single-line summary with key metrics for easy tailing."""
        log_msg = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logging.info(log_msg)

    def _draw_conf_bars(self, frame, of_conf: float, vps_conf: float):
        """Draw simple vertical confidence bars for OF and VPS on the given frame."""
        try:
            h, w = frame.shape[:2]
            bar_w = int(18)
            pad = 10
            max_h = int(0.35 * h)
            # normalize 0..1
            of_h = int(max(0.0, min(1.0, of_conf)) * max_h)
            vps_h = int(max(0.0, min(1.0, vps_conf)) * max_h)
            # background
            cv2.rectangle(frame, (pad, pad), (pad + bar_w, pad + max_h), (40, 40, 40), 1)
            cv2.rectangle(frame, (pad*2 + bar_w, pad), (pad*2 + bar_w*2, pad + max_h), (40, 40, 40), 1)
            # bars
            of_color = (0, 255, 0) if of_conf >= 0.5 else ((0, 200, 255) if of_conf >= 0.25 else (0, 0, 255))
            vps_color = (0, 255, 0) if vps_conf >= 0.5 else ((0, 200, 255) if vps_conf >= 0.25 else (0, 0, 255))
            cv2.rectangle(frame, (pad+1, pad + max_h - of_h), (pad + bar_w - 1, pad + max_h - 1), of_color, -1)
            cv2.rectangle(frame, (pad*2 + bar_w + 1, pad + max_h - vps_h), (pad*2 + bar_w*2 - 1, pad + max_h - 1), vps_color, -1)
            cv2.putText(frame, "OF", (pad, pad + max_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, "VPS", (pad*2 + bar_w, pad + max_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        except Exception:
            pass

    def _draw_map_overlay(self, x, y, yaw=0.0, confidence=0.0):
        """Render a simple map overlay with position trail, heading arrow, and scale."""
        if self.map_image is None:
            return np.zeros((400, 400, 3), dtype=np.uint8)

        overlay = self.map_image.copy()
        # Use map.pixels_per_meter for consistent scaling
        scale = self.cfg.get("map", {}).get("pixels_per_meter", 20) or 20
        h, w = overlay.shape[:2]

        px = int(w / 2 + x * scale)
        py = int(h / 2 - y * scale)

        self.trail.append((px, py))
        if len(self.trail) > 50:
            self.trail.pop(0)

        for i in range(1, len(self.trail)):
            cv2.line(overlay, self.trail[i-1], self.trail[i], (0, 255, 0), 2)

        cv2.circle(overlay, (px, py), 8, (0, 0, 255), -1)

        arrow_length = 30
        end_x = int(px + arrow_length * np.cos(yaw))
        end_y = int(py - arrow_length * np.sin(yaw))
        cv2.arrowedLine(overlay, (px, py), (end_x, end_y), (255, 0, 0), 3, tipLength=0.3)

        text = f"Pos: ({x:.2f}, {y:.2f}, {self.z:.2f}) m"
        cv2.putText(overlay, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(overlay, conf_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Scale bar and north arrow
        if self.ui_show_scale:
            try:
                ppm = float(self._ppm_overlay)
                meters = 5
                px_len = int(meters * ppm)
                px_len = max(40, min(px_len, int(0.25*w)))
                x0 = int(w*0.07)
                y0 = int(h*0.92)
                cv2.line(overlay, (x0, y0), (x0 + px_len, y0), (255,255,255), 3)
                cv2.line(overlay, (x0, y0-5), (x0, y0+5), (255,255,255), 3)
                cv2.line(overlay, (x0 + px_len, y0-5), (x0 + px_len, y0+5), (255,255,255), 3)
                cv2.putText(overlay, f"{meters} m", (x0 + px_len + 8, y0+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                # north arrow (up)
                nax, nay = int(w*0.93), int(h*0.85)
                cv2.arrowedLine(overlay, (nax, nay+25), (nax, nay-25), (255,255,255), 3, tipLength=0.35)
                cv2.putText(overlay, "N", (nax-8, nay-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            except Exception:
                pass

        return overlay

    def _draw_camera_overlay(self, frame, altitude, x, y, z, yaw, confidence):
        """Render text overlays, confidence bars, OF track arrows, and optional mask inset."""
        overlay = frame.copy()

        info_lines = [
            f"Altitude: {altitude:.2f} m",
            f"Position: x={x:.2f} m, y={y:.2f} m, z={z:.2f} m",
            f"Yaw: {np.degrees(yaw):.1f} deg",
            f"VPS Confidence: {confidence:.2f}",
            f"FPS: {self.last_fps:.1f}"
        ]

        y0, dy = 30, 25
        for i, line in enumerate(info_lines):
            ytxt = y0 + i * dy
            cv2.putText(overlay, line, (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Confidence bars
        if self.ui_show_bars:
            self._draw_conf_bars(overlay, getattr(self, '_last_of_conf', 0.0), self.last_confidence)

        # Draw optical flow arrows (tracks)
        if self.ui_show_tracks and hasattr(self.optical_vio, 'last_points') and hasattr(self.optical_vio, 'prev_points'):
            if self.optical_vio.last_points is not None and self.optical_vio.prev_points is not None:
                for p0, p1 in zip(self.optical_vio.prev_points, self.optical_vio.last_points):
                    p0 = tuple(map(int, p0.ravel()))
                    p1 = tuple(map(int, p1.ravel()))
                    cv2.arrowedLine(overlay, p0, p1, (0, 255, 255), 1, tipLength=0.3)

        # Motion mask inset
        try:
            if self.ui_show_mask and hasattr(self.optical_vio, 'motion_mask') and self.optical_vio.motion_mask is not None:
                mask = self.optical_vio.motion_mask
                mh, mw = mask.shape[:2]
                scale = 160 / float(mw)
                mres = cv2.resize(mask, (160, int(mh*scale)), interpolation=cv2.INTER_NEAREST)
                mres_col = cv2.applyColorMap((255-mres), cv2.COLORMAP_OCEAN)
                x0 = overlay.shape[1]-mres_col.shape[1]-10
                y0 = 10
                roi = overlay[y0:y0+mres_col.shape[0], x0:x0+mres_col.shape[1]]
                alpha = 0.55
                cv2.addWeighted(mres_col, alpha, roi, 1-alpha, 0, roi)
                cv2.rectangle(overlay, (x0, y0), (x0+mres_col.shape[1], y0+mres_col.shape[0]), (255,255,255), 1)
                cv2.putText(overlay, "Mask", (x0+4, y0+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        except Exception:
            pass

        return overlay

    def _validate_config(self):
        """Validate and warn about configuration inconsistencies or deprecated keys."""
        try:
            # Required sections
            required_sections = ["camera", "vision", "fusion", "mavlink", "mission", "runtime"]
            for sec in required_sections:
                if sec not in self.cfg:
                    logging.warning(f"[CONFIG] Missing section: {sec}")

            # pixels_per_meter should reside under map
            if "map" in self.cfg:
                if "pixels_per_meter" not in self.cfg["map"]:
                    logging.info("[CONFIG] map.pixels_per_meter not set; defaulting to 10 for overlays and VPS fallback.")
            else:
                logging.warning("[CONFIG] Missing 'map' section; using defaults for scaling.")

            # Deprecated/legacy VIO keys warning
            vio = self.cfg.get("vision", {}).get("vio", {})
            legacy_keys = ["lk_params", "feature_params", "resize_factor", "altitude_ref", "proc_noise_pos", "proc_noise_vel", "proc_noise_att", "flow_noise", "alt_noise", "init_P", "min_cov"]
            found_legacy = [k for k in legacy_keys if k in vio]
            if found_legacy:
                logging.info(f"[CONFIG] Legacy VIO keys present (kept for compatibility, not actively used): {found_legacy}")

            # Vision flow min confidence sanity
            flow_min_conf = self.cfg.get("vision", {}).get("flow", {}).get("min_of_confidence", 0.1)
            if not (0.0 <= float(flow_min_conf) <= 1.0):
                logging.warning(f"[CONFIG] vision.flow.min_of_confidence out of [0,1]: {flow_min_conf}")
        except Exception as e:
            logging.warning(f"[CONFIG] Validation encountered an issue: {e}")

    def _read_imu(self):
        """Fetch latest IMU/ATTITUDE sample from MAVLink caches; append to buffers and record if enabled."""
        imu_raw = self.mav.latest_imu()
        if imu_raw:
            imu_data = IMUData(
                timestamp=time.time(),
                accel=np.array([imu_raw.get("ax", 0), imu_raw.get("ay", 0), imu_raw.get("az", 0)]),
                gyro=np.array([imu_raw.get("gx", 0), imu_raw.get("gy", 0), imu_raw.get("gz", 0)])
            )
            self.imu_buffer.append(imu_data)
            self.imu_time_buffer.append(imu_data.timestamp)
            self._last_imu_ok_mono = self._mono()
            # Record IMU
            if self.rec_enabled and self._rec_imu_writer is not None:
                self._rec_imu_writer.writerow([
                    imu_data.timestamp,
                    imu_data.accel[0], imu_data.accel[1], imu_data.accel[2],
                    imu_data.gyro[0], imu_data.gyro[1], imu_data.gyro[2]
                ])
            return imu_data
        return None

    def _read_frame(self):
        """Read next camera frame (non-blocking) and push to frame buffer; optionally record to disk/video."""
        frame, _ = self.camera.read(timeout=0.1)
        if frame is not None:
            frame_data = FrameData(timestamp=time.time(), image=frame)
            self.frame_buffer.append(frame_data)
            # Record frame
            if self.rec_enabled:
                if self.rec_save_frames:
                    fname = f"frame_{int(frame_data.timestamp*1000)}.{self.rec_image_format}"
                    try:
                        cv2.imwrite(os.path.join(self.rec_dir, fname), frame)
                    except Exception:
                        pass
                if self.rec_save_video and self._rec_vid is not None:
                    try:
                        self._rec_vid.write(frame)
                    except Exception:
                        pass
            return frame_data
        return None

    def _adjust_rates(self, of_conf, vps_conf):
        """Adapt VIO and VPS processing rates based on confidence with hysteresis and EMA smoothing."""
        if not self.adaptive_enabled:
            return
        # VIO rate adapts to OF confidence with hysteresis and smoothing
        if of_conf < (self.of_conf_low - self._rate_hyst):
            self._vio_rate_target = max(self.vio_rate_min, self._vio_rate_target * 0.85)
        elif of_conf > (self.of_conf_high + self._rate_hyst):
            self._vio_rate_target = min(self.vio_rate_max, self._vio_rate_target * 1.10)
        # VPS rate adapts to VPS confidence with hysteresis and smoothing
        if vps_conf < (self.vps_conf_low - self._rate_hyst):
            self._vps_rate_target = max(self.vps_rate_min, self._vps_rate_target * 0.85)
        elif vps_conf > (self.vps_conf_high + self._rate_hyst):
            self._vps_rate_target = min(self.vps_rate_max, self._vps_rate_target * 1.10)
        # Low-pass smoothing to avoid thrash
        self._vio_rate_hz = (1.0 - self._rate_alpha) * self._vio_rate_hz + self._rate_alpha * self._vio_rate_target
        self._vps_rate_hz = (1.0 - self._rate_alpha) * self._vps_rate_hz + self._rate_alpha * self._vps_rate_target

    def _process_optical_flow(self, frame: np.ndarray, altitude: float) -> Optional[OpticalFlowOutput]:
        """Run optical flow update and wrap result into `OpticalFlowOutput`. Returns None on failure."""
        # Use last measured dt from monotonic clock when available
        dt_hint = self._mono() - self._last_vio_time_mono if self._last_vio_time_mono else None
        res = self.optical_vio.update(frame, altitude, dt=dt_hint)
        if res is None:
            return None
        dx, dy, conf = res
        return OpticalFlowOutput(dx, dy, conf)

    def _process_vps(self, frame: np.ndarray, altitude: float) -> Optional[VPSFix]:
        """Run VPS matcher with optional motion mask and return latest fix pushed to buffer."""
        motion_mask = self.optical_vio.motion_mask if self.use_motion_mask else None
        match = self.vps_transform.match(frame, motion_mask, altitude)
        if match:
            fix = VPSFix(
                x=match["x"],
                y=match["y"],
                z=match["z"],
                yaw=match["yaw"],
                confidence=match["confidence"]
            )
            # Sanity gate: drop if jump too large relative to last pose and confidence low
            jump = np.linalg.norm([fix.x - self.x, fix.y - self.y])
            max_jump = float(self.cfg["vision"]["vps"].get("max_jump_m", 10.0))
            if jump > max_jump and fix.confidence < 0.5:
                logging.info(f"[VPS] Dropped large jump {jump:.2f} m at low conf {fix.confidence:.2f}")
            else:
                self.vps_buffer.append(fix)
                self._last_vps_ok_mono = self._mono()
                return fix
        return None

    def _compute_smoothed_vps(self) -> VPSFix:
        """Compute confidence-weighted average of buffered VPS fixes with circular yaw mean."""
        if not self.vps_buffer:
            return VPSFix(0.0, 0.0, 0.0, 0.0, 0.0)

        total_weight = sum(fix.confidence for fix in self.vps_buffer)
        if total_weight == 0:
            return VPSFix(0.0, 0.0, 0.0, 0.0, 0.0)

        avg_x = sum(fix.x * fix.confidence for fix in self.vps_buffer) / total_weight
        avg_y = sum(fix.y * fix.confidence for fix in self.vps_buffer) / total_weight
        avg_z = sum(fix.z * fix.confidence for fix in self.vps_buffer) / total_weight

        yaw_complex = sum(
            complex(np.cos(fix.yaw), np.sin(fix.yaw)) * fix.confidence for fix in self.vps_buffer
        ) / total_weight
        avg_yaw = np.angle(yaw_complex)
        avg_conf = total_weight / len(self.vps_buffer)

        return VPSFix(avg_x, avg_y, avg_z, avg_yaw, avg_conf)

    def _add_to_factor_graph(self, vio_dx, vio_dy, vps_fix: VPSFix, of_confidence: float):
        """Add relative/absolute/yaw/continuity factors with adaptive covariance into the optimizer."""
        now = time.time()
        t0 = self.fuser.timestamps[-1] if getattr(self.fuser, 'timestamps', []) else now

        # Adaptive covariances based on confidence
        vps_cov = (1 - vps_fix.confidence) * self.max_cov
        of_cov = (1 - of_confidence) * self.max_cov

        # Fallback: if VPS confidence low for prolonged time, increase VPS covariance (down-weight VPS)
        if vps_fix.confidence < self.low_conf_threshold:
            self.low_conf_counter += 1
        else:
            self.low_conf_counter = 0

        if self.low_conf_counter > self.low_conf_max_count:
            vps_cov = self.max_cov * 2.0  # reduce VPS trust
            logging.info(f"Low VPS confidence {vps_fix.confidence:.2f} for {self.low_conf_counter} frames; fall back weighting.")

        # Add relative motion from VIO only if confidence is reasonable
        added_relative = False
        if vio_dx is not None and vio_dy is not None and of_confidence >= float(self.cfg["vision"]["flow"].get("min_of_confidence", 0.1)):
            self.fuser.add_relative(t0, now, vio_dx, vio_dy, 0.0, 0.0, covariance=of_cov)
            added_relative = True

        # Add absolute pose from VPS
        self.fuser.add_absolute(now, vps_fix.x, vps_fix.y, vps_fix.z, vps_fix.yaw, covariance=vps_cov)

        # Add yaw prior from ATTITUDE if available
        yaw_att = self._get_attitude_yaw()
        if yaw_att is not None:
            try:
                self.fuser.add_yaw_prior(now, yaw_att, sigma=self.yaw_prior_sigma)
            except Exception:
                pass

        # If no relative factor this cycle and continuity is configured, add continuity prior
        continuity_sigma = self.cfg["fusion"].get("continuity_sigma", None)
        if continuity_sigma is not None and not added_relative:
            try:
                self.fuser.add_continuity(t0, now, sigma=continuity_sigma)
            except Exception:
                pass

    def initialize(self):
        """Block until takeoff altitude is reached, initialize trackers, seed initial pose."""
        print("[INIT] Waiting for takeoff...")
        target_alt = self.cfg["mission"].get("takeoff_alt", 10.0)

        # Wait for altitude
        while True:
            msg = self.mav.conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True)
            if msg:
                alt = msg.relative_alt / 1000.0
                print(f"[INIT] Altitude: {alt:.2f} m")
                if alt >= target_alt:
                    break

        frame_data = self._read_frame()
        if frame_data is None:
            print("[ERROR] No camera frame received during init.")
            sys.exit(1)

        # Initialize visual modules using first frame
        self.optical_vio.initialize(frame_data.image)
        self.imu_vio.initialize(frame_data.image)

        # Bootstrap pose from VPS if possible
        vps_fix = self._process_vps(frame_data.image, target_alt)
        if vps_fix:
            dx, dy, dz, conf = vps_fix.x, vps_fix.y, vps_fix.z, vps_fix.confidence
        else:
            dx = dy = 0.0
            dz = target_alt
            conf = 0.0

        cov = (1 - conf) * self.max_cov
        self.fuser.add_absolute(time.time(), dx, dy, dz, self.yaw, covariance=cov)
        self.x, self.y, self.z = dx, dy, dz
        self.last_confidence = conf

        print(f"[INIT] Start at x={dx:.2f}, y={dy:.2f}, z={dz:.2f}, yaw={self.yaw:.2f}")

    def run(self):
        """Main loop: schedule OF/VIO and VPS, fuse, publish to MAVLink, and optionally render UI."""
        vps_dt = 1.0 / self.cfg["vision"]["vps"].get("rate_hz", 1)
        vio_dt_cfg = 1.0 / self.cfg["vision"]["vio"].get("flow_rate_hz", 30)

        try:
            while True:
                loop_start_mono = self._mono()

                frame_data = self._read_frame()
                imu_latest = self._read_imu()
                altitude = self._get_altitude()

                # Skip iteration if no frame
                if frame_data is None:
                    continue

                # Optical flow + VIO prediction (at vio rate)
                of_confidence = 0.0
                now_mono = self._mono()
                # If adaptive enabled, recompute schedule intervals from current targets
                if self.adaptive_enabled:
                    vio_dt_cfg = 1.0 / max(self.vio_rate_min, min(self.vio_rate_max, self._vio_rate_hz))
                    vps_dt = 1.0 / max(self.vps_rate_min, min(self.vps_rate_max, self._vps_rate_hz))

                measured_dt = now_mono - self._last_vio_time_mono
                if measured_dt >= vio_dt_cfg:
                    of_output = self._process_optical_flow(frame_data.image, altitude)
                    if of_output is None:
                        logging.info("Optical flow update failed")
                        self.of_fail_count += 1
                    else:
                        of_confidence = of_output.confidence
                        self._last_of_conf = of_confidence
                        if of_confidence <= 1e-5:
                            self.of_fail_count += 1
                        else:
                            # reset failure streak on valid update
                            self.of_fail_count = 0

                        # Interpolate IMU to frame timestamp if possible
                        imu_interp = self._interpolate_imu_to(frame_data.timestamp)
                        imu_for_predict = imu_interp if imu_interp is not None else imu_latest
                        # Prefer RAW_IMU if available from MAVLink cache
                        try:
                            raw = self.mav.latest_raw_imu()
                            if raw is not None:
                                imu_for_predict = IMUData(timestamp=raw.get("timestamp", time.time()), accel=np.array([raw.get("ax",0), raw.get("ay",0), raw.get("az",0)]), gyro=np.array([raw.get("gx",0), raw.get("gy",0), raw.get("gz",0)]))
                        except Exception:
                            logging.debug("[IMU] latest_raw_imu() raised; using interpolated/latest IMU")

                        # Predict with measured dt
                        self.imu_vio.predict(imu_for_predict, dt=measured_dt)

                        # VIO correction: position-like or velocity-like depending on flag
                        if self.use_flow_as_velocity and measured_dt > 1e-3:
                            vx, vy = of_output.dx / measured_dt, of_output.dy / measured_dt
                            self.imu_vio.update_velocity(vx, vy, confidence=max(1e-3, of_confidence))
                        else:
                            self.imu_vio.update_flow(of_output.dx, of_output.dy, altitude, confidence=max(1e-3, of_confidence))

                        # Micro ZUPT/AUPT
                        if imu_for_predict is not None:
                            self.imu_vio.zero_velocity_update(imu_for_predict)

                        # Optional yaw update from ATTITUDE directly into EKF
                        if self.yaw_update_from_attitude:
                            yaw_att = self._get_attitude_yaw()
                            if yaw_att is not None:
                                try:
                                    self.imu_vio.update_yaw(yaw_att, confidence=1.0)
                                except Exception:
                                    pass

                        # Relative motion from OF (meters)
                        vio_dx, vio_dy = of_output.dx, of_output.dy

                        self._last_vio_time_mono = now_mono
                        if of_confidence > float(self.cfg["vision"]["flow"].get("min_of_confidence", 0.1)):
                            self._last_of_ok_mono = now_mono
                else:
                    vio_dx = vio_dy = None

                # VPS fix and smoothing (at VPS rate)
                smoothed_vps = VPSFix(self.x, self.y, self.z, self.yaw, 0.0)
                
                if self._mono() - self._last_vps_time_mono >= vps_dt:
                    vps_fix = self._process_vps(frame_data.image, altitude)
                    if vps_fix:
                        if self.main_loop_vps_smoothing:
                            # Legacy path: compute additional smoothing in main loop
                            smoothed_vps = self._compute_smoothed_vps()
                            self.last_confidence = smoothed_vps.confidence
                        else:
                            # Use the most recent fix from VPS buffer (already smoothed inside VPS)
                            smoothed_vps = self.vps_buffer[-1] if self.vps_buffer else VPSFix(vps_fix.x, vps_fix.y, vps_fix.z, vps_fix.yaw, vps_fix.confidence)
                            self.last_confidence = smoothed_vps.confidence

                        self._last_vps_time_mono = self._mono()
                    else:
                        # No VPS fix, increment low confidence counter
                        self.low_conf_counter += 1
                        smoothed_vps = VPSFix(self.x, self.y, self.z, self.yaw, 0.0)
                        self.last_confidence = 0.0

                # Watchdogs (OF/VPS/IMU timeouts and recovery actions)
                noww = time.time()
                noww_mono = self._mono()
                if (noww_mono - self._last_of_ok_mono) > self.of_timeout and self.of_fail_count >= self.of_fail_threshold:
                    logging.info(f"[WD] Optical flow sustained failure ({self.of_fail_count} frames). Resetting tracker.")
                    try:
                        self.optical_vio.reset()
                    except Exception:
                        logging.debug("[WD] Optical flow reset raised; continuing")
                    self.of_fail_count = 0
                    self._last_of_ok_mono = noww_mono
                if (noww_mono - self._last_vps_ok_mono) > self.vps_timeout:
                    logging.info("[WD] VPS timeout. Will down-weight VPS until next fix.")
                    # Set last confidence to 0 to increase covariance temporarily
                    self.last_confidence = 0.0
                    self._last_vps_ok_mono = noww_mono
                if (noww_mono - self._last_imu_ok_mono) > self.imu_timeout:
                    logging.info("[WD] IMU timeout. Holding prediction until IMU returns.")

                # Adaptive rate control
                self._adjust_rates(of_confidence, self.last_confidence)

                # Add measurements to factor graph
                self._add_to_factor_graph(vio_dx, vio_dy, smoothed_vps, of_confidence)

                # Optimize factor graph and get current pose
                self.fuser.optimize()
                pose = self.fuser.get_current_pose()
                if pose:
                    self.x, self.y, self.z, self.yaw = pose

                # Evaluation CSV logging
                if self._csv_writer is not None:
                    truth_x = truth_y = truth_z = truth_yaw = np.nan
                    if self.enable_truth:
                        # Prefer LOCAL_POSITION_NED (NED -> convert to ENU)
                        msg_lpn = self.mav.conn.recv_match(type="LOCAL_POSITION_NED", blocking=False)
                        if msg_lpn:
                            # NED: x=N, y=E, z=D; ENU: x=E, y=N, z=U
                            truth_x = float(msg_lpn.y)  # East
                            truth_y = float(msg_lpn.x)  # North
                            truth_z = float(-msg_lpn.z) # Up
                        else:
                            # Fallback: GLOBAL_POSITION_INT placeholders (not true ENU meters)
                            msg_gps = self.mav.conn.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
                            if msg_gps:
                                truth_x = msg_gps.lon / 1e7
                                truth_y = msg_gps.lat / 1e7
                                truth_z = msg_gps.relative_alt / 1000.0
                        # Yaw from ATTITUDE if present
                        msg_att = self.mav.conn.recv_match(type="ATTITUDE", blocking=False)
                        if msg_att:
                            truth_yaw = msg_att.yaw
                    self._csv_writer.writerow([
                        noww, self.x, self.y, self.z, self.yaw,
                        of_confidence, self.last_confidence, self._vio_rate_hz, self._vps_rate_hz,
                        truth_x, truth_y, truth_z, truth_yaw
                    ])

                # Publish to MAVLink (dynamic covariance)
                pos_cov = max(0.01, (1.0 - min(1.0, max(0.0, self.last_confidence))) * 1.0)
                yaw_cov = max(0.01, (1.0 - min(1.0, max(0.0, self.last_confidence))) * 0.2)
                self.mav.send_vision_position(
                    self.x, self.y, self.z, yaw=self.yaw, covariance=pos_cov, yaw_sigma=yaw_cov
                )

                # Optionally publish vision speed estimate (use OF-based velocity if available)
                if hasattr(self.optical_vio, 'last_vel') and self.optical_vio.last_vel is not None:
                    vx, vy = float(self.optical_vio.last_vel[0]), float(self.optical_vio.last_vel[1])
                    self.mav.send_vision_speed(vx, vy, 0.0)

                # Periodic summary
                if (time.time() - self._last_summary_time) >= self._summary_interval:
                    self._log_summary(
                        fps=round(self.last_fps, 1),
                        of_conf=round(of_confidence, 2),
                        vps_conf=round(self.last_confidence, 2),
                        x=round(self.x, 2), y=round(self.y, 2), z=round(self.z, 2),
                        vio_rate=round(self._vio_rate_hz, 2), vps_rate=round(self._vps_rate_hz, 2),
                        cam_fps=(round(self.camera.get_fps_estimate(),1) if hasattr(self.camera,'get_fps_estimate') and self.camera.get_fps_estimate() is not None else None),
                        of_fail_streak=self.of_fail_count
                    )
                    self._last_summary_time = time.time()

                # Visualization overlays (skip in headless mode)
                if not self.headless:
                    map_overlay = self._draw_map_overlay(self.x, self.y, self.yaw, self.last_confidence)
                    camera_overlay = self._draw_camera_overlay(frame_data.image, altitude, self.x, self.y, self.z, self.yaw, self.last_confidence)
                    cv2.imshow("Camera Feed", camera_overlay)
                    cv2.imshow("Map Position", map_overlay)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('t'):
                        self.ui_show_tracks = not self.ui_show_tracks
                    elif key == ord('m'):
                        self.ui_show_mask = not self.ui_show_mask
                    elif key == ord('b'):
                        self.ui_show_bars = not self.ui_show_bars
                    elif key == ord('n'):
                        self.ui_show_scale = not self.ui_show_scale
                    elif key == ord('r'):
                        try:
                            self.optical_vio.reset()
                        except Exception:
                            pass
                    elif key == ord('p'):
                        try:
                            ts = int(time.time()*1000)
                            cv2.imwrite(os.path.join(self.rec_dir, f"shot_cam_{ts}.png"), camera_overlay)
                            cv2.imwrite(os.path.join(self.rec_dir, f"shot_map_{ts}.png"), map_overlay)
                        except Exception:
                            pass

                # FPS (loop rate measured by monotonic clock)
                end_time_mono = self._mono()
                dt_loop = end_time_mono - loop_start_mono
                if dt_loop > 0:
                    self.last_fps = 1.0 / dt_loop

        except KeyboardInterrupt:
            print("[EXIT] Stopping navigation pipeline...")
        finally:
            # Graceful teardown
            self.camera.stop()
            self.mav.stop()
            if self._csv_file:
                try:
                    self._csv_file.close()
                except Exception:
                    pass
            if self._rec_imu_file:
                try:
                    self._rec_imu_file.close()
                except Exception:
                    pass
            if self._rec_vid is not None:
                try:
                    self._rec_vid.release()
                except Exception:
                    pass
            cv2.destroyAllWindows()


if __name__ == "__main__":
    nav = NavigationPipeline()
    nav.initialize()
    nav.run()


