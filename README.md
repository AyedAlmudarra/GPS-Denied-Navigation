## GPS-Denied Navigation

End-to-end pipeline that fuses high-rate optical-flow VIO and low-rate global Visual Place Recognition (VPS), optimizes with a factor graph, and streams vision poses to ArduPilot via MAVLink. Designed to run without ROS, integrate with SITL + Gazebo, and expose practical knobs for performance and robustness.

### Key features
- **Sensors**: Gazebo H.264 UDP camera client (OpenCV GStreamer); MAVLink IMU/yaw inputs
- **Local motion (VIO)**: ORB-seeded LK optical flow with FB check, affine RANSAC + MAD gating, ZUPT-like damping, EKF predict/correct (position or velocity mode), nonlinear yaw update
- **Global fixes (VPS)**: SIFT pyramid + optional tiling, FLANN ratio test, RANSAC with degeneracy checks, confidence-weighted temporal smoothing, motion mask support
- **Fusion**: GTSAM-based factor graph when available, adaptive covariances, per-factor robust losses, yaw prior from ATTITUDE, continuity prior; buffer-only fallback otherwise
- **Scheduling & resilience**: Adaptive rates (EMA + hysteresis), watchdogs (OF/VPS/IMU), monotonic timing, IMU interpolation
- **Outputs**: MAVLink VISION_POSITION_ESTIMATE (+ optional VISION_SPEED_ESTIMATE), CSV logs, optional recording and UI overlays

---

## Repository structure
```
gps_denied_nav/
├── docs/                      # Annotated docs and notes
├── models/                    # Gazebo models
├── worlds/                    # Gazebo world/SDF
├── src/
│   ├── base_structures.py     # Legacy helpers (not used by runtime)
│   ├── config.yaml            # Central configuration
│   ├── main_loop.py           # Runtime orchestrator (entry point)
│   ├── fusion/
│   │   └── factor_graph.py    # GTSAM fusion; buffered fallback
│   ├── sensors/
│   │   └── gazebo_camera.py   # OpenCV GStreamer UDP camera client
│   ├── vision/
│   │   ├── optical_flow.py    # OpticalFlowTracker (ORB+LK+FB, affine RANSAC, MAD)
│   │   ├── vio_tracker.py     # EKF (p, v, q, biases) + nonlinear yaw update
│   │   └── vps_transformer.py # SIFT VPS (pyramid/tiling, RANSAC, smoothing)
│   └── tools/
│       ├── metrics_cli.py     # Analyze CSV logs; optional plots
│       ├── replay_reader.py   # Replay recorded frames
│       └── self_check.py      # Smoke tests for OF/VIO/VPS
├── requirements.txt           # Python dependencies
├── setup.sh                   # Gazebo plugin build helper (optional)
└── README.md                  # This file
```

---

## Architecture and dataflow
- Camera frames (Gazebo RTP H.264) → `sensors/gazebo_camera.py`
- Optical flow → `vision/optical_flow.py` → (dx, dy, conf) and motion mask
- EKF predict/correct → `vision/vio_tracker.py` (IMU predict; flow or velocity update; yaw update)
- VPS global fix → `vision/vps_transformer.py` (x, y, z, yaw, conf)
- Factor graph fuse → `fusion/factor_graph.py` (relative + absolute + yaw prior; optimize)
- MAVLink publish → `mavlink_sender.py` (position/yaw + optional speed)
- Orchestration (timing, rates, watchdogs, recording, CSV/UI) → `main_loop.py`

---

## Requirements
- Python: 3.9+ (recommend 3.10/3.11)
- OS packages (Ubuntu 22.04):
  - OpenCV dev, GStreamer, gstreamer plugins (bad/libav/gl), toolchain for Gazebo plugins
  - Example:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libopencv-dev gstreamer1.0-tools \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl
```
- Python deps:
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
- ArduPilot SITL + Gazebo (external):
  - Build SITL and launch `sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map`
  - Launch Gazebo world (see `worlds/`) and enable camera streaming (see below)

---

## Gazebo camera streaming (example)
- Launch world (adjust to your SDF/world):
```bash
gz sim -v4 -r worlds/iris_runway_ngps.sdf
```
- Enable camera streaming:
```bash
gz topic -t /world/iris_runway_ngps/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image/enable_streaming \
  -m gz.msgs.Boolean -p "data: true"
```
- Quick test pipeline (receiver):
```bash
gst-launch-1.0 -v udpsrc port=5600 \
  caps="application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96" \
  ! rtpjitterbuffer ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

---

## Installation
```bash
git clone https://github.com/your-username/gps_denied_nav.git
cd gps_denied_nav
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration reference (`src/config.yaml`)
- camera
  - `udp_port`: UDP port for RTP H.264 stream
  - `fx, fy, cx, cy`: intrinsics (optional)
  - `distortion`: 5-coeff vector (optional)
- vision.flow
  - `max_corners, quality_level, min_distance, block_size, win_size, max_level`: ORB/LK params
  - `min_texture_var`: Laplacian variance threshold for texture sufficiency
  - `zupt_threshold`: zero-velocity damping threshold (in meters per update)
  - `focal_length_px`: focal length in pixels (px→m conversion)
  - `ransac_reproj_thresh, residual_scale, lk_error_thresh, roi_pad_px`
  - `fb_check_enabled, fb_error_thresh`: forward–backward LK gating
  - `grid_rows, grid_cols, features_per_cell`: grid seeding (0 disables)
  - `min_of_confidence`: min confidence considered valid for relative factor
  - `downscale`: 1.0 native; <1.0 processes downscaled frames for speed (flow rescaled)
- vision.vio
  - `flow_rate_hz`: nominal OF rate (used for scheduling); adaptive may override
  - `min_cov, max_cov`: bounds for covariance clamping
  - `yaw_update_from_attitude`: if true, fuses ATTITUDE yaw directly into EKF each cycle
- vision.vps
  - `map_path`: path to grayscale map
  - `nfeatures, min_matches, focal_px, rate_hz, max_jump_m`
  - `scale_levels`: e.g. `[1.0, 0.75, 0.5]`
  - `tile_rows, tile_cols, features_per_tile`: enable tiling when >0
  - `ratio_thresh`: Lowe’s ratio test threshold
  - `use_motion_mask`: 0-value regions ignored; supplied by OF outlier mask
  - `early_score_threshold`: early-exit if score exceeds this (0 disables)
  - `main_loop_smoothing`: if true, adds an extra smoothing layer in `main_loop` (defaults false)
- fusion
  - `window_size`: sliding window number of timestamps
  - `abs_sigma, rel_sigma`: base 6D sigmas for absolute/relative factors (GTSAM)
  - `yaw_prior_sigma`: yaw prior sigma (radians)
  - `yaw_sign`: apply to yaw from ATTITUDE/VPS to correct handedness
  - `continuity_sigma`: loose between prior when no rel factor added
  - `robust_loss`: none|huber|cauchy|geman; `robust_delta`: scale
  - `adaptive_cov`: use dynamic covariances passed to add_* vs defaults
  - `use_se2`: planned option (currently uses Pose3 with z=0.0 when enabled)
- mavlink
  - `connection_str`: tcp/udp/udpin/udpout/serial endpoints
  - `vision_rate_hz`: send rate for vision messages
- mission
  - `takeoff_alt`: minimum altitude to start vision loop (m)
- map
  - `pixels_per_meter`: map scale for overlays and px→m fallback
- ui
  - `headless`: true disables UI windows
- logging
  - `summary_interval_sec`: cadence for summary logs
  - `eval_csv_path`: write per-iteration CSV when set
  - `enable_truth`: include LOCAL_POSITION_NED and ATTITUDE yaw as truth columns when present
- runtime.watchdogs
  - `of_timeout_sec, vps_timeout_sec, imu_timeout_sec, of_fail_threshold`
- runtime.adaptive_rates
  - `enabled`, min/max VIO/VPS Hz and confidence thresholds low/high
- runtime.opencv
  - `use_optimized, num_threads`: OpenCV global flags
- runtime.recording
  - `enabled, dir, save_frames, save_imu, save_video, video_fps, image_format`

---

## Run the pipeline
```bash
source venv/bin/activate
python3 src/main_loop.py
```
- The pipeline waits for `mission.takeoff_alt`, initializes OF and VIO, then runs:
  - OF update at adaptive VIO rate; EKF predict/correct; optional yaw from ATTITUDE
  - VPS at adaptive rate; smoothing; confidence-based jump gating
  - Factor graph fusion; MAVLink vision outputs
  - Optional CSV logging, recording, and UI overlays

### Tips
- Headless mode: set `ui.headless: true`
- Recording: enable `runtime.recording` and provide a directory
- CSV: set `logging.eval_csv_path: navigation_eval.csv`

---

## Tools
- `src/tools/self_check.py`
  - Runs smoke checks for OF/VIO; attempts VPS if map exists
  - Exit code 0 on required checks passing
- `src/tools/metrics_cli.py`
  - Usage: `python3 src/tools/metrics_cli.py --csv navigation_eval.csv --plot`
  - Prints row counts, confidence means, rough FPS, and RMSE if truth columns present
- `src/tools/replay_reader.py`
  - Playback of recorded images: `python3 src/tools/replay_reader.py --dir recordings/run_01 --fps 20 --show`

---

## Performance tuning
- Set `vision.flow.downscale: 0.5` on CPU-limited systems
- Reduce SIFT cost via `vision.vps.nfeatures`, `scale_levels`, `features_per_tile`
- Enable adaptive rates (`runtime.adaptive_rates.enabled: true`)
- Increase OpenCV threads (`runtime.opencv.num_threads`)
- Consider disabling `use_motion_mask` if unnecessary (saves mask resizes)

---

## Troubleshooting
- Camera won’t open: verify GStreamer packages and that RTP stream is enabled on the correct `udp_port`
- No VPS fixes: ensure `map_path` is correct and textured; adjust `nfeatures`, `ratio_thresh`; verify `pixels_per_meter`
- Vision pose noisy: increase robust loss; tune `min_of_confidence`, `max_jump_m`; verify focal/altitude scales
- Slow loop: enable `downscale`, reduce VPS workload, verify OpenCV threading
- MAVLink not connected: check `connection_str` scheme and endpoints; heartbeat logs

---

## Security and robustness
- Only MAVLink and UDP camera are consumed; validate endpoints; treat external streams as untrusted
- Watchdogs distinguish OF failure vs stillness; resets only on sustained failures
- VPS degeneracy checks reduce risk of spurious large jumps
- Replace silent exceptions with debug logs in critical paths

---

## Coordinate frames and truth
- VPS/vision positions are in a local planar ENU-like frame centered on the map image center
- Truth (if `enable_truth`) uses LOCAL_POSITION_NED converted to ENU in CSV:
  - `x=East(LOCAL_POSITION_NED.y), y=North(LOCAL_POSITION_NED.x), z=Up(-LOCAL_POSITION_NED.z)`
- Yaw from ATTITUDE is in radians; `fusion.yaw_sign` can flip if necessary

---

## License & credits
- Derived from GSoC 2024 High-Altitude Non-GPS Navigation efforts and community work
- License: MIT (unless otherwise stated in model/world assets)
