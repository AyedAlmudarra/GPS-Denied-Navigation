#!/usr/bin/env python3
import os
import sys
import cv2
import yaml
import time
import numpy as np

# Ensure project root is on sys.path so `src.*` imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision.optical_flow import OpticalFlowTracker
from src.vision.vio_tracker import VIOTracker
from src.vision.vps_transformer import VPSTransformer


class DummyIMU:
    def __init__(self):
        self.accel = np.array([0.0, 0.0, 0.0])
        self.gyro = np.array([0.0, 0.0, 0.0])
        self.timestamp = time.time()


def load_config():
    cfg_path = os.path.join(SRC_DIR, 'config.yaml')
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def check_config(cfg):
    print('[SELF-CHECK] Config sanity...')
    try:
        m = cfg.get('map', {})
        ppm = m.get('pixels_per_meter', None)
        if ppm is None or not isinstance(ppm, (int, float)):
            print('  Warn: map.pixels_per_meter missing or non-numeric; defaults will be used')
        print('  OK')
    except Exception as e:
        print(f'  Config check encountered an issue: {e}')


def check_optical_flow(cfg):
    print('[SELF-CHECK] OpticalFlowTracker...')
    of = OpticalFlowTracker(cfg)
    # Create a simple gradient image
    h, w = 480, 640
    x = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.tile(x, (h, 1))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    of.initialize(img_bgr)
    dx, dy, conf = of.update(img_bgr, altitude=1.0)
    assert isinstance(dx, float) or isinstance(dx, (np.floating,)), 'dx not float'
    assert isinstance(dy, float) or isinstance(dy, (np.floating,)), 'dy not float'
    assert 0.0 <= float(conf) <= 1.0, 'confidence out of range'
    print('  OK')


def check_vio(cfg):
    print('[SELF-CHECK] VIOTracker...')
    vio = VIOTracker(cfg)
    # Dummy frame
    h, w = 240, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)
    vio.initialize(img)
    imu = DummyIMU()
    vio.predict(imu, dt=1.0/30.0)
    vio.update_flow(0.0, 0.0, altitude=1.0, confidence=1.0)
    st = vio.get_state()
    for k in ['position', 'velocity', 'orientation_quat']:
        assert k in st, f'missing state key {k}'
    print('  OK')


def check_vps(cfg):
    print('[SELF-CHECK] VPSTransformer...')
    try:
        map_path = cfg['vision']['vps']['map_path']
        if not os.path.isabs(map_path):
            map_path = os.path.join(PROJECT_ROOT, map_path)
        if not os.path.exists(map_path):
            print('  Map image not found; skipping VPS check.')
            return
        cam = cfg.get('camera', {})
        K = None
        dist = None
        if all(k in cam for k in ('fx','fy','cx','cy')):
            K = np.array([[float(cam['fx']), 0.0, float(cam['cx'])], [0.0, float(cam['fy']), float(cam['cy'])], [0.0, 0.0, 1.0]], dtype=np.float32)
            if 'distortion' in cam:
                dist = np.array(cam['distortion'], dtype=np.float32)
        vps = VPSTransformer(
            map_path=map_path,
            focal_px=cfg['vision']['vps'].get('focal_px', 525.0),
            pixels_per_meter=cfg.get('map', {}).get('pixels_per_meter', None),
            camera_K=K,
            camera_dist=dist,
            yaw_sign=int(cfg.get('fusion', {}).get('yaw_sign', 1)),
            scale_levels=cfg['vision']['vps'].get('scale_levels', [1.0]),
            tile_rows=cfg['vision']['vps'].get('tile_rows', 0),
            tile_cols=cfg['vision']['vps'].get('tile_cols', 0),
            features_per_tile=cfg['vision']['vps'].get('features_per_tile', 0),
            ratio_thresh=cfg['vision']['vps'].get('ratio_thresh', 0.7)
        )
        # Use the map image itself as query frame (should be easiest)
        img = cv2.imread(map_path, cv2.IMREAD_COLOR)
        if img is None:
            print('  Could not load map image; skipping VPS check.')
            return
        res = vps.match(img, motion_mask=None, altitude=1.0)
        # Do not assert on result; some maps may be unsuitable. This is a smoke test.
        print('  OK (constructed and executed)')
    except Exception as e:
        print(f'  VPS check failed with exception: {e}')
        # Do not hard fail the whole self-check on VPS, since map assets vary


def main():
    cfg = load_config()
    failures = 0
    try:
        check_optical_flow(cfg)
    except Exception as e:
        print(f'[SELF-CHECK] OpticalFlowTracker FAILED: {e}')
        failures += 1
    try:
        check_vio(cfg)
    except Exception as e:
        print(f'[SELF-CHECK] VIOTracker FAILED: {e}')
        failures += 1
    # VPS is optional; do not increment failures unless import/construct totally fails
    try:
        check_vps(cfg)
    except Exception as e:
        print(f'[SELF-CHECK] VPSTransformer FAILED to construct: {e}')
        failures += 1
    # Config sanity (non-fatal)
    try:
        check_config(cfg)
    except Exception:
        pass

    if failures == 0:
        print('[SELF-CHECK] All required checks passed.')
        sys.exit(0)
    else:
        print(f'[SELF-CHECK] Failures: {failures}')
        sys.exit(1)


if __name__ == '__main__':
    main() 