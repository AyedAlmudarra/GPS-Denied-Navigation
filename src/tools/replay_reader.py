#!/usr/bin/env python3
"""
replay_reader
-------------
Lightweight playback tool for recorded frames saved by the main loop recorder.

- Input directory should contain files like frame_<timestamp>.png and optionally imu.csv
- Plays back frames at a target FPS, either showing a window or sleeping headless
"""
import argparse
import os
import time
import cv2


def list_frames(dir_path):
    """Return sorted paths of recorded frame_*.png files in the directory."""
    files = sorted([f for f in os.listdir(dir_path) if f.startswith('frame_')])
    return [os.path.join(dir_path, f) for f in files]


def main():
    """CLI: iterate over recorded frames at a given FPS; optionally display."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='Recording directory (contains frame_*.png and imu.csv)')
    ap.add_argument('--fps', type=float, default=20.0, help='Playback FPS')
    ap.add_argument('--show', action='store_true', help='Show frames in a window')
    args = ap.parse_args()

    frames = list_frames(args.dir)
    if not frames:
        print('No frames found')
        return

    dt = 1.0 / args.fps if args.fps > 0 else 0.05
    for fp in frames:
        img = cv2.imread(fp)
        if img is None:
            continue
        if args.show:
            cv2.imshow('Replay', img)
            if cv2.waitKey(int(dt*1000)) & 0xFF == ord('q'):
                break
        else:
            time.sleep(dt)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 