#!/usr/bin/env python3
"""
GazeboCamera via OpenCV GStreamer backend with timestamps and debug logging.
Matches your gst-launch pipeline exactly.
No ROS dependencies.
"""

import cv2
import threading
import time
import logging

class GazeboCamera:
    def __init__(self, udp_port=5600, max_buffers=2, debug=False):
        self.debug = debug
        self.pipeline = (
            f'udpsrc port={udp_port} '
            'caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! '
            'rtpjitterbuffer ! '
            'rtph264depay ! '
            'avdec_h264 ! '
            'videoconvert ! '
            f'appsink sync=false max-buffers={max_buffers} drop=true'
        )

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open GStreamer pipeline:\n" + self.pipeline)

        self._frame = None
        self._timestamp = None
        self._lock = threading.Lock()
        self._running = False
        # Tracking for size and FPS (EMA)
        self._frame_size = None  # (width, height)
        self._fps_ema = None
        self._last_ts_read = None

        if self.debug:
            logging.basicConfig(level=logging.DEBUG, format='[GazeboCamera] %(message)s')

    def start(self):
        """Start background thread to continuously read frames."""
        if self._running:
            return
        self._running = True

        def reader():
            while self._running:
                try:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        time.sleep(0.005)
                        continue
                    ts_now = time.time()
                    with self._lock:
                        self._frame = frame
                        self._timestamp = ts_now
                        self._frame_size = (frame.shape[1], frame.shape[0])
                        if self._last_ts_read is not None:
                            dt = ts_now - self._last_ts_read
                            if dt > 0:
                                fps_inst = 1.0 / dt
                                alpha = 0.2
                                self._fps_ema = fps_inst if self._fps_ema is None else (1.0 - alpha) * self._fps_ema + alpha * fps_inst
                        self._last_ts_read = ts_now
                    if self.debug:
                        logging.debug(f"Frame received: shape={frame.shape}")
                except Exception as e:
                    logging.error(f"Camera read error: {e}")
                    time.sleep(0.1)

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()

    def read(self, timeout=None):
        """
        Return the latest frame and timestamp (BGR, float seconds) or (None, None).
        If timeout (seconds) provided, block up to that time for first frame.
        """
        start = time.time()
        while True:
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
                ts = self._timestamp
            if frame is not None or timeout is None:
                return frame, ts
            if timeout and (time.time() - start) >= timeout:
                return None, None
            time.sleep(0.001)

    def get_frame_size(self):
        with self._lock:
            return self._frame_size

    def get_fps_estimate(self):
        with self._lock:
            return self._fps_ema

    def stop(self):
        """Stop the reader thread and release resources."""
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join()
        self.cap.release()


if __name__ == "__main__":
    cam = GazeboCamera(udp_port=5600, debug=True)
    cam.start()
    cv2.namedWindow("LiveCamera", cv2.WINDOW_NORMAL)
    try:
        while True:
            frame, ts = cam.read(timeout=1.0)
            if frame is not None:
                cv2.imshow("LiveCamera", frame)
                if cam.debug:
                    print(f"[MAIN] Frame timestamp: {ts:.6f}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()

