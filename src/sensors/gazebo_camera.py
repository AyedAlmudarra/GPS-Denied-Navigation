#!/usr/bin/env python3
"""
GazeboCamera via OpenCV GStreamer backend with timestamps and debug logging.
Matches the `gst-launch` pipeline used for local testing.
No ROS dependencies.

Notes
-----
- Requires an OpenCV build with GStreamer support (system `libopencv-dev`).
  Pip wheels of `opencv-python` generally do not include GStreamer; use the
  system package on Ubuntu.
- This class runs a single background thread that continuously reads frames from
  the pipeline and exposes get/read helpers with a lock. The most recent frame
  and a wall-clock timestamp (`time.time()`) are cached.
- `appsink` is configured with `drop=true` and a small `max-buffers` to avoid
  latency buildup if the consumer is slower than the producer.
"""

import cv2
import threading
import time
import logging

class GazeboCamera:
    def __init__(self, udp_port=5600, max_buffers=2, debug=False):
        """Create a Gazebo camera client using OpenCV+GStreamer.

        Parameters
        ----------
        udp_port : int
            UDP port for the RTP/H.264 stream (Gazebo publishes here).
        max_buffers : int
            Max queued buffers at appsink before dropping to keep latency low.
        debug : bool
            If true, enable verbose logging.
        """
        self.debug = debug
        # GStreamer pipeline: udpsrc -> jitterbuffer -> depay -> h264 decoder -> BGR -> appsink
        self.pipeline = (
            f'udpsrc port={udp_port} '
            'caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! '
            'rtpjitterbuffer ! '
            'rtph264depay ! '
            'avdec_h264 ! '
            'videoconvert ! '
            f'appsink sync=false max-buffers={max_buffers} drop=true'
        )

        # OpenCV VideoCapture using CAP_GSTREAMER backend
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open GStreamer pipeline:\n" + self.pipeline)

        # Shared state guarded by a lock
        self._frame = None
        self._timestamp = None
        self._lock = threading.Lock()
        self._running = False
        # Tracking for size and FPS (EMA)
        self._frame_size = None  # (width, height)
        self._fps_ema = None     # exponential moving average of source FPS
        self._last_ts_read = None

        if self.debug:
            logging.basicConfig(level=logging.DEBUG, format='[GazeboCamera] %(message)s')

    def start(self):
        """Start background reader thread.

        The thread continuously calls `cap.read()` and updates the latest frame,
        timestamp, and FPS EMA. If a read fails, it briefly sleeps and tries again.
        """
        if self._running:
            return
        self._running = True

        def reader():
            while self._running:
                try:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        # Avoid a tight loop on transient failures
                        time.sleep(0.005)
                        continue
                    ts_now = time.time()
                    with self._lock:
                        self._frame = frame
                        self._timestamp = ts_now
                        self._frame_size = (frame.shape[1], frame.shape[0])
                        # FPS EMA based on inter-arrival time
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
        """Return the latest frame and timestamp (BGR, float seconds) or (None, None).

        Parameters
        ----------
        timeout : float | None
            If provided, block up to `timeout` seconds for the first frame.
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
        """Return last observed frame size as (width, height) or None."""
        with self._lock:
            return self._frame_size

    def get_fps_estimate(self):
        """Return a rough FPS estimate (EMA) of the source or None."""
        with self._lock:
            return self._fps_ema

    def stop(self):
        """Stop reader thread and release the VideoCapture resource."""
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join()
        self.cap.release()


if __name__ == "__main__":
    # Simple manual test window
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

