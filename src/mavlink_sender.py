#!/usr/bin/env python3
"""
MAVLink Sender for GPS-denied Visual Navigation

- Sends VISION_POSITION_ESTIMATE with yaw + covariance
- Optionally reads IMU (RAW_IMU) or ATTITUDE messages
- Supports dynamic covariance and optional VISION_SPEED_ESTIMATE
"""

import time
import logging
from pymavlink import mavutil

logging.basicConfig(level=logging.INFO, format='[MAVLink] %(asctime)s - %(message)s')

class MAVLinkSender:
    def __init__(self, connection_str="tcp:127.0.0.1:5762", vision_rate_hz=10, heartbeat_timeout_sec=10):
        """
        Initialize MAVLink connection to ArduPilot.
        """
        logging.info(f"Connecting to MAVLink at {connection_str}")
        # Validate connection string scheme
        try:
            scheme = str(connection_str).split(":", 1)[0].lower()
            allowed = {"tcp", "udp", "udpin", "udpout", "serial"}
            if scheme not in allowed:
                logging.warning(f"[MAVLink] Unrecognized connection scheme '{scheme}'. Expected one of {allowed}.")
        except Exception as e:
            logging.debug(f"[MAVLink] Could not parse connection string: {e}")
        self.conn = mavutil.mavlink_connection(connection_str)
        self.vision_rate_hz = vision_rate_hz
        self.start_time = time.time()
        self.last_raw_imu = None
        self.last_attitude = None

        # Heartbeat wait with timeout/backoff
        ok = self._wait_heartbeat_with_timeout(timeout_sec=heartbeat_timeout_sec)
        if ok:
            logging.info(f"Heartbeat received from system {self.conn.target_system}")
        else:
            logging.warning("Heartbeat timeout; proceeding without confirmation (will keep polling)")

    def _wait_heartbeat_with_timeout(self, timeout_sec=10):
        deadline = time.time() + float(timeout_sec)
        # Try native wait_heartbeat with timeout if available
        try:
            self.conn.wait_heartbeat(timeout=timeout_sec)
            return True
        except Exception:
            pass
        # Fallback: poll non-blocking until timeout
        while time.time() < deadline:
            msg = self.conn.recv_match(type='HEARTBEAT', blocking=False)
            if msg:
                return True
            time.sleep(0.1)
        return False

    def set_message_interval(self, msg_id, rate_hz):
        """
        Request ArduPilot to send specified message ID at the desired rate.
        Example: GLOBAL_POSITION_INT = 33
        """
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            int(1e6 / rate_hz),  # microseconds interval
            0, 0, 0, 0, 0
        )
        logging.info(f"Requested MSG ID {msg_id} at {rate_hz} Hz")

    def start(self):
        """
        Start streaming specific MAVLink messages from the autopilot.
        """
        # You can request RAW_IMU (27) or GLOBAL_POSITION_INT (33)
        self.set_message_interval(27, 10)  # RAW_IMU
        self.set_message_interval(30, 20)  # ATTITUDE faster for yaw prior
        self.set_message_interval(33, 10)  # GLOBAL_POSITION_INT at 10 Hz

    def stop(self):
        """
        Clean shutdown of MAVLink connection.
        """
        self.conn.close()
        logging.info("Closed MAVLink connection")

    def _make_cov21(self, pos_variance=1.0, yaw_variance=0.1):
        """
        Build a 21-element covariance vector for VISION_POSITION_ESTIMATE.
        Inputs are variances (not standard deviations).
        Fills diagonals for x,y,z and yaw; others left as zero for simplicity.
        """
        cov = [0.0] * 21
        # Diagonals for x,y,z (positions)
        cov[0] = pos_variance
        cov[6] = pos_variance
        cov[11] = pos_variance
        # Yaw variance is last element (index 20) per MAVLink spec
        cov[20] = yaw_variance
        return cov

    def send_vision_position(self, x, y, z, covariance=1.0, yaw=0.0, yaw_sigma=0.1):
        """
        Send vision position estimate to ArduPilot.

        Params:
            x, y, z    - position in meters
            yaw        - heading in radians
            covariance - base variance for positions (used to build 21-vector)
            yaw_sigma  - variance for yaw element
        """
        t_boot_ms = int((time.time() - self.start_time) * 1000) & 0xFFFFFFFF
        cov_vec = self._make_cov21(pos_variance=covariance, yaw_variance=yaw_sigma)
        self.conn.mav.vision_position_estimate_send(
            t_boot_ms,
            x, y, z,
            0.0, 0.0, yaw,  # roll, pitch, yaw
            cov_vec
        )
        logging.debug(f"Sent VISION_POSITION_ESTIMATE: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.2f}, cov_pos={covariance:.3f}, cov_yaw={yaw_sigma:.3f}")

    def send_vision_speed(self, vx, vy, vz, covariance=1.0):
        """
        Optionally send vision speed estimate if available.
        """
        t_boot_ms = int((time.time() - self.start_time) * 1000) & 0xFFFFFFFF
        # MAVLink VISION_SPEED_ESTIMATE only takes vx,vy,vz (m/s)
        self.conn.mav.vision_speed_estimate_send(
            t_boot_ms,
            vx, vy, vz
        )
        logging.debug(f"Sent VISION_SPEED_ESTIMATE: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")

    def poll(self, max_msgs=10):
        """Poll up to max_msgs non-blocking messages and update caches."""
        for _ in range(max_msgs):
            msg = self.conn.recv_match(type=['RAW_IMU', 'ATTITUDE'], blocking=False)
            if not msg:
                break
            if msg.get_type() == 'RAW_IMU':
                self.last_raw_imu = {
                    "type": "RAW_IMU",
                    "ax": msg.xacc / 1000.0,  # mg -> g
                    "ay": msg.yacc / 1000.0,
                    "az": msg.zacc / 1000.0,
                    "gx": msg.xgyro / 1000.0,  # mrad/s -> rad/s
                    "gy": msg.ygyro / 1000.0,
                    "gz": msg.zgyro / 1000.0,
                    "timestamp": msg.time_usec / 1e6 if hasattr(msg, 'time_usec') else time.time(),
                }
            elif msg.get_type() == 'ATTITUDE':
                self.last_attitude = {
                    "type": "ATTITUDE",
                    "roll": msg.roll,
                    "pitch": msg.pitch,
                    "yaw": msg.yaw,
                    "timestamp": msg.time_boot_ms / 1e3 if hasattr(msg, 'time_boot_ms') else time.time(),
                }

    def latest_raw_imu(self):
        self.poll()
        return self.last_raw_imu

    def latest_attitude(self):
        self.poll()
        return self.last_attitude

    def latest_imu(self):
        """
        Backward-compatible: returns a dict of RAW_IMU or ATTITUDE if present, preferring RAW_IMU.
        """
        self.poll()
        if self.last_raw_imu is not None:
            return self.last_raw_imu
        return self.last_attitude


if __name__ == "__main__":
    # Example usage
    mav = MAVLinkSender("tcp:127.0.0.1:5762")
    mav.start()
    try:
        while True:
            # Send dummy vision position
            mav.send_vision_position(x=1.0, y=2.0, z=-1.0, yaw=0.5)

            # Print any new IMU data
            imu = mav.latest_raw_imu()
            if imu:
                logging.info(f"Received RAW_IMU: {imu}")

            time.sleep(1.0 / mav.vision_rate_hz)
    except KeyboardInterrupt:
        logging.info("Stopping MAVLink sender...")
        mav.stop()

