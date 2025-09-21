import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import logging

class VIOTracker:
    """
    Visual-Inertial Odometry tracker using Extended Kalman Filter (EKF).
    State vector: position (3), velocity (3), orientation quaternion (4), accel bias (3), gyro bias (3)
    State size: 16

    Integrates IMU data in prediction.
    Updates with optical flow deltas and optional VPS global pose fixes.
    Supports online bias estimation and zero-velocity/angular-rate detection.
    """

    def __init__(self, config):
        cfg = config["vision"]["vio"]
        self.dt = 1.0 / cfg["flow_rate_hz"]
        
        # Initialize biases to zero
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        
        
        # Initialize state vector [pos(3), vel(3), quat(4), ba(3), bg(3)]
        self.state = np.zeros(16)
        self.state[6:10] = np.array([0, 0, 0, 1])  # Initial quaternion (x,y,z,w)

        # Covariance matrix (16x16)
        self.P = np.eye(16) * 0.01

        # Process noise covariance
        self.Q = np.eye(16)
        self.Q[0:3, 0:3] *= 0.01    # Position noise
        self.Q[3:6, 3:6] *= 0.1     # Velocity noise
        self.Q[6:10, 6:10] *= 0.001 # Orientation noise
        self.Q[10:13, 10:13] *= 0.0001  # Accel bias noise
        self.Q[13:16, 13:16] *= 0.0001  # Gyro bias noise

        # Measurement noise covariance for optical flow position updates
        self.R_flow = np.eye(3) * 0.05
        # Measurement noise covariance for velocity updates (vx, vy)
        self.R_vel = np.eye(2) * 0.05

        # Measurement noise covariance for VPS position-only updates (xyz)
        self.R_vps = np.eye(3) * 0.1
        # Measurement noise for yaw (scalar); tune via config later if needed
        self.R_yaw = np.array([[0.05]], dtype=float)

        self.gravity = np.array([0, 0, 9.81])

        # Thresholds for zero-velocity and zero-angular-rate updates
        self.zv_thresh_acc = 0.1  # m/s^2
        self.zv_thresh_gyro = 0.05  # rad/s

        # Covariance clamp bounds
        self.min_var = 1e-6
        self.max_var = 1e3

        # Internal timing and gating
        self._last_timestamp = None
        # Chi-square thresholds (95% confidence)
        self._chi2_thresh_flow = 7.815  # dof=3
        self._chi2_thresh_vel = 5.991   # dof=2
        self._chi2_thresh_vps = 7.815   # dof=3 (xyz only now)
        self._chi2_thresh_yaw = 3.841   # dof=1

    def initialize(self, frame):
        """
        Initialization method to prepare the tracker with an initial frame.
        """
        self.initial_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

        # Reset state vector and covariance
        self.state = np.zeros(16)
        self.state[6:10] = np.array([0, 0, 0, 1])  # reset orientation quaternion
        self.P = np.eye(16) * 0.01

        print("[VIOTracker] Initialization complete")

    @staticmethod
    def quat_multiply(q1, q2):
        """Multiply two quaternions q = q1 * q2."""
        # q format: (x, y, z, w)
        x0, y0, z0, w0 = q1
        x1, y1, z1, w1 = q2
        return np.array([
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1,
            w0*w1 - x0*x1 - y0*y1 - z0*z1
        ])

    @staticmethod
    def quat_normalize(q):
        return q / np.linalg.norm(q)

    @staticmethod
    def skew(vec):
        """Skew symmetric matrix for vector cross product."""
        x, y, z = vec
        return np.array([[0, -z, y],
                         [z, 0, -x],
                         [-y, x, 0]])

    @staticmethod
    def _wrap_angle(angle):
        return (float(angle) + np.pi) % (2*np.pi) - np.pi

    def state_to_rotation(self):
        """Get rotation matrix from quaternion in state."""
        q = self.state[6:10]
        # Scipy expects quat as [x,y,z,w]
        return R.from_quat(q).as_matrix()

    def _clamp_covariance(self):
        diag = np.clip(np.diag(self.P), self.min_var, self.max_var)
        self.P[np.diag_indices_from(self.P)] = diag

    def predict(self, imu_data, dt=None):
        """
        EKF prediction step: integrate IMU and propagate covariance.
        imu_data expected to have attributes: ax, ay, az, gx, gy, gz
        """
        # Guard: if no IMU data, skip prediction
        if imu_data is None:
            return

        # Derive dt from IMU timestamp if not provided
        if dt is None:
            ts = getattr(imu_data, 'timestamp', None)
            if ts is not None:
                if self._last_timestamp is not None:
                    dt = ts - self._last_timestamp
                self._last_timestamp = ts
        if dt is None:
            dt = self.dt
        # Clamp dt to reasonable bounds
        dt = float(max(1e-4, min(0.2, dt)))

        # Extract state parts
        p = self.state[0:3]
        v = self.state[3:6]
        q = self.state[6:10]
        ba = self.state[10:13]
        bg = self.state[13:16]

        # Get IMU measurements from either vector attributes or scalar fields
        if hasattr(imu_data, 'accel') and hasattr(imu_data, 'gyro') and imu_data.accel is not None and imu_data.gyro is not None:
            acc_raw = np.array([imu_data.accel[0], imu_data.accel[1], imu_data.accel[2]])
            gyro_raw = np.array([imu_data.gyro[0], imu_data.gyro[1], imu_data.gyro[2]])
        else:
            acc_raw = np.array([
                getattr(imu_data, "ax", 0),
                getattr(imu_data, "ay", 0),
                getattr(imu_data, "az", 0)])
            gyro_raw = np.array([
                getattr(imu_data, "gx", 0),
                getattr(imu_data, "gy", 0),
                getattr(imu_data, "gz", 0)])

        # Remove biases
        acc = acc_raw - self.accel_bias
        gyro = gyro_raw - self.gyro_bias

        # Quaternion to rotation matrix
        Rwb = R.from_quat(q).as_matrix()

        # Gravity compensation
        acc_world = Rwb @ acc - self.gravity

        # State propagation
        p_new = p + v*dt + 0.5 * acc_world * dt*dt
        v_new = v + acc_world * dt

        # Quaternion update via gyro integration
        omega_norm = np.linalg.norm(gyro)
        if omega_norm > 1e-12:
            delta_q = np.concatenate(([0], gyro/omega_norm * np.sin(omega_norm*dt/2)))
            delta_q = np.append(np.cos(omega_norm*dt/2), delta_q[1:])
            # Ensure delta_q is in (x,y,z,w) order for scipy
            delta_q = np.array([delta_q[1], delta_q[2], delta_q[3], delta_q[0]])
        else:
            delta_q = np.array([0, 0, 0, 1])

        # Multiply quaternions (x,y,z,w)
        q_new = self.quat_multiply(q, delta_q)
        q_new = self.quat_normalize(q_new)

        # Update state vector
        self.state[0:3] = p_new
        self.state[3:6] = v_new
        self.state[6:10] = q_new

        # Jacobians (simplified for brevity)
        F = np.eye(16)
        F[0:3, 3:6] = np.eye(3)*dt
        F[3:6, 6:9] = -Rwb @ self.skew(acc) * dt
        F[3:6, 10:13] = -Rwb * dt
        # Keep a small coupling of gyro bias to quaternion as an approximation
        # Note: this is a simplification and may be refined with full preintegration
        # F[6:10, 13:16] left as zeros for stability of this baseline

        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q
        self._clamp_covariance()

    def update_flow(self, dx, dy, altitude, confidence=1.0):
        """
        Update step using optical flow deltas as position correction.
        """
        z = np.array([dx, dy, altitude])

        H = np.zeros((3,16))
        H[0:3, 0:3] = np.eye(3)

        confidence_eff = max(float(confidence), 1e-3)
        Rk = self.R_flow / confidence_eff
        y = z - H @ self.state
        S = H @ self.P @ H.T + Rk

        # Innovation gating
        try:
            md2 = float(y.T @ np.linalg.inv(S) @ y)
            if md2 > self._chi2_thresh_flow:
                logging.debug(f"[VIO] Flow update rejected by gating (md2={md2:.2f})")
                return
        except Exception:
            pass

        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.state[6:10] = self.quat_normalize(self.state[6:10])
        self._clamp_covariance()
        self.P = (np.eye(16) - K @ H) @ self.P

    def update_velocity(self, vx, vy, confidence=1.0):
        """
        Update step using velocity measurement (vx, vy) in m/s.
        """
        z = np.array([vx, vy])

        H = np.zeros((2,16))
        H[0, 3] = 1.0
        H[1, 4] = 1.0

        confidence_eff = max(float(confidence), 1e-3)
        Rk = self.R_vel / confidence_eff
        y = z - H @ self.state
        S = H @ self.P @ H.T + Rk

        # Innovation gating
        try:
            md2 = float(y.T @ np.linalg.inv(S) @ y)
            if md2 > self._chi2_thresh_vel:
                logging.debug(f"[VIO] Velocity update rejected by gating (md2={md2:.2f})")
                return
        except Exception:
            pass

        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.state[6:10] = self.quat_normalize(self.state[6:10])
        self._clamp_covariance()
        self.P = (np.eye(16) - K @ H) @ self.P

    def update_vps(self, x, y, z, yaw, confidence=1.0):
        """
        Update step using global pose fix from VPS (Visual Place Recognition).
        Position-only EKF update here; yaw handled by separate nonlinear update.
        """
        z_vec = np.array([x, y, z])

        H = np.zeros((3,16))
        H[0:3, 0:3] = np.eye(3)

        confidence_eff = max(float(confidence), 1e-3)
        Rk = self.R_vps / confidence_eff
        innov = z_vec - H @ self.state
        S = H @ self.P @ H.T + Rk

        # Innovation gating for xyz
        try:
            md2 = float(innov.T @ np.linalg.inv(S) @ innov)
            if md2 > self._chi2_thresh_vps:
                logging.debug(f"[VIO] VPS position update rejected by gating (md2={md2:.2f})")
                return
        except Exception:
            pass

        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innov
        self.state[6:10] = self.quat_normalize(self.state[6:10])
        self._clamp_covariance()
        self.P = (np.eye(16) - K @ H) @ self.P

        # Separate yaw update using nonlinear measurement model
        try:
            self.update_yaw(yaw, confidence=confidence)
        except Exception:
            pass

    def update_yaw(self, yaw_meas, confidence=1.0):
        """
        Nonlinear EKF update for yaw measurement using numerical Jacobian wrt quaternion.
        yaw_meas expected in radians. Confidence scales measurement noise.
        """
        # Extract current yaw from quaternion
        q = self.state[6:10].copy()
        yaw_est = float(R.from_quat(q).as_euler('ZYX')[0])
        # Residual wrapped to [-pi, pi]
        r = self._wrap_angle(yaw_meas - yaw_est)

        # Build measurement Jacobian H_yaw (1x16), nonzero only for quaternion block
        H = np.zeros((1, 16))
        eps = 1e-6
        for i in range(4):
            dq = np.zeros(4)
            dq[i] = eps
            q_pert = self.quat_normalize(q + dq)
            yaw_pert = float(R.from_quat(q_pert).as_euler('ZYX')[0])
            dyaw_dqi = self._wrap_angle(yaw_pert - yaw_est) / eps
            H[0, 6 + i] = dyaw_dqi

        # Kalman gain
        confidence_eff = max(float(confidence), 1e-3)
        Rk = self.R_yaw / confidence_eff
        S = H @ self.P @ H.T + Rk
        # Gating on scalar residual
        try:
            md2 = float(r * (1.0 / S[0,0]) * r)
            if md2 > self._chi2_thresh_yaw:
                logging.debug(f"[VIO] Yaw update rejected by gating (md2={md2:.2f})")
                return
        except Exception:
            pass

        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + (K.flatten() * r)
        # Renormalize quaternion
        self.state[6:10] = self.quat_normalize(self.state[6:10])
        self._clamp_covariance()
        self.P = (np.eye(16) - K @ H) @ self.P

    def zero_velocity_update(self, imu_data):
        """
        Detect zero-velocity and zero-angular-rate conditions and reset velocity and bias accordingly.
        """
        if hasattr(imu_data, 'accel') and imu_data.accel is not None:
            acc = np.array([imu_data.accel[0], imu_data.accel[1], imu_data.accel[2]])
        else:
            acc = np.array([getattr(imu_data, 'ax', 0.0), getattr(imu_data, 'ay', 0.0), getattr(imu_data, 'az', 0.0)])
        if hasattr(imu_data, 'gyro') and imu_data.gyro is not None:
            gyro = np.array([imu_data.gyro[0], imu_data.gyro[1], imu_data.gyro[2]])
        else:
            gyro = np.array([getattr(imu_data, 'gx', 0.0), getattr(imu_data, 'gy', 0.0), getattr(imu_data, 'gz', 0.0)])

        acc_norm = np.linalg.norm(acc - self.gravity)
        gyro_norm = np.linalg.norm(gyro)

        if acc_norm < self.zv_thresh_acc and gyro_norm < self.zv_thresh_gyro:
            self.state[3:6] = 0
            self.state[10:13] *= 0.99
            self.state[13:16] *= 0.99

    def update(self, dx, dy, altitude, confidence=1.0):
        
        
        self.update_flow(dx, dy, altitude, confidence=confidence)
        pos = self.state[0:3]
        return pos[0], pos[1], pos[2]

    
    def get_state(self):
        """Return current estimated state: position, velocity, quaternion, biases"""
        return {
            "position": self.state[0:3],
            "velocity": self.state[3:6],
            "orientation_quat": self.state[6:10],
            "accel_bias": self.state[10:13],
            "gyro_bias": self.state[13:16]
        }

