#!/usr/bin/env python3
"""
Factor-graph optimizer using GTSAM.

Supports adding absolute poses (e.g., from VPS global fixes) and relative
motions (e.g., from optical flow or VIO), fusing them into a smoothed pose estimate.

Includes fallback mode if GTSAM is unavailable.
"""

import numpy as np

try:
    import gtsam
    from gtsam import (
        NonlinearFactorGraph, Values,
        noiseModel, Pose3, Rot3, Point3,
        PriorFactorPose3, BetweenFactorPose3
    )
    from gtsam.symbol_shorthand import X
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False

class FactorGraphOptimizer:
    def __init__(self, window_size=10, config=None):
        self.window_size = window_size

        if not GTSAM_AVAILABLE:
            # Simple fallback: store recent poses and timestamps
            self.buf = []                 # list of tuples (ts, x, y, z, yaw)
            self.timestamps = []          # parallel list of timestamps
            return

        # Initialize GTSAM graph and values
        self.graph = NonlinearFactorGraph()
        self.initial = Values()
        self.timestamps = []

        cfg = config.get("fusion", {}) if config else {}
        abs_sigma = cfg.get("abs_sigma", 1.0)
        rel_sigma = cfg.get("rel_sigma", 0.1)
        self.continuity_sigma = cfg.get("continuity_sigma", None)
        self.use_se2 = bool(cfg.get("use_se2", False))
        # Robust kernels (global or per-factor)
        self.robust_loss_global = cfg.get("robust_loss", None)  # None, 'huber', 'cauchy', 'geman'
        self.robust_delta = float(cfg.get("robust_delta", 1.0))
        self.robust_loss_abs = cfg.get("robust_loss_abs", self.robust_loss_global)
        self.robust_loss_rel = cfg.get("robust_loss_rel", self.robust_loss_global)
        self.adaptive_cov = bool(cfg.get("adaptive_cov", True))

        # Default noise models for absolute and relative factors
        self.noise_abs_default = noiseModel.Diagonal.Sigmas([abs_sigma]*6)
        self.noise_rel_default = noiseModel.Diagonal.Sigmas([rel_sigma]*6)

        # Operation log for pruning/rebuild
        self._ops = []  # list of tuples describing added factors/priors

    def _make_robust(self, base_noise, which=None):
        loss = self.robust_loss_global if which is None else (self.robust_loss_abs if which == 'abs' else self.robust_loss_rel)
        if loss is None:
            return base_noise
        if isinstance(loss, str):
            lname = loss.lower()
        else:
            lname = str(loss).lower()
        if lname == 'huber':
            return noiseModel.Robust.Create(noiseModel.mEstimator.Huber(self.robust_delta), base_noise)
        if lname == 'cauchy':
            return noiseModel.Robust.Create(noiseModel.mEstimator.Cauchy(self.robust_delta), base_noise)
        if lname == 'geman':
            return noiseModel.Robust.Create(noiseModel.mEstimator.GemanMcClure(self.robust_delta), base_noise)
        return base_noise

    def _sigma6(self, s):
        return noiseModel.Diagonal.Sigmas([s]*6)

    def add_absolute(self, ts, x, y, z, yaw, covariance=None):
        """
        Add an absolute pose prior at timestamp ts.

        Args:
            ts (float): Timestamp (seconds).
            x, y, z (float): Position.
            yaw (float): Yaw angle in radians.
            covariance (float or None): Optional scalar sigma for noise model.
        """
        if not GTSAM_AVAILABLE:
            self.buf.append((ts, x, y, z, yaw))
            self.timestamps.append(ts)
            # prune to window size
            self.buf = self.buf[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]
            return

        key = X(int(ts * 1e3))  # Discretize timestamp to milliseconds as key
        if self.use_se2:
            z = 0.0
        pose = Pose3(Rot3.Yaw(yaw), Point3(x, y, z))
        base_noise = self.noise_abs_default if covariance is None or not self.adaptive_cov else self._sigma6(covariance)
        noise = self._make_robust(base_noise, which='abs')
        self.graph.add(PriorFactorPose3(key, pose, noise))
        
        if self.initial.exists(key): 
            self.initial.update(key, pose)
        else:
            self.initial.insert(key, pose)

        self.timestamps.append(ts)
        # Log op for rebuild
        self._ops.append(("abs", ts, x, y, z, yaw, covariance))
        self._prune()

    def add_relative(self, t0, t1, dx, dy, dz, dyaw, covariance=None):
        """
        Add a relative motion factor between two timestamps.

        Args:
            t0, t1 (float): Start and end timestamps.
            dx, dy, dz (float): Relative translation.
            dyaw (float): Relative yaw rotation in radians.
            covariance (float or None): Optional scalar sigma for noise model.
        """
        if not GTSAM_AVAILABLE:
            # In fallback, just append absolute pose computed incrementally
            if self.buf:
                _, x, y, z, yaw = self.buf[-1]
                self.add_absolute(t1, x+dx, y+dy, z+dz, yaw+dyaw)
            return

        key_i, key_j = X(int(t0*1e3)), X(int(t1*1e3))
        if self.use_se2:
            dz = 0.0
        delta = Pose3(Rot3.Yaw(dyaw), Point3(dx, dy, dz))
        base_noise = self.noise_rel_default if covariance is None or not self.adaptive_cov else self._sigma6(covariance)
        noise = self._make_robust(base_noise, which='rel')
        self.graph.add(BetweenFactorPose3(key_i, key_j, delta, noise))
        if not self.initial.exists(key_j):
            prev = self.initial.atPose3(key_i)
            self.initial.insert(key_j, prev.compose(delta))
        # Log op for rebuild
        self._ops.append(("rel", t0, t1, dx, dy, dz, dyaw, covariance))

    def add_continuity(self, t0, t1, sigma=None):
        """
        Add a loose prior tying two successive poses to discourage drift when sparse.
        Only available when GTSAM is present.
        """
        if not GTSAM_AVAILABLE:
            return
        if sigma is None:
            sigma = self.continuity_sigma
        if sigma is None:
            return
        key_i, key_j = X(int(t0*1e3)), X(int(t1*1e3))
        # Encourage small delta between poses by adding a BetweenFactor with small noise around identity
        delta = Pose3(Rot3(), Point3(0, 0, 0))
        base_noise = self._sigma6(sigma)
        noise = self._make_robust(base_noise)
        self.graph.add(BetweenFactorPose3(key_i, key_j, delta, noise))
        # Log op for rebuild
        self._ops.append(("cont", t0, t1, sigma))

    def add_vio_factor(self, t0, t1, vx, vy, dt, gyro_z):
        """
        Adds a motion factor using VIO velocity and yaw rate.
        Converts velocity and yaw rate into relative motion.
        """
        dx = vx * dt
        dy = vy * dt
        dz = 0.0  # Assume flat plane for now
        dyaw = gyro_z * dt
        self.add_relative(t0, t1, dx, dy, dz, dyaw)

    def add_yaw_prior(self, ts, yaw, sigma=0.05):
        """
        Adds a soft yaw-only prior at the given timestamp.
        Other pose components are left with large uncertainty.
        """
        if not GTSAM_AVAILABLE:
            # fallback mode does not support selective priors
            return

        key = X(int(ts * 1e3))
        # Get current estimate or zero
        if self.initial.exists(key):
            pose = self.initial.atPose3(key)
            t = pose.translation()
        else:
            t = Point3(0, 0, 0)

        rot = Rot3.Yaw(yaw)
        pose = Pose3(rot, t)

        # Very loose on xyz + roll/pitch, tight on yaw
        sigmas = [1000, 1000, 1000, 1000, 1000, sigma]
        base_noise = noiseModel.Diagonal.Sigmas(sigmas)
        noise = self._make_robust(base_noise)
        self.graph.add(PriorFactorPose3(key, pose, noise))

        # Insert if not already initialized
        if not self.initial.exists(key):
            self.initial.insert(key, pose)
        # Log op for rebuild
        self._ops.append(("yaw", ts, yaw, sigma))

    def optimize(self):
        """
        Run the graph optimizer and update the internal state.
        Returns the optimization result object or True for fallback.
        """
        if not GTSAM_AVAILABLE:
            return True
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
        result = optimizer.optimize()
        self.initial = result
        return result

    def get_current_pose(self):
        """
        Return the latest optimized pose as (x, y, z, yaw).
        Works for both GTSAM and fallback modes.
        """
        if not GTSAM_AVAILABLE:
            if not self.buf:
                return (0, 0, 0, 0)
            _, x, y, z, yaw = self.buf[-1]
            return (x, y, z, yaw)

        if not self.timestamps:
            return (0, 0, 0, 0)

        key = X(int(self.timestamps[-1] * 1e3))
        pose = self.initial.atPose3(key)
        t = pose.translation()
        yaw = pose.rotation().yaw()

        # Handle translation type robustness
        if hasattr(t, 'x'):
            return (t.x(), t.y(), t.z(), yaw)
        elif isinstance(t, (list, tuple, np.ndarray)):
            return (t[0], t[1], t[2], yaw)
        else:
            raise TypeError(f"Unknown translation type: {type(t)}")

    def _prune(self):
        """
        Keep only the last `window_size` timestamps.
        For fallback, just trims the buffer and timestamps.
        For GTSAM, rebuilds the factor graph from the last window of ops.
        """
        if not GTSAM_AVAILABLE:
            self.buf = self.buf[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]
            return

        # Determine allowed timestamps
        unique_ts = sorted(set(self.timestamps))
        if len(unique_ts) <= self.window_size:
            return
        allowed = unique_ts[-self.window_size:]
        allowed_set = set(allowed)

        # Rebuild graph and initial from filtered ops
        new_graph = NonlinearFactorGraph()
        new_initial = Values()
        new_timestamps = []

        # Helper to ensure initial insertion for abs ops
        def ensure_initial(ts, pose):
            key = X(int(ts * 1e3))
            if new_initial.exists(key):
                new_initial.update(key, pose)
            else:
                new_initial.insert(key, pose)

        for op in self._ops:
            if op[0] == "abs":
                _, ts, x, y, z, yaw, covariance = op
                if ts not in allowed_set:
                    continue
                key = X(int(ts * 1e3))
                if self.use_se2:
                    z = 0.0
                pose = Pose3(Rot3.Yaw(yaw), Point3(x, y, z))
                base_noise = self.noise_abs_default if covariance is None or not self.adaptive_cov else self._sigma6(covariance)
                noise = self._make_robust(base_noise, which='abs')
                new_graph.add(PriorFactorPose3(key, pose, noise))
                ensure_initial(ts, pose)
                new_timestamps.append(ts)
            elif op[0] == "rel":
                _, t0, t1, dx, dy, dz, dyaw, covariance = op
                if (t0 not in allowed_set) or (t1 not in allowed_set):
                    continue
                if self.use_se2:
                    dz = 0.0
                key_i, key_j = X(int(t0*1e3)), X(int(t1*1e3))
                delta = Pose3(Rot3.Yaw(dyaw), Point3(dx, dy, dz))
                base_noise = self.noise_rel_default if covariance is None or not self.adaptive_cov else self._sigma6(covariance)
                noise = self._make_robust(base_noise, which='rel')
                new_graph.add(BetweenFactorPose3(key_i, key_j, delta, noise))
                if not new_initial.exists(key_j) and new_initial.exists(key_i):
                    prev = new_initial.atPose3(key_i)
                    new_initial.insert(key_j, prev.compose(delta))
            elif op[0] == "cont":
                _, t0, t1, sigma = op
                if (t0 not in allowed_set) or (t1 not in allowed_set):
                    continue
                key_i, key_j = X(int(t0*1e3)), X(int(t1*1e3))
                delta = Pose3(Rot3(), Point3(0, 0, 0))
                base_noise = self._sigma6(self.continuity_sigma if sigma is None else sigma)
                noise = self._make_robust(base_noise)
                new_graph.add(BetweenFactorPose3(key_i, key_j, delta, noise))
            elif op[0] == "yaw":
                _, ts, yaw, sigma = op
                if ts not in allowed_set:
                    continue
                key = X(int(ts * 1e3))
                if new_initial.exists(key):
                    pose = new_initial.atPose3(key)
                    t = pose.translation()
                else:
                    t = Point3(0, 0, 0)
                rot = Rot3.Yaw(yaw)
                pose = Pose3(rot, t)
                sigmas = [1000, 1000, 1000, 1000, 1000, (sigma if sigma is not None else 0.05)]
                base_noise = noiseModel.Diagonal.Sigmas(sigmas)
                noise = self._make_robust(base_noise)
                new_graph.add(PriorFactorPose3(key, pose, noise))
                if not new_initial.exists(key):
                    new_initial.insert(key, pose)

        # Commit rebuild
        self.graph = new_graph
        self.initial = new_initial
        self.timestamps = sorted(set(new_timestamps))


