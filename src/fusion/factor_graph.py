#!/usr/bin/env python3
"""
Factor-graph optimizer using GTSAM (if available), with a graceful fallback when
GTSAM is not installed.

This module provides a small façade, `FactorGraphOptimizer`, that the runtime can
use without having to branch on whether GTSAM is installed. The optimizer
supports:

- Absolute pose priors (e.g., VPS global fixes)
- Relative motion constraints (e.g., optical-flow/VIO deltas)
- Optional continuity priors to encourage smooth trajectories when sparse
- Optional yaw-only priors (e.g., from ATTITUDE yaw)
- Per-factor robust losses and adaptive covariances
- Sliding-window pruning

Design notes:
- When GTSAM is not available, a tiny buffer-only mode is used: the last pose is
  incrementally updated by relative motions and overwritten by absolute
  priors. This mode is intended for demos; it does not perform real optimization.
- Timestamps are discretized to milliseconds to form integer keys (X(k)).
- `abs_sigma`/`rel_sigma` are 6-DOF Pose3 sigmas. When `adaptive_cov` is true,
  a caller-provided scalar covariance replaces the default, applied across all 6
  components for simplicity.
- `use_se2` sets dz=0 on factors and priors. It still uses Pose3 internally but
  constrains the motion to the plane.
"""

import numpy as np

try:
    # GTSAM imports (only if installed). The module exposes a fallback otherwise
    import gtsam
    from gtsam import (
        NonlinearFactorGraph, Values,
        noiseModel, Pose3, Rot3, Point3,
        PriorFactorPose3, BetweenFactorPose3
    )
    from gtsam.symbol_shorthand import X
    GTSAM_AVAILABLE = True
except ImportError:
    # Fallback mode flag
    GTSAM_AVAILABLE = False


class FactorGraphOptimizer:
    """Minimal façade around GTSAM NonlinearFactorGraph.

    Parameters
    ----------
    window_size : int
        Number of unique timestamps to keep in the optimization window.
    config : dict | None
        Project config; we only look at the `fusion` section for noise/robust-params.
    """

    def __init__(self, window_size=10, config=None):
        self.window_size = window_size

        if not GTSAM_AVAILABLE:
            # Simple fallback: store recent poses and timestamps.
            # We accumulate an operation log only for timestamps; estimates are
            # updated incrementally by relative deltas and overwritten by absolute
            # priors, so there is no real optimization here.
            self.buf = []                 # list[tuple(ts, x, y, z, yaw)]
            self.timestamps = []          # list[ts]
            return

        # Initialize GTSAM graph and values
        self.graph = NonlinearFactorGraph()
        self.initial = Values()
        self.timestamps = []

        # Read fusion settings
        cfg = config.get("fusion", {}) if config else {}
        abs_sigma = cfg.get("abs_sigma", 1.0)  # 6D sigma for absolute priors
        rel_sigma = cfg.get("rel_sigma", 0.1)  # 6D sigma for relative constraints
        self.continuity_sigma = cfg.get("continuity_sigma", None)  # None disables
        self.use_se2 = bool(cfg.get("use_se2", False))

        # Robust losses (global or per-factor kinds)
        self.robust_loss_global = cfg.get("robust_loss", None)  # None|'huber'|'cauchy'|'geman'
        self.robust_delta = float(cfg.get("robust_delta", 1.0))
        self.robust_loss_abs = cfg.get("robust_loss_abs", self.robust_loss_global)
        self.robust_loss_rel = cfg.get("robust_loss_rel", self.robust_loss_global)

        # If true, caller-provided scalar covariance replaces defaults for each factor
        self.adaptive_cov = bool(cfg.get("adaptive_cov", True))

        # Default 6D diagonal sigmas for absolute and relative factors
        self.noise_abs_default = noiseModel.Diagonal.Sigmas([abs_sigma]*6)
        self.noise_rel_default = noiseModel.Diagonal.Sigmas([rel_sigma]*6)

        # Operation log used for pruning/rebuild
        # Each op is a tuple that captures enough info to reconstruct the graph window
        self._ops = []  # list[tuple]

    def _make_robust(self, base_noise, which=None):
        """Wrap a noise model with a robust m-estimator if configured.

        Parameters
        ----------
        base_noise : gtsam.noiseModel
            The underlying Gaussian noise.
        which : str | None
            None|'abs'|'rel' selects which robust loss entry to use.
        """
        loss = self.robust_loss_global if which is None else (self.robust_loss_abs if which == 'abs' else self.robust_loss_rel)
        if loss is None:
            return base_noise
        lname = loss.lower() if isinstance(loss, str) else str(loss).lower()
        if lname == 'huber':
            return noiseModel.Robust.Create(noiseModel.mEstimator.Huber(self.robust_delta), base_noise)
        if lname == 'cauchy':
            return noiseModel.Robust.Create(noiseModel.mEstimator.Cauchy(self.robust_delta), base_noise)
        if lname == 'geman':
            return noiseModel.Robust.Create(noiseModel.mEstimator.GemanMcClure(self.robust_delta), base_noise)
        return base_noise

    def _sigma6(self, s):
        """Create a 6D diagonal noise model with all components = s."""
        return noiseModel.Diagonal.Sigmas([s]*6)

    def add_absolute(self, ts, x, y, z, yaw, covariance=None):
        """Add an absolute pose prior at timestamp `ts`.

        Parameters
        ----------
        ts : float
            Timestamp in seconds.
        x, y, z : float
            Position in meters.
        yaw : float
            Heading in radians (about Z).
        covariance : float | None
            If provided and `adaptive_cov` is true, overrides default sigma for this factor.
        """
        if not GTSAM_AVAILABLE:
            # Fallback: store as latest absolute value within the window
            self.buf.append((ts, x, y, z, yaw))
            self.timestamps.append(ts)
            self.buf = self.buf[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]
            return

        key = X(int(ts * 1e3))  # millisecond key
        if self.use_se2:
            z = 0.0
        pose = Pose3(Rot3.Yaw(yaw), Point3(x, y, z))
        base_noise = self.noise_abs_default if covariance is None or not self.adaptive_cov else self._sigma6(covariance)
        noise = self._make_robust(base_noise, which='abs')
        self.graph.add(PriorFactorPose3(key, pose, noise))

        # Seed/update initial guess
        if self.initial.exists(key):
            self.initial.update(key, pose)
        else:
            self.initial.insert(key, pose)

        self.timestamps.append(ts)
        self._ops.append(("abs", ts, x, y, z, yaw, covariance))
        self._prune()

    def add_relative(self, t0, t1, dx, dy, dz, dyaw, covariance=None):
        """Add a relative motion constraint between timestamps t0 and t1.

        Parameters
        ----------
        t0, t1 : float
            Start and end timestamps.
        dx, dy, dz : float
            Relative translation in meters.
        dyaw : float
            Relative yaw rotation in radians.
        covariance : float | None
            Optional scalar sigma for this factor if `adaptive_cov` is enabled.
        """
        if not GTSAM_AVAILABLE:
            # Fallback: integrate relative motion onto last absolute pose, if any
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

        # If we don't have an initial guess for key_j yet, compose from key_i
        if not self.initial.exists(key_j):
            prev = self.initial.atPose3(key_i)
            self.initial.insert(key_j, prev.compose(delta))

        self._ops.append(("rel", t0, t1, dx, dy, dz, dyaw, covariance))

    def add_continuity(self, t0, t1, sigma=None):
        """Add a weak constraint tying successive poses to discourage drift.

        If `sigma` is None, the constructor's `continuity_sigma` is used. If that
        is also None, this function is a no-op.
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
        self._ops.append(("cont", t0, t1, sigma))

    def add_vio_factor(self, t0, t1, vx, vy, dt, gyro_z):
        """Helper to convert velocity/yaw-rate into a relative motion factor."""
        dx = vx * dt
        dy = vy * dt
        dz = 0.0  # planar assumption
        dyaw = gyro_z * dt
        self.add_relative(t0, t1, dx, dy, dz, dyaw)

    def add_yaw_prior(self, ts, yaw, sigma=0.05):
        """Add a soft yaw-only prior at timestamp `ts`.

        Other pose components are left with very large uncertainty.
        """
        if not GTSAM_AVAILABLE:
            # fallback mode does not support selective priors
            return

        key = X(int(ts * 1e3))
        # Find current translation guess if present, otherwise use origin
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

        # Seed initial guess if missing
        if not self.initial.exists(key):
            self.initial.insert(key, pose)
        self._ops.append(("yaw", ts, yaw, sigma))

    def optimize(self):
        """Run Levenberg–Marquardt and replace `initial` with the result.

        Returns
        -------
        gtsam.Values | bool
            Optimized values object in GTSAM mode, or True for fallback mode.
        """
        if not GTSAM_AVAILABLE:
            return True
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
        result = optimizer.optimize()
        self.initial = result
        return result

    def get_current_pose(self):
        """Return the latest optimized pose as (x, y, z, yaw).

        In fallback mode, returns the last absolute pose stored.
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

        # Support both gtsam.Point3 and plain arrays
        if hasattr(t, 'x'):
            return (t.x(), t.y(), t.z(), yaw)
        if isinstance(t, (list, tuple, np.ndarray)):
            return (t[0], t[1], t[2], yaw)
        raise TypeError(f"Unknown translation type: {type(t)}")

    def _prune(self):
        """Keep only the last `window_size` timestamps (sliding window).

        In GTSAM mode we rebuild the factor graph and initial values from the
        operation log filtered to the last window. In fallback mode we just trim
        the pose buffer and timestamps.
        """
        if not GTSAM_AVAILABLE:
            self.buf = self.buf[-self.window_size:]
            self.timestamps = self.timestamps[-self.window_size:]
            return

        # Determine last `window_size` timestamps
        unique_ts = sorted(set(self.timestamps))
        if len(unique_ts) <= self.window_size:
            return
        allowed = unique_ts[-self.window_size:]
        allowed_set = set(allowed)

        # Rebuild graph and initial from filtered ops
        new_graph = NonlinearFactorGraph()
        new_initial = Values()
        new_timestamps = []

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


