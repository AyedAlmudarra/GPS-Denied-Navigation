#!/usr/bin/env python3
"""
Basic angle utilities for the navigation toolkit.
- wrap_angle_rad: normalize radians to [-pi, pi]
- deg2rad/rad2deg: degree/radian conversions
"""
import math


def wrap_angle_rad(angle):
    """Wrap angle (radians) to the range [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def deg2rad(deg):
    """Convert degrees to radians."""
    return deg * math.pi / 180.0


def rad2deg(rad):
    """Convert radians to degrees."""
    return rad * 180.0 / math.pi 