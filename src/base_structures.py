from dataclasses import dataclass
import math

@dataclass
class Location:
    """Geodetic location: latitude (°), longitude (°), altitude (m)."""
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0

    def distance_to(self, other: "Location") -> float:
        """Approximate Haversine distance (in meters)."""
        R = 6_371_000.0  # Earth radius, m
        φ1, φ2 = math.radians(self.lat), math.radians(other.lat)
        Δφ = math.radians(other.lat - self.lat)
        Δλ = math.radians(other.lon - self.lon)
        a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

