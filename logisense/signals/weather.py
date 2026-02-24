"""
Weather Risk Processor
=======================

Translates NWP (Numerical Weather Prediction) forecasts into per-node
supply chain disruption risk scores.

Sources
-------
- NOAA GFS: 16-day global forecast at 0.25° resolution
- ECMWF IFS: 15-day medium-range forecast
- NOAA National Hurricane Center: tropical cyclone track / intensity

Weather ↔ supply impact mapping
---------------------------------
Tropical cyclone     → port closure, inland flooding
Winter storm         → ground transport, DC operations
Coastal / river flood → road closures, DC inundation
Extreme heat         → labour disruption, cold-chain break
Fog / low visibility → port and airport delays
Drought              → agricultural supply, hydro power

Features produced (12 per node)
---------------------------------
cyclone_proximity, cyclone_intensity, precip_risk,
flood_risk_72h, wind_zscore, temp_anomaly,
snow_risk, visibility_risk, lightning,
heat_stress, drought_severity, severe_composite
"""

import numpy as np
from typing import Dict, List, Optional

N_WEATHER_FEATURES = 12

FEATURE_NAMES = [
    "cyclone_proximity", "cyclone_intensity", "precip_risk",
    "flood_risk_72h", "wind_zscore", "temp_anomaly",
    "snow_risk", "visibility_risk", "lightning",
    "heat_stress", "drought_severity", "severe_composite",
]


class WeatherProcessor:
    """
    Converts NWP data into (N, 12) per-node weather risk features.

    Uses geographic proximity decay to map storm tracks to supply nodes:

        risk_i = Σ_e  severity_e * exp(−dist(node_i, event_e) / decay_km)

    Args:
        decay_km:   Geographic decay constant (default 500 km).
        horizon_h:  Forecast horizon in hours (default 240 h = 10 days).
    """

    def __init__(self, decay_km: float = 500.0, horizon_h: int = 240):
        self.decay_km = decay_km
        self.horizon_h = horizon_h

    def fetch(self, node_ids: List[str], mock: bool = True) -> np.ndarray:
        """Return (N, 12) weather feature matrix."""
        if mock:
            return self._mock(len(node_ids))
        raise NotImplementedError("Set NOAA_API_KEY for live weather data.")

    def _mock(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(seed=13)
        data = rng.uniform(0, 0.2, (n, N_WEATHER_FEATURES)).astype(np.float32)
        # Cyclone approaching node 2
        data[2, 0] = 0.82
        data[2, 1] = 0.71
        data[2, 2] = 0.90
        # Flood risk at node 5
        data[5, 3] = 0.75
        data[5, 11] = 0.78
        return data

    @staticmethod
    def cyclone_risk(
        node_lat: float, node_lon: float,
        storm_lat: float, storm_lon: float,
        category: int, landfall_hours: float,
        decay_km: float = 500.0,
    ) -> float:
        """
        Risk of cyclone disruption at a node.
        Decays with geographic distance and forecast lead time.

        Args:
            category:       Saffir-Simpson category (1–5).
            landfall_hours: Hours until predicted landfall.
        """
        dlat = np.radians(node_lat - storm_lat)
        dlon = np.radians(node_lon - storm_lon)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(storm_lat))
             * np.cos(np.radians(node_lat))
             * np.sin(dlon / 2) ** 2)
        dist_km = 6371.0 * 2.0 * np.arcsin(np.sqrt(a))
        intensity = category / 5.0
        time_decay = np.exp(-landfall_hours / 144.0)   # 144 h = 6-day e-fold
        return float(intensity * np.exp(-dist_km / decay_km) * time_decay)
