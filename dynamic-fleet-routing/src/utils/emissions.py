"""Carbon emissions calculator for EV vs ICE fleet comparison.

Provides standardized CO₂ emission factors for electric vehicle (EV)
and internal combustion engine (ICE) fleets, enabling quantitative
carbon footprint analysis across dispatch methods.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmissionsResult:
    """Result of a comparative emissions calculation.

    Attributes:
        ev_co2_kg: CO₂ emissions from EV fleet (kg).
        ice_co2_kg: CO₂ emissions from equivalent ICE fleet (kg).
        co2_saved_kg: Absolute CO₂ savings (ICE - EV) in kg.
        co2_reduction_pct: Percentage reduction vs ICE baseline.
        ev_energy_kwh: Total energy consumed by EV fleet (kWh).
        distance_km: Total fleet distance (km).
    """
    ev_co2_kg: float
    ice_co2_kg: float
    co2_saved_kg: float
    co2_reduction_pct: float
    ev_energy_kwh: float
    distance_km: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "ev_co2_kg": round(self.ev_co2_kg, 3),
            "ice_co2_kg": round(self.ice_co2_kg, 3),
            "co2_saved_kg": round(self.co2_saved_kg, 3),
            "co2_reduction_pct": round(self.co2_reduction_pct, 2),
            "ev_energy_kwh": round(self.ev_energy_kwh, 3),
            "distance_km": round(self.distance_km, 3),
        }


class EmissionsCalculator:
    """Compares CO₂ emissions between electric vehicle and ICE fleets.

    Default emission factors:
        - EV: 0.233 kg CO₂/kWh (US grid average, EPA 2024)
        - ICE: 0.21 kg CO₂/km (average light commercial vehicle, ICCT)

    These factors are configurable to model different electricity
    grid mixes (e.g., France nuclear: 0.05, India coal-heavy: 0.7)
    and different ICE vehicle classes.

    Attributes:
        ev_co2_per_kwh: EV emission factor (kg CO₂ per kWh consumed).
        ice_co2_per_km: ICE emission factor (kg CO₂ per km driven).
    """

    def __init__(
        self,
        ev_co2_per_kwh: float = 0.233,
        ice_co2_per_km: float = 0.21,
    ) -> None:
        """Initialize the emissions calculator.

        Args:
            ev_co2_per_kwh: kg CO₂ per kWh of electricity consumed.
            ice_co2_per_km: kg CO₂ per km driven by ICE vehicle.

        Raises:
            ValueError: If emission factors are negative.
        """
        if ev_co2_per_kwh < 0:
            raise ValueError(f"EV emission factor must be non-negative: {ev_co2_per_kwh}")
        if ice_co2_per_km < 0:
            raise ValueError(f"ICE emission factor must be non-negative: {ice_co2_per_km}")
        self.ev_co2_per_kwh = ev_co2_per_kwh
        self.ice_co2_per_km = ice_co2_per_km

    def calculate_ev_emissions(self, energy_kwh: float) -> float:
        """Calculate CO₂ emissions for an EV fleet.

        Args:
            energy_kwh: Total energy consumed (kWh).

        Returns:
            CO₂ emissions in kg.
        """
        return max(0.0, energy_kwh) * self.ev_co2_per_kwh

    def calculate_ice_emissions(self, distance_km: float) -> float:
        """Calculate CO₂ emissions for an equivalent ICE fleet.

        Args:
            distance_km: Total distance driven (km).

        Returns:
            CO₂ emissions in kg.
        """
        return max(0.0, distance_km) * self.ice_co2_per_km

    def calculate_savings(
        self, distance_km: float, energy_kwh: float
    ) -> EmissionsResult:
        """Compare EV vs ICE emissions and calculate savings.

        Args:
            distance_km: Total fleet distance driven (km).
            energy_kwh: Total EV energy consumed (kWh).

        Returns:
            EmissionsResult with full comparison metrics.
        """
        ev_co2 = self.calculate_ev_emissions(energy_kwh)
        ice_co2 = self.calculate_ice_emissions(distance_km)
        saved = ice_co2 - ev_co2
        pct = (saved / ice_co2 * 100.0) if ice_co2 > 0 else 0.0

        return EmissionsResult(
            ev_co2_kg=ev_co2,
            ice_co2_kg=ice_co2,
            co2_saved_kg=saved,
            co2_reduction_pct=pct,
            ev_energy_kwh=energy_kwh,
            distance_km=distance_km,
        )
