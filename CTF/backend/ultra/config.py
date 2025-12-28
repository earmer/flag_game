from dataclasses import dataclass, field

from .weights import UltraWeights


@dataclass(frozen=True)
class UltraConfig:
    intercept_horizon_trap: int = 12
    intercept_horizon_carrier: int = 9
    intercept_arrival_slack: int = 1

    carrier_direct_target_threshold: int = 1
    carrier_chase_detour_slack: int = 1
    carrier_chase_score_slack: float = 0.5
    carrier_chase_min_ticks: int = 24

    trap_intruder_close_steps: int = 3
    trap_urgent_escape_steps: int = 2

    double_team_score_threshold: float = 0.0
    double_team_advantage_margin: int = 1

    weights: UltraWeights = field(default_factory=UltraWeights)
