from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundaryPathWeights:
    steps: float = 1.0
    guard_bias: float = 0.1


@dataclass(frozen=True)
class InterceptWeights:
    steps: float = 1.0
    margin: float = 0.8
    index: float = 0.2


@dataclass(frozen=True)
class CarrierChaseWeights:
    intercept_steps: float = 1.0
    detour: float = 1.25
    threat: float = 0.5
    margin: float = 0.8
    direct: float = 1.0


@dataclass(frozen=True)
class TrapWeights:
    intercept_steps: float = 1.0
    escape_steps: float = 0.6
    margin: float = 0.8
    index: float = 0.15
    secondary_penalty: float = 0.35


@dataclass(frozen=True)
class DoubleTeamWeights:
    advantage: float = 1.0
    urgency: float = 1.5
    risk: float = 1.0


@dataclass(frozen=True)
class UltraWeights:
    boundary: BoundaryPathWeights = field(default_factory=BoundaryPathWeights)
    intercept: InterceptWeights = field(default_factory=InterceptWeights)
    carrier: CarrierChaseWeights = field(default_factory=CarrierChaseWeights)
    trap: TrapWeights = field(default_factory=TrapWeights)
    double_team: DoubleTeamWeights = field(default_factory=DoubleTeamWeights)


def score_boundary_path(steps, guard_dist, *, weights):
    return weights.steps * steps - weights.guard_bias * guard_dist


def score_intercept(steps, margin, index, *, weights):
    return weights.steps * steps - weights.margin * margin + weights.index * index


def score_carrier_chase(intercept_steps, detour, threat_steps, margin, *, weights):
    return (
        weights.intercept_steps * intercept_steps
        + weights.detour * detour
        + weights.threat * threat_steps
        - weights.margin * margin
    )


def score_direct_path(direct_steps, *, weights):
    return weights.direct * direct_steps


def score_trap_candidate(intercept_steps, escape_steps, margin, index, *, weights, secondary=False):
    score = (
        weights.intercept_steps * intercept_steps
        + weights.escape_steps * escape_steps
        - weights.margin * margin
        + weights.index * index
    )
    if secondary:
        score += weights.secondary_penalty
    return score


def score_double_team(advantage, urgent, risk, *, weights):
    return weights.advantage * advantage + weights.urgency * urgent - weights.risk * risk
