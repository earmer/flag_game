from dataclasses import dataclass

from .accel import argmin
from .intercept import InterceptOption
from .weights import score_carrier_chase, score_direct_path


@dataclass(frozen=True)
class CarrierChaseOption:
    target: dict
    intercept: InterceptOption
    after_steps: int
    detour: int
    threat_steps: int
    score: float


def find_best_carrier_chase(ai, start, opponents, targets, my_flags, prisons, config):
    direct_steps = None
    if targets:
        direct_steps = ai._min_steps_any(start, list(targets), restrict_safe=True)

    if direct_steps is not None and direct_steps <= int(config.carrier_direct_target_threshold):
        return None, direct_steps, None

    candidates = []
    scores = []

    for o in opponents or []:
        o_pos = ai._pos(o)
        if not ai._is_safe(o_pos):
            continue

        o_path = ai._predict_intruder_goal_path(o, my_flags=my_flags, prisons=prisons)
        if not o_path:
            continue

        intercept = ai._best_intercept_on_path(
            start,
            o_path,
            horizon=config.intercept_horizon_carrier,
        )
        if not intercept:
            continue

        after_steps = 0
        detour = 0
        if targets:
            after_steps = ai._min_steps_any(intercept.cell, list(targets), restrict_safe=True)
            if after_steps is None:
                continue
            if direct_steps is not None:
                detour = (intercept.steps + after_steps) - direct_steps
                if detour > int(config.carrier_chase_detour_slack):
                    continue
                detour = max(0, detour)

        threat_steps = len(o_path) - 1
        score = score_carrier_chase(
            intercept.steps,
            detour,
            threat_steps,
            intercept.margin,
            weights=config.weights.carrier,
        )
        candidates.append((o, intercept, after_steps, detour, threat_steps, score))
        scores.append(score)

    idx = argmin(scores)
    if idx is None:
        return None, direct_steps, None

    target, intercept, after_steps, detour, threat_steps, score = candidates[idx]
    direct_score = None
    if direct_steps is not None:
        direct_score = score_direct_path(direct_steps, weights=config.weights.carrier)

    return (
        CarrierChaseOption(
            target=target,
            intercept=intercept,
            after_steps=after_steps,
            detour=detour,
            threat_steps=threat_steps,
            score=score,
        ),
        direct_steps,
        direct_score,
    )
