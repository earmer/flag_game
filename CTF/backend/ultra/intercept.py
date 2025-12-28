from dataclasses import dataclass

from .accel import argmin
from .weights import score_boundary_path, score_intercept


@dataclass(frozen=True)
class InterceptOption:
    cell: tuple
    path: list
    index: int
    margin: int
    steps: int
    score: float


def _choose_boundary_path(ai, options, *, weights):
    guard_y = ai.guard_post[1] if ai.guard_post is not None else None
    scores = []
    for steps, entry, _path in options:
        guard_dist = abs(int(entry[1]) - int(guard_y)) if guard_y is not None else 0
        scores.append(score_boundary_path(int(steps), guard_dist, weights=weights))
    idx = argmin(scores)
    if idx is None:
        return None
    return options[idx]


def predict_intruder_goal_path(ai, intruder, *, my_flags, prisons, weights):
    start = ai._pos(intruder)

    if intruder.get("hasFlag"):
        options = ai._best_two_boundary_paths_safe(start)
        if not options:
            return None
        best = _choose_boundary_path(ai, options, weights=weights)
        return best[2] if best else None

    goals = []
    if my_flags:
        goals.extend(ai._flag_pos(f) for f in my_flags)
    if prisons:
        goals.extend(list(prisons))

    if not goals:
        return None

    path = ai._route_any(start, goals, restrict_safe=True)
    if not path:
        path = ai._route_any(start, goals, restrict_safe=False)
    return path or None


def best_intercept_on_path(ai, pursuer_start, opponent_path, *, horizon, arrival_slack, weights):
    if not opponent_path:
        return None
    if pursuer_start == opponent_path[0]:
        return InterceptOption(
            cell=opponent_path[0],
            path=[pursuer_start],
            index=0,
            margin=0,
            steps=0,
            score=0.0,
        )

    max_i = min(int(horizon), len(opponent_path) - 1)
    candidates = []
    scores = []
    for i in range(1, max_i + 1):
        cell = opponent_path[i]
        path = ai._route(pursuer_start, cell, restrict_safe=True)
        if not path:
            continue
        steps = len(path) - 1
        if steps > i + int(arrival_slack):
            continue
        margin = i - steps
        score = score_intercept(steps, margin, i, weights=weights)
        candidates.append((cell, path, i, margin, steps, score))
        scores.append(score)

    idx = argmin(scores)
    if idx is None:
        return None

    cell, path, index, margin, steps, score = candidates[idx]
    return InterceptOption(
        cell=cell,
        path=path,
        index=index,
        margin=margin,
        steps=steps,
        score=score,
    )
