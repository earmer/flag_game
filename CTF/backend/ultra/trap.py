from .accel import argmin
from .weights import score_double_team, score_trap_candidate


def _best_runner_intercept(ai, runners, paths, escape_steps, config, *, secondary):
    candidates = []
    scores = []
    for p in runners:
        p_pos = ai._pos(p)
        for path in paths:
            intercept = ai._best_intercept_on_path(
                p_pos,
                path,
                horizon=config.intercept_horizon_trap,
            )
            if not intercept:
                continue
            score = score_trap_candidate(
                intercept.steps,
                escape_steps,
                intercept.margin,
                intercept.index,
                weights=config.weights.trap,
                secondary=secondary,
            )
            candidates.append((p, intercept, score))
            scores.append(score)

    idx = argmin(scores)
    if idx is None:
        return None
    return candidates[idx]


def plan_trap_on_carriers(ai, runners, intruder_carriers, intruders, my_flags, config):
    actions = {}
    reserved = set()

    if not runners or not intruder_carriers or not ai.boundary_ys:
        return actions, reserved

    my_flag_positions = [ai._flag_pos(f) for f in my_flags] if my_flags else []
    intruders_close = []
    if intruders and my_flag_positions:
        for o in intruders:
            dist = ai._min_steps_any(ai._pos(o), my_flag_positions, restrict_safe=True)
            if dist is not None and dist <= int(config.trap_intruder_close_steps):
                intruders_close.append(o)

    carrier_infos = []
    for o in intruder_carriers:
        o_pos = ai._pos(o)
        options = ai._best_two_boundary_paths_safe(o_pos)
        if not options:
            continue
        carrier_infos.append((int(options[0][0]), o, options))
    carrier_infos.sort(key=lambda item: item[0])

    available = list(runners)
    for idx, (escape_steps, carrier, options) in enumerate(carrier_infos):
        if not available:
            break

        carrier_pos = ai._pos(carrier)
        paths = [opt[2] for opt in options]

        best_primary = _best_runner_intercept(
            ai,
            available,
            paths,
            escape_steps,
            config,
            secondary=False,
        )

        if best_primary is None:
            info = ai._zone_intercept_info(carrier_pos)
            if info and info.get("intercept_cell") is not None:
                cell = info["intercept_cell"]
                best_runner, path = ai._closest_runner_to_cell(available, cell)
                if best_runner is not None and path:
                    move = ai._next_move(ai._pos(best_runner), path)
                    if move:
                        actions[best_runner["name"]] = move
                    reserved.add(best_runner["name"])
                    available = [p for p in available if p["name"] != best_runner["name"]]
            continue

        primary, primary_intercept, _score = best_primary
        move = ai._next_move(ai._pos(primary), primary_intercept.path)
        if move:
            actions[primary["name"]] = move
        reserved.add(primary["name"])
        available = [p for p in available if p["name"] != primary["name"]]

        urgent = int(escape_steps) <= int(config.trap_urgent_escape_steps)
        remaining_carriers = len(carrier_infos) - idx - 1
        other_threats = int(remaining_carriers) + len(intruders_close)

        advantage = len(available) - other_threats
        risk = max(0, other_threats - (len(available) - 1))
        double_team_score = score_double_team(
            advantage,
            1 if urgent else 0,
            risk,
            weights=config.weights.double_team,
        )
        allow_double_team = double_team_score >= float(config.double_team_score_threshold)
        if advantage < -int(config.double_team_advantage_margin):
            allow_double_team = False
        if other_threats > 0 and (len(available) - 1) < other_threats:
            allow_double_team = False
        if not allow_double_team or not available:
            continue

        alt_paths = [options[1][2]] if len(options) >= 2 else [options[0][2]]
        best_secondary = _best_runner_intercept(
            ai,
            available,
            alt_paths,
            escape_steps,
            config,
            secondary=True,
        )
        if best_secondary is None:
            continue

        secondary, secondary_intercept, _score = best_secondary
        move = ai._next_move(ai._pos(secondary), secondary_intercept.path)
        if move:
            actions[secondary["name"]] = move
        reserved.add(secondary["name"])
        available = [p for p in available if p["name"] != secondary["name"]]

    return actions, reserved
