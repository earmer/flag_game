import asyncio
import importlib.machinery
import importlib.util
from pathlib import Path

from lib.game_engine import run_game_server


def _load_nt2_module():
    path = Path(__file__).with_name("pick_flag_elite_ai-nt2.py")
    loader = importlib.machinery.SourceFileLoader("pick_flag_elite_ai_nt2", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Missing loader for {path}")
    spec.loader.exec_module(module)
    return module


_NT2 = _load_nt2_module()
EliteCTFAI_NT2 = _NT2.EliteCTFAI_NT2
BETTER_LURE = bool(getattr(_NT2, "BETTER_LURE", True))


class EliteCTFAI_NT3(EliteCTFAI_NT2):
    def __init__(self, show_gap_in_msec=1000.0):
        super().__init__(show_gap_in_msec=show_gap_in_msec)

        self.intercept_horizon_trap = 14
        self.intercept_horizon_carrier = 10
        self.intercept_arrival_slack = 0

        self.carrier_chase_detour_slack = 2

    def _predict_intruder_goal_path(self, intruder, *, my_flags, prisons):
        start = self._pos(intruder)

        if intruder.get("hasFlag"):
            options = self._best_two_boundary_paths_safe(start)
            if not options:
                return None
            return options[0][2]

        goals = []
        if my_flags:
            goals.extend(self._flag_pos(f) for f in my_flags)
        if prisons:
            goals.extend(list(prisons))

        if not goals:
            return None

        path = self._route_any(start, goals, restrict_safe=True)
        if not path:
            path = self._route_any(start, goals, restrict_safe=False)
        return path or None

    def _best_intercept_on_path(self, pursuer_start, opponent_path, *, horizon):
        if not opponent_path:
            return None
        if pursuer_start == opponent_path[0]:
            return opponent_path[0], [pursuer_start], 0, 0

        best = None
        best_key = None
        max_i = min(int(horizon), len(opponent_path) - 1)
        for i in range(1, max_i + 1):
            cell = opponent_path[i]
            path = self._route(pursuer_start, cell, restrict_safe=True)
            if not path:
                continue
            steps = len(path) - 1
            if steps > i + int(self.intercept_arrival_slack):
                continue
            margin = i - steps
            key = (steps, -margin, i)
            if best_key is None or key < best_key:
                best_key = key
                best = (cell, path, i, margin)
        return best

    def _move_carrier(self, player, opponents, targets):
        start = self._pos(player)
        name = player["name"]

        if not self._is_safe(start):
            return super()._move_carrier(player, opponents, targets)

        direct_steps = self._min_steps_any(start, list(targets or []), restrict_safe=True) if targets else None
        if direct_steps is not None and direct_steps <= 1:
            self._carrier_chase.pop(name, None)
            return super()._move_carrier(player, opponents, targets)

        opponents_our_side = [o for o in opponents or [] if self._is_safe(self._pos(o))]

        best = None
        best_key = None
        for o in opponents_our_side:
            o_path = self._predict_intruder_goal_path(o, my_flags=self.world.list_flags(mine=True, canPickup=True), prisons=self.world.list_prisons(mine=True))
            if not o_path:
                continue
            intercept = self._best_intercept_on_path(start, o_path, horizon=self.intercept_horizon_carrier)
            if not intercept:
                continue
            cell, path_to_cell, _oi, margin = intercept
            to_intercept = len(path_to_cell) - 1

            if targets:
                after_steps = self._min_steps_any(cell, list(targets), restrict_safe=True)
                if after_steps is None:
                    continue
                if direct_steps is not None and (to_intercept + after_steps) > direct_steps + int(self.carrier_chase_detour_slack):
                    continue

            key = (to_intercept, -margin, o["name"])
            if best_key is None or key < best_key:
                best_key = key
                best = (o, cell, path_to_cell)

        if best is not None:
            target, cell, path_to_cell = best
            self._carrier_chase[name] = {
                "target": target["name"],
                "until": self.tick + int(self.carrier_chase_min_ticks),
                "last_pos": self._pos(target),
                "intercept": cell,
            }
            move = self._next_move(start, path_to_cell)
            if move:
                return move

        chase_state = self._carrier_chase.get(name)
        if chase_state and self.tick <= int(chase_state.get("until", -1)):
            opponents_by_name = {o["name"]: o for o in opponents_our_side}
            target = opponents_by_name.get(chase_state.get("target"))
            if target is not None:
                o_path = self._predict_intruder_goal_path(
                    target,
                    my_flags=self.world.list_flags(mine=True, canPickup=True),
                    prisons=self.world.list_prisons(mine=True),
                )
                if o_path:
                    intercept = self._best_intercept_on_path(start, o_path, horizon=self.intercept_horizon_carrier)
                    if intercept:
                        _cell, path_to_cell, _oi, _margin = intercept
                        move = self._next_move(start, path_to_cell)
                        if move:
                            chase_state["last_pos"] = self._pos(target)
                            return move
            last_pos = chase_state.get("last_pos")
            if last_pos is not None:
                path = self._route(start, last_pos, restrict_safe=True)
                move = self._next_move(start, path)
                if move:
                    return move
        else:
            self._carrier_chase.pop(name, None)

        return super()._move_carrier(player, opponents, targets)

    def _plan_trap_on_carriers(self, runners, intruder_carriers, intruders, my_flags):
        actions = {}
        reserved = set()

        if not runners or not intruder_carriers or not self.boundary_ys:
            return actions, reserved

        my_flag_positions = [self._flag_pos(f) for f in my_flags] if my_flags else []
        intruders_close = []
        if intruders and my_flag_positions:
            for o in intruders:
                dist = self._min_steps_any(self._pos(o), my_flag_positions, restrict_safe=True)
                if dist is not None and dist <= 3:
                    intruders_close.append(o)

        offense_threat_count = len(intruder_carriers) + len(intruders_close)
        defense_count = len(runners)
        defense_advantage = defense_count >= offense_threat_count + int(self.double_team_advantage_margin)

        carrier_infos = []
        for o in intruder_carriers:
            o_pos = self._pos(o)
            options = self._best_two_boundary_paths_safe(o_pos)
            if not options:
                continue
            carrier_infos.append((int(options[0][0]), o, options))
        carrier_infos.sort(key=lambda item: item[0])

        available = list(runners)
        for idx, (escape_steps, carrier, options) in enumerate(carrier_infos):
            if not available:
                break

            carrier_pos = self._pos(carrier)

            best_primary = None
            best_key = None
            for p in available:
                p_pos = self._pos(p)
                for _dist, _entry, path in options:
                    intercept = self._best_intercept_on_path(p_pos, path, horizon=self.intercept_horizon_trap)
                    if not intercept:
                        continue
                    cell, path_to_cell, oi, margin = intercept
                    key = (len(path_to_cell) - 1, -margin, oi, p["name"])
                    if best_key is None or key < best_key:
                        best_key = key
                        best_primary = (p, cell, path_to_cell)

            if best_primary is None:
                info = self._zone_intercept_info(carrier_pos)
                if info and info.get("intercept_cell") is not None:
                    cell = info["intercept_cell"]
                    best_runner, path = self._closest_runner_to_cell(available, cell)
                    if best_runner is not None and path:
                        move = self._next_move(self._pos(best_runner), path)
                        if move:
                            actions[best_runner["name"]] = move
                        reserved.add(best_runner["name"])
                        available = [p for p in available if p["name"] != best_runner["name"]]
                continue

            primary, _cell, primary_path = best_primary
            move = self._next_move(self._pos(primary), primary_path)
            if move:
                actions[primary["name"]] = move
            reserved.add(primary["name"])
            available = [p for p in available if p["name"] != primary["name"]]

            urgent = int(escape_steps) <= 2
            remaining_carriers = len(carrier_infos) - idx - 1
            other_threats = int(remaining_carriers) + len(intruders_close)
            allow_double_team = urgent or defense_advantage
            if other_threats > 0 and (len(available) - 1) < other_threats:
                allow_double_team = False
            if not allow_double_team or not available:
                continue

            best_secondary = None
            best_skey = None
            alt_paths = [options[1][2]] if len(options) >= 2 else [options[0][2]]
            for p in available:
                p_pos = self._pos(p)
                for path in alt_paths:
                    intercept = self._best_intercept_on_path(p_pos, path, horizon=self.intercept_horizon_trap)
                    if not intercept:
                        continue
                    cell, path_to_cell, oi, margin = intercept
                    skey = (len(path_to_cell) - 1, -margin, oi, p["name"])
                    if best_skey is None or skey < best_skey:
                        best_skey = skey
                        best_secondary = (p, cell, path_to_cell)

            if best_secondary is None:
                continue

            secondary, _cell, secondary_path = best_secondary
            move = self._next_move(self._pos(secondary), secondary_path)
            if move:
                actions[secondary["name"]] = move
            reserved.add(secondary["name"])
            available = [p for p in available if p["name"] != secondary["name"]]

        return actions, reserved


AI = EliteCTFAI_NT3()


def start_game(req):
    AI.start_game(req)


def plan_next_actions(req):
    return AI.plan_next_actions(req)


def game_over(req):
    AI.game_over(req)


async def main():
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <port>")
        print(f"Example: python3 {sys.argv[0]} 8080")
        sys.exit(1)

    port = int(sys.argv[1])
    print(f"AI backend running on port {port} ...")

    try:
        await run_game_server(port, start_game, plan_next_actions, game_over)
    except Exception as exc:
        print(f"Server stopped: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

