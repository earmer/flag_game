import asyncio
import importlib.machinery
import importlib.util
from pathlib import Path

from lib.game_engine import run_game_server


def _load_nt_module():
    path = Path(__file__).with_name("pick_flag_elite_ai-nt.py")
    loader = importlib.machinery.SourceFileLoader("pick_flag_elite_ai_nt", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Missing loader for {path}")
    spec.loader.exec_module(module)
    return module


_NT = _load_nt_module()
EliteCTFAI_NT = _NT.EliteCTFAI_NT
BETTER_LURE = bool(getattr(_NT, "BETTER_LURE", True))


class EliteCTFAI_NT2(EliteCTFAI_NT):
    def __init__(self, show_gap_in_msec=1000.0):
        super().__init__(show_gap_in_msec=show_gap_in_msec)

        self.zone_press_enabled = True
        self.zone_press_depth = 2
        self.zone_press_mid_slack = 2
        self.zone_press_time_buffer = 0

        self.lure_abort_flag_buffer = 2

        self.line_threat_radius = 1

        self.double_team_advantage_margin = 2

    def _defense_delta(self):
        return -1 if self.my_side_is_left else 1

    def _retreat_move(self):
        return "left" if self.my_side_is_left else "right"

    def _cross_move(self):
        return "right" if self.my_side_is_left else "left"

    def _best_two_boundary_paths_safe(self, start):
        if not self.boundary_ys:
            return []
        results = []
        for y in self.boundary_ys:
            entry = (self.our_boundary_x, y)
            path = self._route(start, entry, restrict_safe=True)
            if not path:
                continue
            results.append((len(path) - 1, entry, path))
        results.sort(key=lambda item: item[0])
        return results[:2]

    def _zone_intercept_info(self, carrier_pos):
        options = self._best_two_boundary_paths_safe(carrier_pos)
        if not options:
            return None

        escape_steps, entry, path = options[0]
        lane_y = int(entry[1])

        intercept_cell = None
        carrier_to_intercept_steps = None
        if (
            self.zone_press_enabled
            and len(options) >= 2
            and int(options[1][0]) <= int(escape_steps) + int(self.zone_press_mid_slack)
        ):
            y_mid = int(round((int(entry[1]) + int(options[1][1][1])) / 2))
            intercept_cell = self._staging_cell(y_mid, max_depth=int(self.zone_press_depth))
        if intercept_cell is None and self.zone_press_enabled:
            intercept_cell = self._staging_cell(lane_y, max_depth=int(self.zone_press_depth))
        if intercept_cell is None:
            idx = max(1, min(len(path) - 2, (len(path) - 1) // 2))
            intercept_cell = path[idx]
            carrier_to_intercept_steps = idx
        else:
            carrier_to_intercept_steps = self._min_steps(carrier_pos, intercept_cell, restrict_safe=True)

        return {
            "lane_y": lane_y,
            "escape_steps": escape_steps,
            "entry": entry,
            "path": path,
            "intercept_cell": intercept_cell,
            "carrier_to_intercept_steps": carrier_to_intercept_steps,
        }

    def _lane_is_hot(self, lane_y, opponents_free, *, radius):
        if lane_y is None:
            return False
        bait_cell = (self.enemy_boundary_x, int(lane_y))
        if bait_cell in self.world.walls:
            return True

        enemies_enemy_side = [self._pos(o) for o in opponents_free if not self._is_safe(self._pos(o))]
        if not enemies_enemy_side:
            return False
        dist = self._min_steps_from_any(enemies_enemy_side, bait_cell, restrict_safe=False)
        if dist is None:
            return False
        return dist <= int(radius)

    def _boundary_slide_move(self, start, opponents_free, *, prefer_y=None):
        x, y = start
        if x != self.our_boundary_x:
            return None
        candidates = []
        for dy, move in ((-1, "up"), (1, "down")):
            ny = y + dy
            cell = (x, ny)
            if not (0 <= ny < self.world.height):
                continue
            if cell in self.world.walls:
                continue
            if not self._is_safe(cell):
                continue
            if self._lane_is_hot(ny, opponents_free, radius=self.line_threat_radius):
                continue
            score = 0
            if prefer_y is not None:
                score -= abs(int(ny) - int(prefer_y))
            candidates.append((score, move))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

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
            info = self._zone_intercept_info(o_pos)
            if not info:
                continue
            carrier_infos.append((int(info["escape_steps"]), o, info))
        carrier_infos.sort(key=lambda item: item[0])

        available = list(runners)
        for idx, (escape_dist, carrier, info) in enumerate(carrier_infos):
            if not available:
                break

            carrier_pos = self._pos(carrier)
            lane_y = int(info["lane_y"])
            intercept_cell = info["intercept_cell"]
            carrier_to_intercept_steps = info["carrier_to_intercept_steps"]

            pressor = None
            press_path = None
            if intercept_cell is not None:
                pressor, press_path = self._closest_runner_to_cell(available, intercept_cell)

            chaser = None
            chase_path = None
            for p in available:
                path = self._route(self._pos(p), carrier_pos, restrict_safe=True)
                if path and (chase_path is None or len(path) < len(chase_path)):
                    chase_path = path
                    chaser = p

            block_cell = (self.our_boundary_x, lane_y)
            blocker = None
            blocker_path = None
            for p in available:
                path = self._route(self._pos(p), block_cell, restrict_safe=True)
                if path and (blocker_path is None or len(path) < len(blocker_path)):
                    blocker_path = path
                    blocker = p

            can_press = False
            if (
                pressor is not None
                and press_path is not None
                and carrier_to_intercept_steps is not None
            ):
                press_steps = len(press_path) - 1
                can_press = (
                    press_steps + int(self.zone_press_time_buffer) <= int(carrier_to_intercept_steps)
                )

            primary = None
            primary_path = None
            primary_kind = None
            if can_press:
                primary = pressor
                primary_path = press_path
                primary_kind = "press"
            elif chaser is not None and chase_path is not None:
                primary = chaser
                primary_path = chase_path
                primary_kind = "chase"
            elif blocker is not None and blocker_path is not None:
                primary = blocker
                primary_path = blocker_path
                primary_kind = "block"

            if primary is None or primary_path is None:
                continue

            move = self._next_move(self._pos(primary), primary_path)
            if move:
                actions[primary["name"]] = move
            reserved.add(primary["name"])
            available = [p for p in available if p["name"] != primary["name"]]

            urgent = int(escape_dist) <= 2
            remaining_carriers = len(carrier_infos) - idx - 1
            other_threats = int(remaining_carriers) + len(intruders_close)

            allow_double_team = urgent or defense_advantage
            if other_threats > 0 and (len(available) - 1) < other_threats:
                allow_double_team = False

            secondary = None
            secondary_path = None
            if primary_kind != "block" and blocker is not None and blocker_path is not None:
                secondary, secondary_path = blocker, blocker_path
            elif primary_kind != "chase" and chaser is not None and chase_path is not None:
                secondary, secondary_path = chaser, chase_path
            elif primary_kind != "press" and pressor is not None and press_path is not None:
                secondary, secondary_path = pressor, press_path

            if not allow_double_team or not available or secondary is None or secondary_path is None:
                continue
            if secondary["name"] not in {p["name"] for p in available}:
                continue

            move = self._next_move(self._pos(secondary), secondary_path)
            if move:
                actions[secondary["name"]] = move
            reserved.add(secondary["name"])
            available = [p for p in available if p["name"] != secondary["name"]]

        return actions, reserved

    def _plan_active_bait(self, my_free, opponents_free, enemy_flags):
        actions = {}
        reserved = set()

        if not self.bait_enabled:
            return actions, reserved
        if not self._baiting:
            return actions, reserved
        if not self.boundary_ys:
            self._baiting = None
            return actions, reserved

        name = self._baiting.get("name")
        lane_y = self._baiting.get("lane_y")
        ticks_left = int(self._baiting.get("ticks_left") or 0)
        towards_enemy = bool(self._baiting.get("towards_enemy", True))

        if not name or lane_y is None or ticks_left <= 0:
            self._baiting = None
            return actions, reserved

        player = None
        for p in my_free or []:
            if p["name"] == name:
                player = p
                break
        if player is None or player.get("inPrison") or player.get("hasFlag"):
            self._baiting = None
            return actions, reserved

        entry_safe = (self.our_boundary_x, int(lane_y))
        bait_cell = (self.enemy_boundary_x, int(lane_y))
        if entry_safe in self.world.walls or bait_cell in self.world.walls:
            self._baiting = None
            return actions, reserved

        cross_move = self._cross_move()
        return_move = self._retreat_move()

        opponents_enemy_side = [
            self._pos(o) for o in opponents_free if not self._is_safe(self._pos(o))
        ]
        enemy_dist = self._min_steps_from_any(opponents_enemy_side, bait_cell, restrict_safe=False)
        if enemy_dist is None:
            enemy_dist = 10**6

        safe_to_cross = enemy_dist >= self.bait_min_enemy_dist
        lane_hot = self._lane_is_hot(lane_y, opponents_free, radius=self.line_threat_radius)

        start = self._pos(player)
        move = None

        if not self._is_safe(start):
            if safe_to_cross and enemy_flags:
                flag_positions = [self._flag_pos(f) for f in enemy_flags]
                best_flag_path = self._route_any(start, flag_positions, restrict_safe=False)
                move = self._next_move(start, best_flag_path)
                if move:
                    actions[name] = move
                    reserved.add(name)
                    self._baiting["ticks_left"] = max(1, ticks_left - 1)
                    self._baiting["towards_enemy"] = towards_enemy
                    return actions, reserved

            if start == bait_cell:
                move = return_move
                towards_enemy = True
            else:
                path = self._route(start, bait_cell, restrict_safe=False)
                move = self._next_move(start, path) or return_move
        else:
            if not safe_to_cross or lane_hot:
                if start == entry_safe:
                    move = self._boundary_slide_move(start, opponents_free, prefer_y=self.guard_post[1] if self.guard_post else None)
                    if move is None:
                        move = return_move
                else:
                    path = self._route(start, entry_safe, restrict_safe=True)
                    move = self._next_move(start, path)
            else:
                if start == entry_safe:
                    if towards_enemy:
                        move = cross_move
                        towards_enemy = False
                else:
                    path = self._route(start, entry_safe, restrict_safe=True)
                    move = self._next_move(start, path)

        if move:
            actions[name] = move
        reserved.add(name)

        ticks_left -= 1
        if ticks_left <= 0:
            self._baiting = None
        else:
            self._baiting["ticks_left"] = ticks_left
            self._baiting["towards_enemy"] = towards_enemy

        return actions, reserved

    def plan_next_actions(self, req):
        if not self.world.update(req):
            return {}

        self.tick += 1
        self._doomed_this_tick = set()

        my_free = self.world.list_players(mine=True, inPrison=False, hasFlag=None) or []
        if not my_free:
            return {}

        my_prisoners = self.world.list_players(mine=True, inPrison=True, hasFlag=None) or []
        opponents_free = self.world.list_players(mine=False, inPrison=False, hasFlag=None) or []
        enemy_flags = self.world.list_flags(mine=False, canPickup=True) or []
        my_flags = self.world.list_flags(mine=True, canPickup=True) or []

        targets = set(self.world.list_targets(mine=True) or [])
        prisons = set(self.world.list_prisons(mine=True) or [])

        opponent_positions_enemy_side = [
            self._pos(o) for o in opponents_free if not self._is_safe(self._pos(o))
        ]
        my_score = int(req.get("myteamScore", 0) or 0)
        opponent_score = int(req.get("opponentScore", 0) or 0)

        actions = {}
        reserved = set()

        for p in my_free:
            if not p.get("hasFlag"):
                continue
            move = self._move_carrier(p, opponents_free, targets)
            if move:
                actions[p["name"]] = move
            reserved.add(p["name"])

        runners = [p for p in my_free if (not p.get("hasFlag")) and p["name"] not in reserved]

        intruder_carriers = [
            o for o in opponents_free if o.get("hasFlag") and self._is_safe(self._pos(o))
        ]
        intruders = [
            o for o in opponents_free if (not o.get("hasFlag")) and self._is_safe(self._pos(o))
        ]

        if intruder_carriers and runners:
            defense_actions, defense_reserved = self._plan_trap_on_carriers(
                runners,
                intruder_carriers,
                intruders,
                my_flags,
            )
            actions.update(defense_actions)
            reserved |= defense_reserved
            runners = [p for p in runners if p["name"] not in reserved]

        doomed_names = set(self._doomed_this_tick)
        for p in runners:
            p_pos = self._pos(p)
            if self._is_safe(p_pos):
                continue
            if self._escape_almost_impossible(p_pos, opponents_free):
                doomed_names.add(p["name"])

        if not doomed_names:
            self._baiting = None

        avoid_lane_y = None
        if doomed_names:
            for p in my_free:
                if p["name"] not in doomed_names:
                    continue
                if not p.get("hasFlag"):
                    continue
                p_pos = self._pos(p)
                entry, _path = self._best_boundary_entry_path(p_pos, opponents_free)
                if entry:
                    avoid_lane_y = entry[1]
                    break

        need_prison_ready = bool(my_prisoners) or bool(doomed_names)
        if need_prison_ready and prisons and runners:
            if not any(self._pos(p) in prisons for p in my_free):
                prison_actions, prison_reserved = self._plan_prison_ready(runners, prisons)
                actions.update(prison_actions)
                reserved |= prison_reserved
                runners = [p for p in runners if p["name"] not in reserved]

        defended_intruders = False
        if not intruder_carriers and intruders and runners:
            allow_lure = False
            if len(runners) >= self.lure_min_runners and self._lure_is_reasonable(req, my_flags):
                allow_lure = True

                if my_flags and intruders:
                    my_flag_positions = [self._flag_pos(f) for f in my_flags]
                    min_intruder_steps = None
                    for o in intruders:
                        dist = self._min_steps_any(self._pos(o), my_flag_positions, restrict_safe=True)
                        if dist is None:
                            continue
                        if min_intruder_steps is None or dist < min_intruder_steps:
                            min_intruder_steps = dist
                    if min_intruder_steps is not None and min_intruder_steps <= int(self.lure_abort_flag_buffer):
                        allow_lure = False

                if allow_lure and BETTER_LURE and not self._better_lure_is_winning(runners, intruders, my_flags):
                    allow_lure = False
            if allow_lure:
                lure_actions, lure_reserved = self._plan_lure_stage(runners, intruders, my_flags)
                actions.update(lure_actions)
                reserved |= lure_reserved
                runners = [p for p in runners if p["name"] not in reserved]
                defended_intruders = bool(lure_reserved)
            else:
                intruder_actions, intruder_reserved = self._plan_chase_intruder(runners, intruders, my_flags)
                actions.update(intruder_actions)
                reserved |= intruder_reserved
                runners = [p for p in runners if p["name"] not in reserved]
                defended_intruders = bool(intruder_reserved)

        baiter_name = self._baiting.get("name") if self._baiting else None
        if baiter_name and not intruder_carriers:
            runner_names = {p["name"] for p in runners}
            if baiter_name in runner_names:
                bait_actions, bait_reserved = self._plan_active_bait(my_free, opponents_free, enemy_flags)
                actions.update(bait_actions)
                reserved |= bait_reserved
                runners = [p for p in runners if p["name"] not in reserved]
            else:
                self._baiting = None

        if doomed_names and runners and not intruder_carriers and self._baiting is None:
            bait_actions, bait_reserved = self._plan_safe_bait(runners, opponents_free, avoid_lane_y=avoid_lane_y)
            actions.update(bait_actions)
            reserved |= bait_reserved
            runners = [p for p in runners if p["name"] not in reserved]

        if runners and my_flags:
            defense_actions, defense_reserved = self._plan_defend_my_flags(
                runners,
                my_flags,
                opponents_free,
                enemy_flags,
                my_score=my_score,
                opponent_score=opponent_score,
            )
            actions.update(defense_actions)
            reserved |= defense_reserved
            runners = [p for p in runners if p["name"] not in reserved]

        intruder_pressure = False
        if intruder_carriers:
            intruder_pressure = True
        elif intruders and my_flags:
            my_flag_positions = [self._flag_pos(f) for f in my_flags]
            for o in intruders:
                dist = self._min_steps_any(self._pos(o), my_flag_positions, restrict_safe=True)
                if dist is not None and dist <= 4:
                    intruder_pressure = True
                    break

        threat_present = intruder_pressure or (defended_intruders and my_score > opponent_score)
        if (
            self.guard_enabled
            and threat_present
            and runners
            and len(runners) >= self.guard_min_runners
        ):
            safe_runners = [p for p in runners if self._is_safe(self._pos(p))]
            if safe_runners:
                opponent_positions_our_side = [
                    self._pos(o) for o in opponents_free if self._is_safe(self._pos(o))
                ]
                guard_target = None
                if my_flags:
                    flag_positions = [self._flag_pos(f) for f in my_flags]
                    if opponent_positions_our_side:
                        guard_target = min(
                            flag_positions,
                            key=lambda fpos: self._min_steps_from_any(
                                opponent_positions_our_side, fpos, restrict_safe=True
                            )
                            or 10**6,
                        )
                    else:
                        runner_positions = [self._pos(p) for p in safe_runners]
                        guard_target = min(
                            flag_positions,
                            key=lambda fpos: self._min_steps_from_any(
                                runner_positions, fpos, restrict_safe=True
                            )
                            or 10**6,
                        )
                if guard_target is None:
                    guard_target = self.guard_post

                if guard_target is not None:
                    best_guard, path = self._closest_runner_to_cell(safe_runners, guard_target)
                    if best_guard is not None:
                        move = self._next_move(self._pos(best_guard), path)
                        if move:
                            actions[best_guard["name"]] = move
                        reserved.add(best_guard["name"])
                        runners = [p for p in runners if p["name"] != best_guard["name"]]

        attackers = [p for p in runners if p["name"] not in reserved]
        flag_positions = [self._flag_pos(f) for f in enemy_flags]
        assignment = self._assign_flags(attackers, flag_positions, opponent_positions_enemy_side)
        self.flag_assignment = assignment

        avoid1 = self._expanded_enemy_obstacles(opponents_free, self.avoid_radius_runner)
        avoid0 = self._expanded_enemy_obstacles(opponents_free, 0)

        cross_move = self._cross_move()
        for p in attackers:
            target = assignment.get(p["name"])
            if not target:
                continue
            start = self._pos(p)
            move = None
            for avoid in (avoid1, avoid0, None):
                path = self._route(start, target, extra_obstacles=avoid, restrict_safe=False)
                move = self._next_move(start, path)
                if move:
                    break

            if (
                move == cross_move
                and start[0] == self.our_boundary_x
                and self._lane_is_hot(start[1], opponents_free, radius=self.line_threat_radius)
            ):
                slide = self._boundary_slide_move(start, opponents_free, prefer_y=target[1])
                move = slide or self._retreat_move()

            if move:
                actions[p["name"]] = move

        return actions


AI = EliteCTFAI_NT2()


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

