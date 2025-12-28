import asyncio

from lib.game_engine import run_game_server
from pick_flag_elite_ai import EliteCTFAI


BETTER_LURE = True


class EliteCTFAI_NT(EliteCTFAI):
    def __init__(self, show_gap_in_msec=1000.0):
        super().__init__(show_gap_in_msec=show_gap_in_msec)
        self.lure_enabled = True
        self.lure_min_runners = 4
        self.lure_min_flag_depth = 2
        self.lure_stage_depth = 3
        self.better_lure_time_buffer = 1

        self.guard_enabled = True
        self.guard_min_runners = 4
        self.defense_chase_buffer = 1
        self.defense_block_buffer = 0

        self.prison_ready_enabled = True
        self.escape_impossible_buffer = 1
        self.doomed_entry_slack = 1

        self.bait_enabled = True
        self.bait_min_enemy_dist = 3
        self.bait_persist_ticks = 6

        self.endgame_defense_enabled = True
        self.endgame_flags_left = 3
        self.endgame_min_attackers = 1

        self.double_team_max_margin = 1

        self._doomed_this_tick = set()
        self._baiting = None

    def start_game(self, req):
        super().start_game(req)
        self._baiting = None

    def _flag_pos(self, flag):
        return (int(flag["posX"]), int(flag["posY"]))

    def _lure_is_reasonable(self, req, my_flags):
        if not self.lure_enabled:
            return False
        if not my_flags:
            return False
        my_score = int(req.get("myteamScore", 0) or 0)
        opponent_score = int(req.get("opponentScore", 0) or 0)
        if opponent_score > my_score:
            return False
        min_depth = min(abs(self._flag_pos(f)[0] - self.our_boundary_x) for f in my_flags)
        return min_depth >= self.lure_min_flag_depth

    def _min_steps(self, start, goal, *, restrict_safe):
        path = self._route(start, goal, restrict_safe=restrict_safe)
        return (len(path) - 1) if path else None

    def _min_steps_any(self, start, goals, *, restrict_safe):
        best = None
        for goal in goals or []:
            steps = self._min_steps(start, goal, restrict_safe=restrict_safe)
            if steps is None:
                continue
            if best is None or steps < best:
                best = steps
        return best

    def _min_steps_from_any(self, starts, goal, *, restrict_safe):
        best = None
        for start in starts or []:
            steps = self._min_steps(start, goal, restrict_safe=restrict_safe)
            if steps is None:
                continue
            if best is None or steps < best:
                best = steps
        return best

    def _boundary_entries(self):
        return [(self.our_boundary_x, y) for y in (self.boundary_ys or [])]

    def _best_boundary_entry_path(self, start, opponents):
        if not self.boundary_ys:
            return None, None

        enemy_positions = [self._pos(o) for o in opponents if not self._is_safe(self._pos(o))]
        avoid0 = self._expanded_enemy_obstacles(opponents, 0)

        candidates = []
        for y in self.boundary_ys:
            entry = (self.our_boundary_x, y)
            path = self._route(start, entry, extra_obstacles=avoid0, restrict_safe=False)
            if not path:
                path = self._route(start, entry, restrict_safe=False)
            if not path:
                continue

            steps = len(path) - 1
            enemy_dist = self._min_steps_from_any(enemy_positions, entry, restrict_safe=False)
            if enemy_dist is None:
                enemy_dist = 10**6
            candidates.append((steps, -enemy_dist, entry, path))

        if not candidates:
            return None, None

        min_steps = min(item[0] for item in candidates)
        slack = max(0, int(self.doomed_entry_slack))
        shortlist = [item for item in candidates if item[0] <= min_steps + slack]
        shortlist.sort(key=lambda item: (item[0], item[1]))
        _steps, _neg_enemy_dist, entry, path = shortlist[0]
        return entry, path

    def _escape_almost_impossible(self, start, opponents):
        if self._is_safe(start):
            return False
        if not self.boundary_ys:
            return False
        enemy_positions = [self._pos(o) for o in opponents if not self._is_safe(self._pos(o))]
        if not enemy_positions:
            return False

        path_to_boundary = self._route_any(start, self._boundary_entries(), restrict_safe=False)
        if not path_to_boundary:
            return False
        my_escape_steps = len(path_to_boundary) - 1

        enemy_catch_steps = self._min_steps_from_any(enemy_positions, start, restrict_safe=False)
        if enemy_catch_steps is None:
            return False

        return enemy_catch_steps + self.escape_impossible_buffer <= my_escape_steps

    def _plan_prison_ready(self, runners, prisons):
        actions = {}
        reserved = set()

        if not self.prison_ready_enabled:
            return actions, reserved
        if not runners or not prisons:
            return actions, reserved

        best_runner = None
        best_path = None
        for p in runners:
            p_pos = self._pos(p)
            if p_pos in prisons:
                reserved.add(p["name"])
                return actions, reserved
            path = self._route_any(p_pos, prisons, restrict_safe=True)
            if not path:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path
                best_runner = p

        if best_runner is None or best_path is None:
            return actions, reserved

        move = self._next_move(self._pos(best_runner), best_path)
        if move:
            actions[best_runner["name"]] = move
        reserved.add(best_runner["name"])
        return actions, reserved

    def _plan_safe_bait(self, runners, opponents, *, avoid_lane_y=None):
        actions = {}
        reserved = set()

        if not self.bait_enabled:
            return actions, reserved
        if self._baiting is not None:
            return actions, reserved
        if not runners or not self.boundary_ys:
            return actions, reserved

        opponents_enemy_side = [self._pos(o) for o in opponents if not self._is_safe(self._pos(o))]
        if not opponents_enemy_side:
            return actions, reserved

        cross_move = "right" if self.my_side_is_left else "left"

        best = None
        best_score = None
        for y in self.boundary_ys:
            entry_safe = (self.our_boundary_x, y)
            bait_cell = (self.enemy_boundary_x, y)
            if entry_safe in self.world.walls or bait_cell in self.world.walls:
                continue

            enemy_dist = self._min_steps_from_any(opponents_enemy_side, bait_cell, restrict_safe=False)
            if enemy_dist is None:
                enemy_dist = 10**6
            if enemy_dist < self.bait_min_enemy_dist:
                continue

            lane_sep = abs(y - avoid_lane_y) if avoid_lane_y is not None else 0

            for p in runners:
                p_pos = self._pos(p)
                if not self._is_safe(p_pos):
                    continue
                path = self._route(p_pos, entry_safe, restrict_safe=True)
                if not path:
                    continue
                dist = len(path) - 1

                score = 2.0 * lane_sep + 0.5 * enemy_dist - 0.6 * dist
                if best_score is None or score > best_score:
                    best_score = score
                    best = (p, entry_safe)

        if best is None:
            return actions, reserved

        runner, entry_safe = best
        self._baiting = {
            "name": runner["name"],
            "lane_y": entry_safe[1],
            "ticks_left": int(self.bait_persist_ticks),
            "towards_enemy": True,
        }

        runner_pos = self._pos(runner)
        if runner_pos == entry_safe:
            actions[runner["name"]] = cross_move
            reserved.add(runner["name"])
            return actions, reserved

        path = self._route(runner_pos, entry_safe, restrict_safe=True)
        move = self._next_move(runner_pos, path)
        if move:
            actions[runner["name"]] = move
        reserved.add(runner["name"])
        return actions, reserved

    def _plan_active_bait(self, my_free, opponents_free):
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

        cross_move = "right" if self.my_side_is_left else "left"
        return_move = "left" if self.my_side_is_left else "right"

        opponents_enemy_side = [
            self._pos(o) for o in opponents_free if not self._is_safe(self._pos(o))
        ]
        enemy_dist = self._min_steps_from_any(opponents_enemy_side, bait_cell, restrict_safe=False)
        if enemy_dist is None:
            enemy_dist = 10**6
        safe_to_cross = enemy_dist >= self.bait_min_enemy_dist

        start = self._pos(player)
        move = None
        if not self._is_safe(start):
            if start == bait_cell:
                move = return_move
                towards_enemy = True
            else:
                path = self._route(start, bait_cell, restrict_safe=False)
                move = self._next_move(start, path) or return_move
        else:
            if not safe_to_cross:
                if start != entry_safe:
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

    def _plan_defend_my_flags(self, runners, my_flags, opponents_free, enemy_flags, *, my_score, opponent_score):
        actions = {}
        reserved = set()

        if not self.endgame_defense_enabled:
            return actions, reserved
        if not runners or not my_flags:
            return actions, reserved
        if len(my_flags) > self.endgame_flags_left:
            return actions, reserved

        min_attackers = int(self.endgame_min_attackers)
        if my_score < opponent_score and len(runners) >= 3:
            min_attackers = max(min_attackers, 2)
        if not enemy_flags:
            min_attackers = 0

        defenders_budget = max(0, len(runners) - min_attackers)
        if defenders_budget <= 0:
            return actions, reserved

        flag_positions = [self._flag_pos(f) for f in my_flags]
        opponents_on_our_side = [self._pos(o) for o in opponents_free if self._is_safe(self._pos(o))]

        def threat_steps(flag_pos):
            steps = self._min_steps_from_any(opponents_on_our_side, flag_pos, restrict_safe=True)
            return steps if steps is not None else 10**6

        threat_by_flag = {pos: threat_steps(pos) for pos in flag_positions}

        defender_candidates = []
        for p in runners:
            p_pos = self._pos(p)
            best_flag = None
            best_path = None
            best_threat = None
            for fpos in flag_positions:
                path = self._route(p_pos, fpos, restrict_safe=True)
                if not path:
                    continue
                steps = len(path) - 1
                threat = threat_by_flag.get(fpos, 10**6)
                score = (steps, threat)
                if best_path is None or score < (len(best_path) - 1, best_threat):
                    best_flag = fpos
                    best_path = path
                    best_threat = threat
            if best_flag is None or best_path is None:
                continue
            defender_candidates.append((len(best_path) - 1, best_threat, p, best_flag, best_path))

        defender_candidates.sort(key=lambda item: (item[0], item[1]))
        for _steps, _threat, p, _flag, path in defender_candidates[:defenders_budget]:
            move = self._next_move(self._pos(p), path)
            if move:
                actions[p["name"]] = move
            reserved.add(p["name"])

        return actions, reserved

    def _move_carrier(self, player, opponents, targets):
        start = self._pos(player)

        if self._is_safe(start):
            path = self._route_any(start, targets, restrict_safe=True)
            return self._next_move(start, path)

        doomed = self._escape_almost_impossible(start, opponents)
        if doomed:
            self._doomed_this_tick.add(player["name"])
            _entry, path = self._best_boundary_entry_path(start, opponents)
            move = self._next_move(start, path)
            if move:
                return move

        return super()._move_carrier(player, opponents, targets)

    def _better_lure_is_winning(self, runners, intruders, my_flags):
        if not runners or not intruders or not my_flags:
            return False
        if len(runners) < 2:
            return False
        if not self.boundary_ys:
            return False

        flag_positions = [self._flag_pos(f) for f in my_flags]

        best_intruder = None
        best_flag = None
        best_pick_steps = None
        for o in intruders:
            o_pos = self._pos(o)
            for fpos in flag_positions:
                steps = self._min_steps(o_pos, fpos, restrict_safe=True)
                if steps is None:
                    continue
                if best_pick_steps is None or steps < best_pick_steps:
                    best_pick_steps = steps
                    best_intruder = o
                    best_flag = fpos

        if best_intruder is None or best_flag is None or best_pick_steps is None:
            return False

        fastest_lane_y = None
        fastest_escape_steps = None
        for y in self.boundary_ys:
            boundary_cell = (self.our_boundary_x, y)
            steps_flag_to_boundary = self._min_steps(best_flag, boundary_cell, restrict_safe=True)
            if steps_flag_to_boundary is None:
                continue
            total = best_pick_steps + steps_flag_to_boundary
            if fastest_escape_steps is None or total < fastest_escape_steps:
                fastest_escape_steps = total
                fastest_lane_y = y

        if fastest_lane_y is None or fastest_escape_steps is None:
            return False

        boundary_cell = (self.our_boundary_x, fastest_lane_y)
        best_our_steps = None
        for p in runners:
            steps = self._min_steps(self._pos(p), boundary_cell, restrict_safe=True)
            if steps is None:
                continue
            if best_our_steps is None or steps < best_our_steps:
                best_our_steps = steps

        if best_our_steps is None:
            return False

        return best_our_steps + self.better_lure_time_buffer <= fastest_escape_steps

    def _staging_cell(self, lane_y, *, max_depth):
        delta = -1 if self.my_side_is_left else 1
        for depth in range(1, max_depth + 1):
            x = self.our_boundary_x + delta * depth
            cell = (x, lane_y)
            if 0 <= x < self.world.width and 0 <= lane_y < self.world.height and cell not in self.world.walls:
                if self._is_safe(cell):
                    return cell
        cell = (self.our_boundary_x, lane_y)
        if cell not in self.world.walls and self._is_safe(cell):
            return cell
        return None

    def _closest_runner_to_cell(self, runners, cell):
        best_runner = None
        best_path = []
        for p in runners:
            path = self._route(self._pos(p), cell, restrict_safe=True)
            if path and (not best_path or len(path) < len(best_path)):
                best_path = path
                best_runner = p
        return best_runner, best_path

    def _best_block_and_chase_pair(self, runners, block_cell, carrier_pos):
        best = None
        best_total = None
        best_block_path = None
        best_chase_path = None

        for blocker in runners:
            blocker_path = self._route(self._pos(blocker), block_cell, restrict_safe=True)
            if not blocker_path:
                continue
            for chaser in runners:
                if chaser["name"] == blocker["name"]:
                    continue
                chase_path = self._route(self._pos(chaser), carrier_pos, restrict_safe=True)
                if not chase_path:
                    continue
                total = (len(blocker_path) - 1) + (len(chase_path) - 1)
                if best_total is None or total < best_total:
                    best_total = total
                    best = (blocker, chaser)
                    best_block_path = blocker_path
                    best_chase_path = chase_path

        if best is None:
            return None
        return best[0], best[1], best_block_path, best_chase_path

    def _plan_trap_on_carriers(self, runners, intruder_carriers):
        actions = {}
        reserved = set()

        carrier_infos = []
        for o in intruder_carriers:
            o_pos = self._pos(o)
            lane_y, escape_dist = self._predict_escape_lane(o_pos)
            if lane_y is None or escape_dist is None:
                continue
            carrier_infos.append((escape_dist, o, lane_y))
        carrier_infos.sort(key=lambda item: item[0])

        if not carrier_infos:
            return actions, reserved

        available = list(runners)

        for escape_dist, carrier, lane_y in carrier_infos:
            if not available:
                break

            carrier_pos = self._pos(carrier)
            block_cell = (self.our_boundary_x, lane_y)

            chaser = None
            chase_path = None
            for p in available:
                path = self._route(self._pos(p), carrier_pos, restrict_safe=True)
                if path and (chase_path is None or len(path) < len(chase_path)):
                    chase_path = path
                    chaser = p

            blocker = None
            blocker_path = None
            for p in available:
                path = self._route(self._pos(p), block_cell, restrict_safe=True)
                if path and (blocker_path is None or len(path) < len(blocker_path)):
                    blocker_path = path
                    blocker = p

            chase_steps = (len(chase_path) - 1) if chase_path else None
            block_steps = (len(blocker_path) - 1) if blocker_path else None

            chase_margin = None
            if chase_steps is not None and escape_dist is not None:
                chase_margin = (chase_steps + self.defense_chase_buffer) - escape_dist

            block_margin = None
            if block_steps is not None and escape_dist is not None:
                block_margin = block_steps - (escape_dist - self.defense_block_buffer)

            primary = None
            primary_path = None
            secondary = None
            secondary_path = None
            secondary_margin = None

            if chase_margin is not None and chase_margin <= 0:
                primary = chaser
                primary_path = chase_path
            elif block_margin is not None and block_margin <= 0:
                primary = blocker
                primary_path = blocker_path
            else:
                choice = []
                if chase_margin is not None and chaser is not None and chase_path is not None:
                    choice.append((chase_margin, chaser, chase_path, "chase"))
                if block_margin is not None and blocker is not None and blocker_path is not None:
                    choice.append((block_margin, blocker, blocker_path, "block"))
                if not choice:
                    continue
                choice.sort(key=lambda item: item[0])
                _margin, primary, primary_path, kind = choice[0]

                if kind == "chase":
                    secondary = blocker
                    secondary_path = blocker_path
                    secondary_margin = block_margin
                else:
                    secondary = chaser
                    secondary_path = chase_path
                    secondary_margin = chase_margin

            if primary is None or primary_path is None:
                continue

            move = self._next_move(self._pos(primary), primary_path)
            if move:
                actions[primary["name"]] = move
            reserved.add(primary["name"])
            available = [p for p in available if p["name"] != primary["name"]]

            urgent = escape_dist is not None and escape_dist <= 2
            allow_double_team = urgent or len(runners) >= 4
            if not urgent and secondary_margin is not None and secondary_margin > self.double_team_max_margin:
                continue
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

    def _choose_lure_lane(self, intruders, my_flags):
        best = None
        best_score = None

        flag_positions = [self._flag_pos(f) for f in my_flags]
        for o in intruders:
            o_pos = self._pos(o)
            best_flag = None
            best_flag_path = []
            for fpos in flag_positions:
                path = self._route(o_pos, fpos, restrict_safe=True)
                if path and (not best_flag_path or len(path) < len(best_flag_path)):
                    best_flag_path = path
                    best_flag = fpos
            if best_flag is None:
                continue
            dist_to_flag = len(best_flag_path) - 1
            lane_y, escape_dist = self._predict_escape_lane(best_flag)
            if lane_y is None:
                continue
            score = dist_to_flag + 0.25 * (escape_dist if escape_dist is not None else 0)
            if best_score is None or score < best_score:
                best_score = score
                best = (lane_y, escape_dist)
        return best

    def _plan_chase_intruder(self, runners, intruders, my_flags):
        actions = {}
        reserved = set()

        if not runners or not intruders:
            return actions, reserved

        flag_positions = [self._flag_pos(f) for f in my_flags]
        if not flag_positions:
            return actions, reserved

        best_intruder = None
        best_intruder_dist = None
        for o in intruders:
            o_pos = self._pos(o)
            dist = self._min_steps_any(o_pos, flag_positions, restrict_safe=True)
            if dist is None:
                continue
            if best_intruder_dist is None or dist < best_intruder_dist:
                best_intruder_dist = dist
                best_intruder = o

        if best_intruder is None or best_intruder_dist is None:
            return actions, reserved

        if best_intruder_dist > 4:
            return actions, reserved

        intruder_pos = self._pos(best_intruder)
        chaser = None
        chase_path = None
        for p in runners:
            path = self._route(self._pos(p), intruder_pos, restrict_safe=True)
            if path and (chase_path is None or len(path) < len(chase_path)):
                chase_path = path
                chaser = p

        if chaser is None or chase_path is None:
            return actions, reserved

        move = self._next_move(self._pos(chaser), chase_path)
        if move:
            actions[chaser["name"]] = move
        reserved.add(chaser["name"])
        return actions, reserved

    def _plan_lure_stage(self, runners, intruders, my_flags):
        actions = {}
        reserved = set()

        lure = self._choose_lure_lane(intruders, my_flags)
        if lure is None:
            return actions, reserved
        lane_y, _escape_dist = lure
        stage_cell = self._staging_cell(lane_y, max_depth=self.lure_stage_depth)
        if stage_cell is None:
            return actions, reserved

        stager, path = self._closest_runner_to_cell(runners, stage_cell)
        if stager is not None:
            move = self._next_move(self._pos(stager), path)
            if move:
                actions[stager["name"]] = move
            reserved.add(stager["name"])

        remaining = [p for p in runners if p["name"] not in reserved]
        if self.guard_post is not None and remaining and len(runners) >= self.lure_min_runners + 1:
            guard_stage = self._staging_cell(self.guard_post[1], max_depth=self.lure_stage_depth)
            if guard_stage is not None:
                backstop, path = self._closest_runner_to_cell(remaining, guard_stage)
                if backstop is not None:
                    move = self._next_move(self._pos(backstop), path)
                    if move:
                        actions[backstop["name"]] = move
                    reserved.add(backstop["name"])

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
            defense_actions, defense_reserved = self._plan_trap_on_carriers(runners, intruder_carriers)
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
                if BETTER_LURE and not self._better_lure_is_winning(runners, intruders, my_flags):
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
                bait_actions, bait_reserved = self._plan_active_bait(my_free, opponents_free)
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
            if move:
                actions[p["name"]] = move

        return actions


AI = EliteCTFAI_NT()


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
