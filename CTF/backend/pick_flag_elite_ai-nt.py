import asyncio

from lib.game_engine import run_game_server
from pick_flag_elite_ai import EliteCTFAI


BETTER_LURE = True


class EliteCTFAI_NT(EliteCTFAI):
    def __init__(self, show_gap_in_msec=1000.0):
        super().__init__(show_gap_in_msec=show_gap_in_msec)
        self.lure_enabled = True
        self.lure_min_flag_depth = 2
        self.lure_stage_depth = 3
        self.better_lure_time_buffer = 1

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
            if lane_y is None:
                continue
            carrier_infos.append((escape_dist if escape_dist is not None else 10**6, o, lane_y))
        carrier_infos.sort(key=lambda item: item[0])

        if not carrier_infos:
            return actions, reserved

        if len(runners) == 1:
            runner = runners[0]
            _dist, carrier, lane_y = carrier_infos[0]
            carrier_pos = self._pos(carrier)
            block_cell = (self.our_boundary_x, lane_y)
            path_block = self._route(self._pos(runner), block_cell, restrict_safe=True)
            path_chase = self._route(self._pos(runner), carrier_pos, restrict_safe=True)
            best_path = path_chase
            if path_block and (not best_path or len(path_block) <= len(best_path)):
                best_path = path_block
            move = self._next_move(self._pos(runner), best_path)
            if move:
                actions[runner["name"]] = move
                reserved.add(runner["name"])
            return actions, reserved

        desired_backstop = 1 if self.guard_post is not None and len(runners) >= 3 else 0
        if len(carrier_infos) > 1 and len(runners) < 5:
            desired_backstop = 0
        pairs_max = min(len(carrier_infos), max(0, (len(runners) - desired_backstop) // 2))
        if pairs_max == 0:
            pairs_max = min(len(carrier_infos), len(runners) // 2)

        available = list(runners)
        pairs_done = 0
        for _escape_dist, carrier, lane_y in carrier_infos:
            if pairs_done >= pairs_max or len(available) < 2:
                break
            carrier_pos = self._pos(carrier)
            block_cell = (self.our_boundary_x, lane_y)
            pair = self._best_block_and_chase_pair(available, block_cell, carrier_pos)
            if pair is None:
                break
            blocker, chaser, blocker_path, chase_path = pair

            blocker_move = self._next_move(self._pos(blocker), blocker_path)
            if blocker_move:
                actions[blocker["name"]] = blocker_move
            reserved.add(blocker["name"])

            chaser_move = self._next_move(self._pos(chaser), chase_path)
            if chaser_move:
                actions[chaser["name"]] = chaser_move
            reserved.add(chaser["name"])

            available = [p for p in available if p["name"] not in {blocker["name"], chaser["name"]}]
            pairs_done += 1

        if self.guard_post is not None and available:
            backstop, path = self._closest_runner_to_cell(available, self.guard_post)
            if backstop is not None:
                move = self._next_move(self._pos(backstop), path)
                if move:
                    actions[backstop["name"]] = move
                reserved.add(backstop["name"])

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
        if self.guard_post is not None and remaining:
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
        elif intruders and runners and self._lure_is_reasonable(req, my_flags):
            allow_lure = True
            if BETTER_LURE and not self._better_lure_is_winning(runners, intruders, my_flags):
                allow_lure = False
            if allow_lure:
                lure_actions, lure_reserved = self._plan_lure_stage(runners, intruders, my_flags)
                actions.update(lure_actions)
                reserved |= lure_reserved
                runners = [p for p in runners if p["name"] not in reserved]

        should_rescue = bool(my_prisoners) and not intruder_carriers
        if should_rescue and prisons and runners:
            if any(self._pos(p) in prisons for p in my_free):
                should_rescue = False

        if should_rescue and prisons and runners:
            best_rescuer = None
            best_rescue_dist = None
            for p in runners:
                p_pos = self._pos(p)
                path = self._route_any(p_pos, prisons, restrict_safe=True)
                if not path:
                    continue
                dist = len(path) - 1
                if best_rescue_dist is None or dist < best_rescue_dist:
                    best_rescue_dist = dist
                    best_rescuer = p
            if best_rescuer is not None:
                path = self._route_any(self._pos(best_rescuer), prisons, restrict_safe=True)
                move = self._next_move(self._pos(best_rescuer), path)
                if move:
                    actions[best_rescuer["name"]] = move
                reserved.add(best_rescuer["name"])
                runners = [p for p in runners if p["name"] != best_rescuer["name"]]

        if self.guard_post is not None and runners:
            safe_runners = [p for p in runners if self._is_safe(self._pos(p))]
            if safe_runners:
                best_guard, path = self._closest_runner_to_cell(safe_runners, self.guard_post)
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
