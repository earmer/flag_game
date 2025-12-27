import asyncio
import itertools
import random

from lib.game_engine import GameMap, run_game_server


class EliteCTFAI:
    def __init__(self, show_gap_in_msec=1000.0):
        self.world = GameMap(show_gap_in_msec=show_gap_in_msec)
        self.my_side_is_left = True
        self.left_max_x = 0
        self.right_min_x = 0
        self.our_boundary_x = 0
        self.enemy_boundary_x = 0
        self.safe_cells = set()
        self.enemy_cells = set()
        self.boundary_ys = []
        self.guard_post = None
        self.flag_assignment = {}
        self.tick = 0

        self.switch_penalty = 3.0
        self.flag_risk_weight = 5.0
        self.avoid_radius_runner = 1
        self.avoid_radius_carrier = 2

    def start_game(self, req):
        self.world.init(req)
        self.tick = 0
        self.flag_assignment = {}
        self._init_geometry()
        print(
            f"Elite AI started. Side: {'Left' if self.my_side_is_left else 'Right'}; "
            f"guard_post={self.guard_post}"
        )

    def game_over(self, _req):
        print("Game Over!")

    def _init_geometry(self):
        width = self.world.width
        height = self.world.height

        targets = list(self.world.list_targets(mine=True) or [])
        if targets:
            self.my_side_is_left = self.world.is_on_left(next(iter(targets)))
        else:
            self.my_side_is_left = self.world.my_team_name == "L"

        self.left_max_x = (width - 1) // 2
        self.right_min_x = self.left_max_x + 1

        if self.my_side_is_left:
            self.our_boundary_x = self.left_max_x
            self.enemy_boundary_x = self.right_min_x
        else:
            self.our_boundary_x = self.right_min_x
            self.enemy_boundary_x = self.left_max_x

        self.safe_cells = set()
        self.enemy_cells = set()
        for x in range(width):
            for y in range(height):
                pos = (x, y)
                if self._is_safe(pos):
                    self.safe_cells.add(pos)
                else:
                    self.enemy_cells.add(pos)

        self.boundary_ys = [
            y
            for y in range(height)
            if (self.our_boundary_x, y) not in self.world.walls
            and (self.enemy_boundary_x, y) not in self.world.walls
        ]
        self.guard_post = self._choose_guard_post()

    def _is_safe(self, pos):
        return self.world.is_on_left(pos) == self.my_side_is_left

    @staticmethod
    def _pos(entity):
        return (int(entity["posX"]), int(entity["posY"]))

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _risk_score(self, pos, enemy_positions):
        score = 0.0
        for epos in enemy_positions:
            dist = self._manhattan(pos, epos)
            score += 1.0 / ((dist + 1.0) ** 2)
        return score

    def _expanded_enemy_obstacles(self, opponents, radius):
        if radius <= 0:
            return {self._pos(o) for o in opponents if not self._is_safe(self._pos(o))}

        obstacles = set()
        for o in opponents:
            epos = self._pos(o)
            if self._is_safe(epos):
                continue
            ex, ey = epos
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) > radius:
                        continue
                    nx, ny = ex + dx, ey + dy
                    if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                        obstacles.add((nx, ny))
        return obstacles

    def _route(self, start, goal, *, extra_obstacles=None, restrict_safe=False):
        extras = set(extra_obstacles or [])
        if restrict_safe:
            extras |= self.enemy_cells
        return self.world.route_to(start, goal, extra_obstacles=extras)

    def _route_any(self, start, goals, *, extra_obstacles=None, restrict_safe=False):
        best = []
        for goal in goals:
            path = self._route(start, goal, extra_obstacles=extra_obstacles, restrict_safe=restrict_safe)
            if path and (not best or len(path) < len(best)):
                best = path
        return best

    def _next_move(self, start, path):
        if not path or len(path) < 2:
            return None
        return GameMap.get_direction(start, path[1])

    def _choose_guard_post(self):
        if not self.boundary_ys:
            return None

        targets = set(self.world.list_targets(mine=True) or [])
        prisons = set(self.world.list_prisons(mine=True) or [])
        if not targets:
            return (self.our_boundary_x, self.world.height // 2)

        best_score = None
        best_cell = None

        for y in self.boundary_ys:
            cell = (self.our_boundary_x, y)
            path_to_target = self._route_any(cell, targets, restrict_safe=True)
            if not path_to_target:
                continue
            dist_target = len(path_to_target) - 1
            dist_prison = 0
            if prisons:
                path_to_prison = self._route_any(cell, prisons, restrict_safe=True)
                dist_prison = (len(path_to_prison) - 1) if path_to_prison else 50

            score = dist_target + 0.6 * dist_prison + 0.05 * abs(y - (self.world.height / 2.0))
            if best_score is None or score < best_score:
                best_score = score
                best_cell = cell

        return best_cell or (self.our_boundary_x, self.world.height // 2)

    def _predict_escape_lane(self, carrier_pos):
        if not self.boundary_ys:
            return None, None

        best_y = None
        best_dist = None
        for y in self.boundary_ys:
            dest = (self.enemy_boundary_x, y)
            path = self._route(carrier_pos, dest, extra_obstacles=None, restrict_safe=False)
            if not path:
                continue
            dist = len(path) - 1
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_y = y
        return best_y, best_dist

    def _move_carrier(self, player, opponents, targets):
        start = self._pos(player)

        if self._is_safe(start):
            path = self._route_any(start, targets, restrict_safe=True)
            return self._next_move(start, path)

        enemy_positions = [self._pos(o) for o in opponents if not self._is_safe(self._pos(o))]
        avoid2 = self._expanded_enemy_obstacles(opponents, self.avoid_radius_carrier)
        avoid1 = self._expanded_enemy_obstacles(opponents, 1)
        avoid0 = self._expanded_enemy_obstacles(opponents, 0)

        best_entry = None
        best_score = None
        for y in self.boundary_ys:
            entry = (self.our_boundary_x, y)
            path = self._route(start, entry, extra_obstacles=avoid1, restrict_safe=False)
            if not path:
                continue
            dist = len(path) - 1
            score = dist + 6.0 * self._risk_score(entry, enemy_positions)
            if best_score is None or score < best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None:
            for avoid in (avoid2, avoid1, avoid0, None):
                path = self._route(start, best_entry, extra_obstacles=avoid, restrict_safe=False)
                move = self._next_move(start, path)
                if move:
                    return move

        for avoid in (avoid2, avoid1, avoid0, None):
            path = self._route_any(start, targets, extra_obstacles=avoid, restrict_safe=False)
            move = self._next_move(start, path)
            if move:
                return move

        return None

    def _assign_flags(self, attackers, flag_positions, opponents_enemy_side):
        if not attackers or not flag_positions:
            return {}

        attackers = list(attackers)
        flags = list(flag_positions)

        if len(attackers) > len(flags):
            scored = []
            for p in attackers:
                start = self._pos(p)
                best = None
                for fpos in flags:
                    path = self._route(start, fpos, restrict_safe=False)
                    if not path:
                        continue
                    dist = len(path) - 1
                    if best is None or dist < best:
                        best = dist
                scored.append((best if best is not None else 10**9, p))
            attackers = [p for _dist, p in sorted(scored, key=lambda item: item[0])[: len(flags)]]

        prev = dict(self.flag_assignment)
        current_flags = set(flags)
        for name, pos in list(prev.items()):
            if pos not in current_flags:
                prev.pop(name, None)

        cost = {}
        for p in attackers:
            start = self._pos(p)
            for fpos in flags:
                path = self._route(start, fpos, restrict_safe=False)
                dist = (len(path) - 1) if path else 10**6
                risk = self._risk_score(fpos, opponents_enemy_side)
                switch = 0.0
                if p["name"] in prev and prev[p["name"]] != fpos:
                    switch = self.switch_penalty
                cost[(p["name"], fpos)] = dist + self.flag_risk_weight * risk + switch

        best_total = None
        best_assignment = {}
        for perm in itertools.permutations(flags, len(attackers)):
            total = 0.0
            for p, fpos in zip(attackers, perm):
                total += cost[(p["name"], fpos)]
                if best_total is not None and total >= best_total:
                    break
            else:
                if best_total is None or total < best_total:
                    best_total = total
                    best_assignment = {p["name"]: fpos for p, fpos in zip(attackers, perm)}

        return best_assignment

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
        if intruder_carriers and runners:
            best_block = None
            best_block_dist = None
            best_lane_y = None

            for o in intruder_carriers:
                o_pos = self._pos(o)
                lane_y, _escape_dist = self._predict_escape_lane(o_pos)
                if lane_y is None:
                    continue
                block_cell = (self.our_boundary_x, lane_y)
                for p in runners:
                    p_pos = self._pos(p)
                    path = self._route(p_pos, block_cell, restrict_safe=True)
                    if not path:
                        continue
                    dist = len(path) - 1
                    if best_block_dist is None or dist < best_block_dist:
                        best_block_dist = dist
                        best_block = (p, block_cell)
                        best_lane_y = lane_y

            if best_block is not None:
                blocker, block_cell = best_block
                path = self._route(self._pos(blocker), block_cell, restrict_safe=True)
                move = self._next_move(self._pos(blocker), path)
                if move:
                    actions[blocker["name"]] = move
                reserved.add(blocker["name"])
                runners = [p for p in runners if p["name"] != blocker["name"]]

                if runners:
                    best_chase = None
                    best_chase_dist = None
                    best_target = None
                    for o in intruder_carriers:
                        o_pos = self._pos(o)
                        for p in runners:
                            p_pos = self._pos(p)
                            path = self._route(p_pos, o_pos, restrict_safe=True)
                            if not path:
                                continue
                            dist = len(path) - 1
                            if best_chase_dist is None or dist < best_chase_dist:
                                best_chase_dist = dist
                                best_chase = p
                                best_target = o_pos

                    if best_chase is not None and best_target is not None:
                        path = self._route(self._pos(best_chase), best_target, restrict_safe=True)
                        move = self._next_move(self._pos(best_chase), path)
                        if move:
                            actions[best_chase["name"]] = move
                        reserved.add(best_chase["name"])
                        runners = [p for p in runners if p["name"] != best_chase["name"]]

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
                best_guard = None
                best_guard_dist = None
                for p in safe_runners:
                    p_pos = self._pos(p)
                    path = self._route(p_pos, self.guard_post, restrict_safe=True)
                    if not path:
                        continue
                    dist = len(path) - 1
                    if best_guard_dist is None or dist < best_guard_dist:
                        best_guard_dist = dist
                        best_guard = p
                if best_guard is not None:
                    path = self._route(self._pos(best_guard), self.guard_post, restrict_safe=True)
                    move = self._next_move(self._pos(best_guard), path)
                    if move:
                        actions[best_guard["name"]] = move
                    reserved.add(best_guard["name"])
                    runners = [p for p in runners if p["name"] != best_guard["name"]]

        attackers = [p for p in runners if p["name"] not in reserved]
        flag_positions = [(int(f["posX"]), int(f["posY"])) for f in enemy_flags]
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


AI = EliteCTFAI()


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

