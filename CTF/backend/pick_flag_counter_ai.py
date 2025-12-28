import asyncio
import itertools

from lib.game_engine import GameMap, run_game_server


class CounterCTFAI:
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
        self.defender_name = None
        self.flag_assignment = {}
        self.tick = 0

        self.avoid_radius = 1
        self.lane_risk_weight = 8.0
        self.lane_separation_weight = 3.0

    def start_game(self, req):
        self.world.init(req)
        self.tick = 0
        self.flag_assignment = {}
        self.defender_name = None
        self._init_geometry()
        side = "Left" if self.my_side_is_left else "Right"
        print(f"Counter AI started. Side: {side}; guard_post={self.guard_post}")

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
        return (int(round(entity["posX"])), int(round(entity["posY"])))

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _sorted_players(players):
        return sorted(players, key=lambda p: p.get("name", ""))

    def _choose_guard_post(self):
        if not self.boundary_ys:
            return next(iter(self.safe_cells), (0, 0))
        center_y = self.world.height // 2
        best_y = min(self.boundary_ys, key=lambda y: abs(y - center_y))
        return (self.our_boundary_x, best_y)

    def _route(self, start, goal, *, extra_obstacles=None, restrict_safe=False):
        extras = set(extra_obstacles or [])
        if restrict_safe:
            extras |= self.enemy_cells
        return self.world.route_to(start, goal, extra_obstacles=extras)

    def _route_any(self, start, goals, *, extra_obstacles=None, restrict_safe=False):
        best = []
        for goal in goals or []:
            path = self._route(start, goal, extra_obstacles=extra_obstacles, restrict_safe=restrict_safe)
            if path and (not best or len(path) < len(best)):
                best = path
        return best

    def _next_move(self, start, path):
        if not path or len(path) < 2:
            return None
        return GameMap.get_direction(start, path[1])

    def _expanded_enemy_obstacles(self, opponents, radius):
        if radius <= 0:
            return {self._pos(o) for o in opponents}
        obstacles = set()
        for o in opponents:
            ex, ey = self._pos(o)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) > radius:
                        continue
                    nx, ny = ex + dx, ey + dy
                    if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                        obstacles.add((nx, ny))
        return obstacles

    def _lane_risk(self, lane_y, opponent_positions):
        risk = 0.0
        for ox, oy in opponent_positions:
            dist = abs(ox - self.our_boundary_x) + abs(oy - lane_y)
            risk += 1.0 / ((dist + 1.0) ** 2)
        return risk

    def _choose_defender(self, my_free):
        if not my_free:
            return None
        if self.defender_name:
            for p in my_free:
                if p["name"] == self.defender_name:
                    return p
        guard = self.guard_post or (self.our_boundary_x, self.world.height // 2)
        defender = min(
            self._sorted_players(my_free),
            key=lambda p: (self._manhattan(self._pos(p), guard), p.get("name", "")),
        )
        self.defender_name = defender["name"]
        return defender

    def _assign_return_lanes(self, carriers, opponents_on_our_side):
        lanes = sorted(self.boundary_ys) if self.boundary_ys else [self.world.height // 2]
        opponent_positions = [self._pos(o) for o in opponents_on_our_side]

        carriers_enemy_side = [p for p in carriers if not self._is_safe(self._pos(p))]
        if not carriers_enemy_side:
            return {}

        assignments = {}
        taken = []
        carriers_enemy_side = sorted(
            carriers_enemy_side,
            key=lambda p: (self._manhattan(self._pos(p), (self.our_boundary_x, self.world.height // 2)), p["name"]),
        )

        for p in carriers_enemy_side:
            start = self._pos(p)
            best_lane = None
            best_score = None
            for y in lanes:
                dist = abs(start[1] - y)
                risk = self._lane_risk(y, opponent_positions)
                separation = sum(1.0 / (abs(y - other) + 1.0) for other in taken)
                score = dist + self.lane_risk_weight * risk + self.lane_separation_weight * separation
                if best_score is None or score < best_score:
                    best_score = score
                    best_lane = y
            if best_lane is None:
                best_lane = lanes[0]
            assignments[p["name"]] = best_lane
            taken.append(best_lane)
        return assignments

    def _assign_attack_targets(self, attackers, flag_positions, opponents_on_enemy_side):
        assignments = {}
        available = set(flag_positions)

        for name, pos in list(self.flag_assignment.items()):
            if pos in available and any(p["name"] == name for p in attackers):
                assignments[name] = pos
                available.remove(pos)

        attackers = [p for p in attackers if p["name"] not in assignments]
        if not attackers or not available:
            self.flag_assignment = assignments
            return assignments

        flags_sorted = sorted(available, key=lambda f: (f[1], f[0]))
        picks = []
        if len(flags_sorted) >= 2 and len(attackers) >= 2:
            picks = [flags_sorted[0], flags_sorted[-1]]
        else:
            picks = [flags_sorted[0]]

        avoid = self._expanded_enemy_obstacles(opponents_on_enemy_side, self.avoid_radius)
        if len(attackers) == 1 or len(picks) == 1:
            attacker = attackers[0]
            best_flag = None
            best_dist = None
            for fpos in picks:
                path = self._route(self._pos(attacker), fpos, extra_obstacles=avoid, restrict_safe=False)
                dist = (len(path) - 1) if path else 10**6
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_flag = fpos
            if best_flag is not None:
                assignments[attacker["name"]] = best_flag
        else:
            best_total = None
            best_pairing = None
            for perm in itertools.permutations(picks, 2):
                total = 0
                for attacker, fpos in zip(attackers[:2], perm):
                    path = self._route(self._pos(attacker), fpos, extra_obstacles=avoid, restrict_safe=False)
                    total += (len(path) - 1) if path else 10**6
                if best_total is None or total < best_total:
                    best_total = total
                    best_pairing = list(zip(attackers[:2], perm))
            if best_pairing:
                for attacker, fpos in best_pairing:
                    assignments[attacker["name"]] = fpos

        self.flag_assignment = assignments
        return assignments

    def _plan_defender(self, defender, carriers_on_our_side, opponents_on_our_side, opponents_all):
        start = self._pos(defender)
        targets = []
        if carriers_on_our_side:
            targets = [self._pos(o) for o in carriers_on_our_side]
        elif opponents_on_our_side:
            targets = [self._pos(o) for o in opponents_on_our_side]
        else:
            if opponents_all:
                avg_y = sum(o["posY"] for o in opponents_all) / float(len(opponents_all))
                target_y = min(self.boundary_ys or [self.world.height // 2], key=lambda y: abs(y - avg_y))
                targets = [(self.our_boundary_x, target_y)]
            elif self.guard_post:
                targets = [self.guard_post]

        if not targets:
            return None

        avoid = self._expanded_enemy_obstacles(opponents_on_our_side, self.avoid_radius)
        path = self._route_any(start, targets, extra_obstacles=avoid, restrict_safe=True)
        if not path and self.guard_post:
            path = self._route(start, self.guard_post, restrict_safe=True)
        return self._next_move(start, path)

    def _plan_rescue(self, rescuer, prisons, opponents_on_our_side):
        if not prisons:
            return None
        start = self._pos(rescuer)
        avoid = self._expanded_enemy_obstacles(opponents_on_our_side, self.avoid_radius)
        path = self._route_any(start, prisons, extra_obstacles=avoid, restrict_safe=True)
        return self._next_move(start, path)

    def _plan_carrier(self, player, lane_y, opponents_on_enemy_side, opponents_on_our_side, targets):
        start = self._pos(player)
        if self._is_safe(start):
            avoid = self._expanded_enemy_obstacles(opponents_on_our_side, self.avoid_radius)
            path = self._route_any(start, targets, extra_obstacles=avoid, restrict_safe=True)
            return self._next_move(start, path)

        if lane_y is None:
            lane_y = self.world.height // 2
        entry = (self.our_boundary_x, lane_y)
        avoid_enemy = self._expanded_enemy_obstacles(opponents_on_enemy_side, self.avoid_radius)

        for avoid in (avoid_enemy, None):
            path = self._route(start, entry, extra_obstacles=avoid, restrict_safe=False)
            move = self._next_move(start, path)
            if move:
                return move

        path = self._route_any(start, targets, extra_obstacles=avoid_enemy, restrict_safe=False)
        return self._next_move(start, path)

    def _plan_attacker(self, player, target, opponents_on_enemy_side):
        if target is None:
            return None
        start = self._pos(player)
        avoid = self._expanded_enemy_obstacles(opponents_on_enemy_side, self.avoid_radius)
        for obstacles in (avoid, None):
            path = self._route(start, target, extra_obstacles=obstacles, restrict_safe=False)
            move = self._next_move(start, path)
            if move:
                return move
        return None

    def plan_next_actions(self, req):
        if not self.world.update(req):
            return {}

        self.tick += 1

        my_all = self._sorted_players(self.world.list_players(mine=True, inPrison=None, hasFlag=None) or [])
        my_free = [p for p in my_all if not p.get("inPrison")]
        if not my_free:
            return {}

        opponents_free = self._sorted_players(
            self.world.list_players(mine=False, inPrison=False, hasFlag=None) or []
        )
        my_prisoners = self.world.list_players(mine=True, inPrison=True, hasFlag=None) or []
        enemy_flags = self.world.list_flags(mine=False, canPickup=True) or []
        targets = list(self.world.list_targets(mine=True) or [])
        prisons = list(self.world.list_prisons(mine=True) or [])

        opponents_on_our_side = [o for o in opponents_free if self._is_safe(self._pos(o))]
        opponents_on_enemy_side = [o for o in opponents_free if not self._is_safe(self._pos(o))]
        carriers_on_our_side = [o for o in opponents_on_our_side if o.get("hasFlag")]

        actions = {}
        reserved = set()

        defender = self._choose_defender(my_free)
        attackers = [p for p in my_free if not defender or p["name"] != defender["name"]]

        rescuer = None
        if my_prisoners and prisons and not carriers_on_our_side:
            candidates = [p for p in attackers if not p.get("hasFlag")]
            if not candidates and defender and not defender.get("hasFlag"):
                candidates = [defender]
            if candidates:
                rescuer = min(
                    candidates,
                    key=lambda p: self._manhattan(self._pos(p), prisons[0]),
                )

        if rescuer:
            move = self._plan_rescue(rescuer, prisons, opponents_on_our_side)
            if move:
                actions[rescuer["name"]] = move
            reserved.add(rescuer["name"])

        if defender and defender["name"] not in reserved:
            move = self._plan_defender(defender, carriers_on_our_side, opponents_on_our_side, opponents_free)
            if move:
                actions[defender["name"]] = move
            reserved.add(defender["name"])

        carriers = [p for p in my_free if p.get("hasFlag") and p["name"] not in reserved]
        lane_assignment = self._assign_return_lanes(carriers, opponents_on_our_side)
        for player in carriers:
            lane_y = lane_assignment.get(player["name"])
            move = self._plan_carrier(player, lane_y, opponents_on_enemy_side, opponents_on_our_side, targets)
            if move:
                actions[player["name"]] = move
            reserved.add(player["name"])

        attackers_no_flag = [
            p for p in my_free
            if (not p.get("hasFlag")) and p["name"] not in reserved
        ]
        if attackers_no_flag:
            flag_positions = [
                (int(f["posX"]), int(f["posY"]))
                for f in enemy_flags
                if f.get("canPickup")
            ]
            assignments = self._assign_attack_targets(attackers_no_flag, flag_positions, opponents_on_enemy_side)
            for player in attackers_no_flag:
                target = assignments.get(player["name"])
                move = self._plan_attacker(player, target, opponents_on_enemy_side)
                if move:
                    actions[player["name"]] = move

        return actions


AI = CounterCTFAI()


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
