import asyncio
import itertools
import random

from lib.game_engine import GameMap, run_game_server


class AggressiveCTFAI:
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
        self.cover_cells = set()
        self.lane_ys = []
        self.lane_assignment = {}
        self.flag_assignment = {}
        self.phase = "attack"
        self.attack_stage = "staging"
        self.tick = 0
        self.num_flags = 0
        self.stall_score_threshold = 0

        self.flag_switch_penalty = 3.0
        self.lane_switch_penalty = 1.5
        self.lane_weight = 1.4
        self.flag_risk_weight = 4.0
        self.avoid_radius_runner = 1
        self.avoid_radius_carrier = 2
        self.cover_trigger_distance = 3
        self.cover_detour_allowance = 4
        self.boundary_alert_distance = 4

    def start_game(self, req):
        self.world.init(req)
        self.tick = 0
        self.phase = "attack"
        self.attack_stage = "staging"
        self.flag_assignment = {}
        self.lane_assignment = {}
        self.lane_ys = []
        self.num_flags = req.get("numFlags", 0)
        self.stall_score_threshold = max(0, self.num_flags - 2)
        self._init_geometry()
        side = "Left" if self.my_side_is_left else "Right"
        print(f"Aggressive AI started. Side: {side}")

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
        self._build_cover_cells()

    def _build_cover_cells(self):
        self.cover_cells = set()
        width = self.world.width
        height = self.world.height
        for x in range(width):
            for y in range(height):
                pos = (x, y)
                if pos in self.world.walls:
                    continue
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and (nx, ny) in self.world.walls:
                        self.cover_cells.add(pos)
                        break

    def _is_safe(self, pos):
        return self.world.is_on_left(pos) == self.my_side_is_left

    @staticmethod
    def _pos(entity):
        return (int(round(entity["posX"])), int(round(entity["posY"])))

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _risk_score(self, pos, enemy_positions):
        risk = 0.0
        for epos in enemy_positions:
            dist = self._manhattan(pos, epos)
            risk += 1.0 / ((dist + 1.0) ** 2)
        return risk

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

    def _closest_boundary_y(self, target_y):
        if not self.boundary_ys:
            return max(0, min(self.world.height - 1, int(round(target_y))))
        return min(self.boundary_ys, key=lambda y: abs(y - target_y))

    def _choose_lanes(self, count):
        if count <= 0:
            return []
        height = self.world.height
        candidates = list(self.boundary_ys) if self.boundary_ys else list(range(height))
        if len(candidates) <= count:
            return candidates

        targets = [
            int(round((idx + 1) * (height - 1) / (count + 1)))
            for idx in range(count)
        ]

        best = None
        best_score = None
        for combo in itertools.combinations(candidates, count):
            ys = sorted(combo)
            if count == 1:
                min_dist = height
            else:
                min_dist = min(abs(ys[i] - ys[j]) for i in range(count) for j in range(i + 1, count))
            spread = ys[-1] - ys[0]
            penalty = sum(abs(ys[i] - targets[i]) for i in range(count))
            score = (min_dist, spread, -penalty)
            if best_score is None or score > best_score:
                best_score = score
                best = ys
        return best or candidates[:count]

    def _assign_lanes_to_players(self, players):
        players = list(players)
        if not players:
            self.lane_assignment = {}
            return {}

        lane_count = min(len(players), max(1, len(self.boundary_ys)))
        lane_ys = self._choose_lanes(min(3, lane_count))
        if not lane_ys:
            lane_ys = [self.world.height // 2]
        self.lane_ys = lane_ys

        if len(lane_ys) < len(players):
            repeated = (lane_ys * ((len(players) + len(lane_ys) - 1) // len(lane_ys)))[: len(players)]
            lane_ys = repeated

        prev = dict(self.lane_assignment)
        best = None
        best_cost = None
        if len(set(lane_ys)) >= len(players):
            for perm in itertools.permutations(lane_ys, len(players)):
                cost = 0.0
                for p, lane_y in zip(players, perm):
                    p_y = int(round(p["posY"]))
                    cost += abs(p_y - lane_y)
                    if prev.get(p["name"]) is not None and prev.get(p["name"]) != lane_y:
                        cost += self.lane_switch_penalty
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best = {p["name"]: lane_y for p, lane_y in zip(players, perm)}
        else:
            players_sorted = sorted(players, key=lambda p: p["posY"])
            lanes_sorted = sorted(lane_ys)
            best = {}
            for idx, p in enumerate(players_sorted):
                best[p["name"]] = lanes_sorted[idx % len(lanes_sorted)]

        self.lane_assignment = best or {}
        return self.lane_assignment

    def _assign_flags_by_lane(self, attackers, flag_positions, enemy_positions):
        if not attackers or not flag_positions:
            self.flag_assignment = {}
            return {}

        attackers = list(attackers)
        flags = list(flag_positions)

        prev = dict(self.flag_assignment)
        current_flags = set(flags)
        for name, pos in list(prev.items()):
            if pos not in current_flags:
                prev.pop(name, None)

        cost = {}
        for p in attackers:
            start = self._pos(p)
            lane_y = self.lane_assignment.get(p["name"])
            for fpos in flags:
                path = self._route(start, fpos, restrict_safe=False)
                dist = (len(path) - 1) if path else 10**6
                lane_cost = abs(fpos[1] - lane_y) if lane_y is not None else 0.0
                risk = self._risk_score(fpos, enemy_positions)
                switch = 0.0
                if p["name"] in prev and prev[p["name"]] != fpos:
                    switch = self.flag_switch_penalty
                cost[(p["name"], fpos)] = dist + self.lane_weight * lane_cost + self.flag_risk_weight * risk + switch

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

    def _lane_entry_for_player(self, player):
        start = self._pos(player)
        lane_y = self.lane_assignment.get(player["name"])
        if lane_y is None:
            lane_y = self._closest_boundary_y(start[1])
            self.lane_assignment[player["name"]] = lane_y
        return (self.our_boundary_x, lane_y)

    def _assign_defense_targets(self, defenders, targets):
        defenders = list(defenders)
        targets = list(targets)
        assignments = {}
        if not defenders or not targets:
            return assignments

        defenders = sorted(defenders, key=lambda p: p.get("name", ""))
        for p in defenders:
            if not targets:
                break
            start = self._pos(p)
            best = min(targets, key=lambda t: self._manhattan(start, t))
            assignments[p["name"]] = best
            targets.remove(best)
        return assignments

    def _should_rescue(self, prisoners, free_players):
        if not prisoners or not free_players:
            return False
        if any(p.get("hasFlag") for p in free_players):
            return False
        if any(not self._is_safe(self._pos(p)) for p in free_players):
            return False
        return True

    def _should_stall(self, my_score, opponent_score, enemy_flags, free_players, opponents_free):
        if not enemy_flags or not free_players:
            return False
        if my_score < self.stall_score_threshold:
            return False
        if opponent_score >= self.stall_score_threshold:
            return False
        if len(enemy_flags) > len(free_players):
            return False
        if any(o.get("hasFlag") for o in opponents_free):
            return False
        return True

    def _should_use_cover(self, start, opponents):
        for o in opponents:
            if self._manhattan(start, self._pos(o)) <= self.cover_trigger_distance:
                return True
        return False

    def _route_with_cover(self, start, goal, *, opponents, extra_obstacles=None, restrict_safe=False, lane_y=None):
        direct = self._route(start, goal, extra_obstacles=extra_obstacles, restrict_safe=restrict_safe)
        if not self.cover_cells or not opponents or not self._should_use_cover(start, opponents):
            return direct

        candidates = [
            c for c in self.cover_cells
            if lane_y is None or abs(c[1] - lane_y) <= 2
        ]
        if restrict_safe:
            candidates = [c for c in candidates if self._is_safe(c)]

        candidates.sort(key=lambda c: self._manhattan(start, c))
        candidates = candidates[:10]
        best = None
        best_len = None

        for cover in candidates:
            path1 = self._route(start, cover, extra_obstacles=extra_obstacles, restrict_safe=restrict_safe)
            if not path1:
                continue
            path2 = self._route(cover, goal, extra_obstacles=extra_obstacles, restrict_safe=restrict_safe)
            if not path2:
                continue
            combined = path1 + path2[1:]
            if direct and len(combined) > len(direct) + self.cover_detour_allowance:
                continue
            if best_len is None or len(combined) < best_len:
                best_len = len(combined)
                best = combined

        return best or direct

    def _fallback_step(self, start, opponents, extra_obstacles=None, restrict_safe=False):
        options = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = start[0] + dx, start[1] + dy
            if not (0 <= nx < self.world.width and 0 <= ny < self.world.height):
                continue
            pos = (nx, ny)
            if pos in self.world.walls:
                continue
            if restrict_safe and not self._is_safe(pos):
                continue
            if extra_obstacles and pos in extra_obstacles:
                continue
            options.append(pos)
        if not options:
            return None

        if opponents:
            def score(pos):
                return min(self._manhattan(pos, self._pos(o)) for o in opponents)
            options.sort(key=score, reverse=True)
            best = options[0]
        else:
            best = random.choice(options)
        return GameMap.get_direction(start, best)

    def _move_towards(self, start, goal, *, opponents, avoid_radius, restrict_safe=False, lane_y=None):
        avoid = self._expanded_enemy_obstacles(opponents, avoid_radius)
        path = self._route_with_cover(
            start,
            goal,
            opponents=opponents,
            extra_obstacles=avoid,
            restrict_safe=restrict_safe,
            lane_y=lane_y,
        )
        move = self._next_move(start, path)
        if move:
            return move
        return self._fallback_step(start, opponents, extra_obstacles=avoid, restrict_safe=restrict_safe)

    def _closest_target(self, targets, lane_y, start):
        if not targets:
            return None
        if lane_y is None:
            return min(targets, key=lambda t: self._manhattan(start, t))
        return min(targets, key=lambda t: (abs(t[1] - lane_y), self._manhattan(start, t)))

    def _staging_actions(self, free_players, opponents_free, prisoners):
        self._assign_lanes_to_players(free_players)
        ready = []
        waiting = []
        actions = {}

        for p in free_players:
            start = self._pos(p)
            entry = self._lane_entry_for_player(p)
            if self._is_safe(start) and start == entry:
                ready.append(p)
            else:
                waiting.append(p)

        intruders = [o for o in opponents_free if self._is_safe(self._pos(o))]
        intruder_carriers = [o for o in intruders if o.get("hasFlag")]
        boundary_threats = [
            o for o in opponents_free
            if not self._is_safe(self._pos(o))
            and abs(self._pos(o)[0] - self.our_boundary_x) <= self.boundary_alert_distance
        ]

        all_ready = not prisoners and not waiting and not intruders
        if all_ready:
            return actions, True

        for p in waiting:
            start = self._pos(p)
            entry = self._lane_entry_for_player(p)
            restrict_safe = self._is_safe(start)
            opponents = intruders if restrict_safe else opponents_free
            move = self._move_towards(
                start,
                entry,
                opponents=opponents,
                avoid_radius=1,
                restrict_safe=restrict_safe,
                lane_y=entry[1],
            )
            if move:
                actions[p["name"]] = move

        if intruder_carriers and (ready or waiting):
            defenders = [p for p in (ready + waiting) if self._is_safe(self._pos(p))]
            reserved = set()
            block_targets = [
                (self.our_boundary_x, self._closest_boundary_y(self._pos(o)[1]))
                for o in intruder_carriers
            ]
            block_assignments = self._assign_defense_targets(defenders, block_targets)
            for p in defenders:
                target = block_assignments.get(p["name"])
                if not target:
                    continue
                start = self._pos(p)
                move = self._move_towards(
                    start,
                    target,
                    opponents=intruders,
                    avoid_radius=1,
                    restrict_safe=True,
                    lane_y=target[1],
                )
                if move:
                    actions[p["name"]] = move
                reserved.add(p["name"])

            chasers = [p for p in defenders if p["name"] not in reserved]
            if chasers:
                chase_targets = [self._pos(o) for o in intruder_carriers]
                chase_assignments = self._assign_defense_targets(chasers, chase_targets)
                for p in chasers:
                    target = chase_assignments.get(p["name"])
                    if not target:
                        continue
                    start = self._pos(p)
                    move = self._move_towards(
                        start,
                        target,
                        opponents=intruders,
                        avoid_radius=1,
                        restrict_safe=True,
                        lane_y=target[1],
                    )
                    if move:
                        actions[p["name"]] = move
        elif intruders and ready:
            defenders = [p for p in ready if self._is_safe(self._pos(p))]
            reserved = set()
            block_targets = [
                (self.our_boundary_x, self._closest_boundary_y(self._pos(o)[1]))
                for o in intruders
            ]
            block_assignments = self._assign_defense_targets(defenders, block_targets)
            for p in defenders:
                target = block_assignments.get(p["name"])
                if not target:
                    continue
                start = self._pos(p)
                move = self._move_towards(
                    start,
                    target,
                    opponents=intruders,
                    avoid_radius=1,
                    restrict_safe=True,
                    lane_y=target[1],
                )
                if move:
                    actions[p["name"]] = move
                reserved.add(p["name"])

            chasers = [p for p in defenders if p["name"] not in reserved]
            if chasers:
                chase_targets = [self._pos(o) for o in intruders]
                chase_assignments = self._assign_defense_targets(chasers, chase_targets)
                for p in chasers:
                    target = chase_assignments.get(p["name"])
                    if not target:
                        continue
                    start = self._pos(p)
                    move = self._move_towards(
                        start,
                        target,
                        opponents=intruders,
                        avoid_radius=1,
                        restrict_safe=True,
                        lane_y=target[1],
                    )
                    if move:
                        actions[p["name"]] = move
        elif boundary_threats and ready:
            defenders = [p for p in ready if self._is_safe(self._pos(p))]
            block_positions = [
                (self.our_boundary_x, self._closest_boundary_y(self._pos(o)[1]))
                for o in boundary_threats
            ]
            assignments = self._assign_defense_targets(defenders, block_positions)
            for p in defenders:
                target = assignments.get(p["name"])
                if not target:
                    continue
                start = self._pos(p)
                move = self._move_towards(
                    start,
                    target,
                    opponents=boundary_threats,
                    avoid_radius=1,
                    restrict_safe=True,
                    lane_y=target[1],
                )
                if move:
                    actions[p["name"]] = move

        return actions, False

    def _stall_moves(self, free_players, enemy_flags, opponents):
        self._assign_lanes_to_players(free_players)
        flag_positions = [(int(f["posX"]), int(f["posY"])) for f in enemy_flags]
        enemy_positions = [self._pos(o) for o in opponents if not self._is_safe(self._pos(o))]
        assignment = self._assign_flags_by_lane(free_players, flag_positions, enemy_positions)
        self.flag_assignment = assignment

        actions = {}
        for p in free_players:
            target = assignment.get(p["name"])
            if not target:
                continue
            start = self._pos(p)
            if start == target:
                continue
            move = self._move_towards(
                start,
                target,
                opponents=opponents,
                avoid_radius=self.avoid_radius_runner,
                restrict_safe=False,
                lane_y=target[1],
            )
            if move:
                actions[p["name"]] = move
        return actions

    def _rescue_moves(self, free_players, prisons, opponents):
        actions = {}
        if not prisons:
            return actions
        for p in free_players:
            start = self._pos(p)
            target = min(prisons, key=lambda t: self._manhattan(start, t))
            move = self._move_towards(
                start,
                target,
                opponents=opponents,
                avoid_radius=1,
                restrict_safe=True,
                lane_y=target[1],
            )
            if move:
                actions[p["name"]] = move
        return actions

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

        targets = list(self.world.list_targets(mine=True) or [])
        prisons = list(self.world.list_prisons(mine=True) or [])

        my_score = req.get("myteamScore", 0)
        opponent_score = req.get("opponentScore", 0)

        if self._should_stall(my_score, opponent_score, enemy_flags, my_free, opponents_free):
            self.phase = "stall"
            return self._stall_moves(my_free, enemy_flags, opponents_free)

        if self.phase != "rescue" and self._should_rescue(my_prisoners, my_free):
            self.phase = "rescue"
            self.attack_stage = "staging"
        elif self.phase == "rescue" and not my_prisoners:
            self.phase = "attack"
            self.attack_stage = "staging"

        if self.phase == "rescue":
            return self._rescue_moves(my_free, prisons, opponents_free)

        self.phase = "attack"

        carriers = [p for p in my_free if p.get("hasFlag")]
        all_safe_no_flag = all(
            self._is_safe(self._pos(p)) and not p.get("hasFlag") for p in my_free
        )
        if not carriers and all_safe_no_flag and self.attack_stage == "go":
            self.attack_stage = "staging"
            self.flag_assignment = {}

        if not carriers and self.attack_stage == "staging":
            actions, all_ready = self._staging_actions(my_free, opponents_free, my_prisoners)
            if not all_ready:
                return actions
            self.attack_stage = "go"

        self._assign_lanes_to_players(my_free)

        opponents_safe_side = [o for o in opponents_free if self._is_safe(self._pos(o))]
        opponents_enemy_side = [o for o in opponents_free if not self._is_safe(self._pos(o))]
        enemy_positions = [self._pos(o) for o in opponents_enemy_side]

        actions = {}

        runners = [p for p in my_free if not p.get("hasFlag")]

        for p in carriers:
            start = self._pos(p)
            lane_y = self.lane_assignment.get(p["name"])
            if lane_y is None:
                lane_y = self._closest_boundary_y(start[1])
                self.lane_assignment[p["name"]] = lane_y
            entry = (self.our_boundary_x, lane_y)

            if not self._is_safe(start):
                move = self._move_towards(
                    start,
                    entry,
                    opponents=opponents_enemy_side,
                    avoid_radius=self.avoid_radius_carrier,
                    restrict_safe=False,
                    lane_y=lane_y,
                )
                if move:
                    actions[p["name"]] = move
                continue

            home = self._closest_target(targets, lane_y, start)
            if home is None:
                continue
            move = self._move_towards(
                start,
                home,
                opponents=opponents_safe_side,
                avoid_radius=1,
                restrict_safe=True,
                lane_y=lane_y,
            )
            if move:
                actions[p["name"]] = move

        if runners and enemy_flags:
            flag_positions = [(int(f["posX"]), int(f["posY"])) for f in enemy_flags]
            assignment = self._assign_flags_by_lane(runners, flag_positions, enemy_positions)
            self.flag_assignment = assignment
        else:
            assignment = {}
            self.flag_assignment = assignment

        for p in runners:
            start = self._pos(p)
            lane_y = self.lane_assignment.get(p["name"])
            if lane_y is None:
                lane_y = self._closest_boundary_y(start[1])
                self.lane_assignment[p["name"]] = lane_y
            entry = (self.enemy_boundary_x, lane_y)
            target = assignment.get(p["name"])

            if target:
                if self._is_safe(start):
                    move = self._move_towards(
                        start,
                        entry,
                        opponents=opponents_safe_side,
                        avoid_radius=self.avoid_radius_runner,
                        restrict_safe=False,
                        lane_y=lane_y,
                    )
                    if move:
                        actions[p["name"]] = move
                        continue
                move = self._move_towards(
                    start,
                    target,
                    opponents=opponents_enemy_side,
                    avoid_radius=self.avoid_radius_runner,
                    restrict_safe=False,
                    lane_y=lane_y,
                )
                if move:
                    actions[p["name"]] = move
                continue

            if self._is_safe(start):
                move = self._move_towards(
                    start,
                    entry,
                    opponents=opponents_safe_side,
                    avoid_radius=self.avoid_radius_runner,
                    restrict_safe=False,
                    lane_y=lane_y,
                )
            else:
                move = self._fallback_step(
                    start,
                    opponents_enemy_side,
                    extra_obstacles=self._expanded_enemy_obstacles(opponents_enemy_side, self.avoid_radius_runner),
                    restrict_safe=False,
                )
            if move:
                actions[p["name"]] = move

        return actions


AI = AggressiveCTFAI()


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
