import asyncio

from lib.game_engine import GameMap, run_game_server


class ExtremeCTFAI:
    MOVES = (
        ("up", 0, -1),
        ("down", 0, 1),
        ("left", -1, 0),
        ("right", 1, 0),
    )
    ROLE_ORDER = ("shadow", "sprinter", "flanker")
    ROLE_WEIGHTS = {
        "shadow": {"progress": 1.5, "safety": 8.0, "lane": 2.5},
        "sprinter": {"progress": 8.0, "safety": 1.5, "lane": 0.5},
        "flanker": {"progress": 3.0, "safety": 2.0, "lane": 8.0},
    }
    CARRIER_WEIGHTS = {"progress": 8.0, "safety": 4.0, "lane": 1.0}

    AVOID_RADIUS = {"shadow": 2, "sprinter": 0, "flanker": 1}
    CARRIER_AVOID_RADIUS = 2
    DANGER_DISTANCE = 1
    DANGER_PENALTY = 60.0
    AVOID_PENALTY = 8.0
    INTERCEPT_RADIUS = 4
    SAFE_PUSH_WEIGHT = 4.0
    RETREAT_PENALTY = 8.0
    LANE_RELAX_FACTOR = 0.25

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
        self.center_y = 0
        self.role_by_name = {}
        self.lane_by_role = {}
        self.tick = 0

    def start_game(self, req):
        self.world.init(req)
        self.tick = 0
        self.role_by_name = {}
        self._init_geometry()
        side = "Left" if self.my_side_is_left else "Right"
        print(f"Extreme AI started. Side: {side}; lanes={self.lane_by_role}")

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
        self.center_y = height // 2
        self._choose_lanes()

    def _choose_lanes(self):
        height = self.world.height
        if self.boundary_ys:
            ys = sorted(self.boundary_ys)
            top = ys[0]
            bottom = ys[-1]
            mid = ys[len(ys) // 2]
        else:
            top = 0
            bottom = height - 1
            mid = height // 2
        self.lane_by_role = {"shadow": top, "sprinter": mid, "flanker": bottom}

    def _ensure_roles(self):
        players = self.world.list_players(mine=True, inPrison=None, hasFlag=None) or []
        names = sorted({p["name"] for p in players})
        if not names:
            return
        if set(names) != set(self.role_by_name.keys()):
            self.role_by_name = {}
            for idx, name in enumerate(names):
                self.role_by_name[name] = self.ROLE_ORDER[idx % len(self.ROLE_ORDER)]

    def _is_safe(self, pos):
        return self.world.is_on_left(pos) == self.my_side_is_left

    @staticmethod
    def _pos(entity):
        return (int(round(entity["posX"])), int(round(entity["posY"])))

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _min_enemy_distance(self, pos, enemy_positions):
        if not enemy_positions:
            return self.world.width + self.world.height
        return min(self._manhattan(pos, epos) for epos in enemy_positions)

    def _expanded_positions(self, positions, radius):
        if radius <= 0:
            return set(positions)
        width = self.world.width
        height = self.world.height
        obstacles = set()
        for px, py in positions:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) > radius:
                        continue
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        obstacles.add((nx, ny))
        return obstacles

    def _route_length(self, start, goal, avoid=None):
        if start == goal:
            return 0
        path = self.world.route_to(start, goal, extra_obstacles=avoid)
        if not path and avoid:
            path = self.world.route_to(start, goal, extra_obstacles=None)
        if not path:
            return self._manhattan(start, goal)
        return len(path) - 1

    def _path_direction(self, start, target, avoid=None):
        if not target:
            return None
        path = self.world.route_to(start, target, extra_obstacles=avoid)
        if not path and avoid:
            path = self.world.route_to(start, target, extra_obstacles=None)
        if path and len(path) > 1:
            return GameMap.get_direction(start, path[1])
        return None

    def _closest_by_path(self, start, goals, avoid=None):
        best_goal = None
        best_dist = None
        for goal in goals or []:
            dist = self._route_length(start, goal, avoid)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_goal = goal
        return best_goal

    def _closest_by_manhattan(self, start, goals):
        best_goal = None
        best_dist = None
        for goal in goals or []:
            dist = self._manhattan(start, goal)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_goal = goal
        return best_goal, best_dist

    def _select_flag_for_role(self, start, flags, role, lane_y, enemy_positions, avoid):
        if not flags:
            return None
        best_flag = None
        best_key = None
        for fpos in flags:
            dist = self._route_length(start, fpos, avoid)
            safety = self._min_enemy_distance(fpos, enemy_positions)
            lane_dist = abs(fpos[1] - lane_y) if lane_y is not None else 0
            lane_extreme = abs(fpos[1] - self.center_y)
            if role == "sprinter":
                key = (dist, -safety)
            elif role == "shadow":
                key = (-safety, lane_dist, dist)
            else:
                key = (lane_dist, -lane_extreme, dist)
            if best_key is None or key < best_key:
                best_key = key
                best_flag = fpos
        return best_flag

    def _fallback_target(self, lane_y):
        if lane_y is None:
            return None
        return (self.our_boundary_x, lane_y)

    def _choose_move(self, start, target, role, lane_y, enemy_positions, avoid, is_carrier):
        if not target:
            return None
        avoid = set(avoid or [])
        dist_before = self._route_length(start, target, avoid)
        path_dir = self._path_direction(start, target, avoid)
        start_is_safe = self._is_safe(start)
        target_is_enemy = target is not None and not self._is_safe(target)
        threat_distance = self._min_enemy_distance(start, enemy_positions)
        threatened = (not start_is_safe) and threat_distance <= (self.DANGER_DISTANCE + 1)
        best_score = None
        best_move = None
        best_progress = None

        for direction, dx, dy in self.MOVES:
            nx, ny = start[0] + dx, start[1] + dy
            if nx < 0 or ny < 0 or nx >= self.world.width or ny >= self.world.height:
                continue
            if (nx, ny) in self.world.walls:
                continue
            next_pos = (nx, ny)

            dist_after = self._route_length(next_pos, target, avoid)
            progress = dist_before - dist_after
            safety = self._min_enemy_distance(next_pos, enemy_positions)
            lane_score = -abs(ny - lane_y) if lane_y is not None else 0.0

            weights = self.CARRIER_WEIGHTS if is_carrier else self.ROLE_WEIGHTS.get(role, self.ROLE_WEIGHTS["sprinter"])
            lane_weight = weights["lane"]
            if role == "flanker" and progress > 0:
                lane_weight *= self.LANE_RELAX_FACTOR
            score = (
                weights["progress"] * progress
                + weights["safety"] * safety
                + lane_weight * lane_score
            )

            if not self._is_safe(next_pos) and safety <= self.DANGER_DISTANCE:
                score -= self.DANGER_PENALTY
            if next_pos in avoid:
                score -= self.AVOID_PENALTY
            if (not is_carrier) and target_is_enemy and start_is_safe:
                if progress < 0:
                    score -= self.RETREAT_PENALTY * abs(progress)
                else:
                    score += self.SAFE_PUSH_WEIGHT * progress

            if best_score is None or score > best_score:
                best_score = score
                best_move = direction
                best_progress = progress

        if best_move and best_progress is not None and best_progress <= 0 and path_dir and not threatened:
            return path_dir
        return best_move or path_dir

    def plan_next_actions(self, req):
        if not self.world.update(req):
            return {}

        self.tick += 1
        self._ensure_roles()

        my_free = self.world.list_players(mine=True, inPrison=False, hasFlag=None) or []
        if not my_free:
            return {}

        opponents = self.world.list_players(mine=False, inPrison=False, hasFlag=None) or []
        enemy_flags = self.world.list_flags(mine=False, canPickup=True) or []
        targets = list(self.world.list_targets(mine=True) or [])

        enemy_positions = [self._pos(o) for o in opponents]
        enemy_positions_enemy_side = [
            self._pos(o) for o in opponents if not self._is_safe(self._pos(o))
        ]
        intruders = [
            self._pos(o)
            for o in opponents
            if o.get("hasFlag") and self._is_safe(self._pos(o))
        ]

        actions = {}
        flag_positions = [(int(f["posX"]), int(f["posY"])) for f in enemy_flags]

        for player in my_free:
            name = player["name"]
            role = self.role_by_name.get(name, "sprinter")
            lane_y = self.lane_by_role.get(role, self.center_y)
            start = self._pos(player)

            if player.get("hasFlag"):
                avoid = self._expanded_positions(enemy_positions_enemy_side, self.CARRIER_AVOID_RADIUS)
                target = self._closest_by_path(start, targets, avoid)
            else:
                avoid_radius = self.AVOID_RADIUS.get(role, 1)
                avoid = self._expanded_positions(enemy_positions_enemy_side, avoid_radius)
                target = None
                if intruders:
                    intruder, dist = self._closest_by_manhattan(start, intruders)
                    if dist is not None and dist <= self.INTERCEPT_RADIUS:
                        target = intruder
                if target is None:
                    target = self._select_flag_for_role(
                        start, flag_positions, role, lane_y, enemy_positions, avoid
                    )
                if target is None:
                    target = self._fallback_target(lane_y)

            move = self._choose_move(
                start, target, role, lane_y, enemy_positions, avoid, player.get("hasFlag")
            )
            if move:
                actions[name] = move

        return actions


AI = ExtremeCTFAI()


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
