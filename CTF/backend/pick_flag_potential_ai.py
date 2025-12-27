import asyncio
import json
import math
import os
import random

from lib.game_engine import GameMap, run_game_server


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "pick_flag_potential_ai.json")

DEFAULT_CONFIG = {
    "version": 1,
    "distance_metric": "manhattan",
    "potential": {
        "flag": {"weight": -8.0, "decay": 1.2, "max_distance": 30},
        "home": {"weight": -10.0, "decay": 1.2, "max_distance": 30},
        "enemy": {"weight": 12.0, "decay": 1.0, "max_distance": 8},
        "obstacle": {"weight": 10.0, "decay": 1.0, "max_distance": 6},
    },
    "movement": {
        "allow_stay": True,
        "random_tie_break": True,
    },
    "debug": {
        "log_every_n_ticks": 0
    },
}


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path):
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            file_config = json.load(handle)
        deep_update(config, file_config)
    except FileNotFoundError:
        print(f"Config not found at {path}, using defaults.")
    except Exception as exc:
        print(f"Failed to load config {path}: {exc}. Using defaults.")
    return config


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_distance_fn(metric):
    if metric == "euclidean":
        return euclidean
    return manhattan


def normalize_positions(items):
    positions = []
    for item in items or []:
        if isinstance(item, dict):
            if "posX" in item and "posY" in item:
                positions.append((to_grid(item["posX"]), to_grid(item["posY"])))
            elif "x" in item and "y" in item:
                positions.append((to_grid(item["x"]), to_grid(item["y"])))
        else:
            positions.append((to_grid(item[0]), to_grid(item[1])))
    return positions


def to_grid(value):
    return int(round(value))


def potential_from_sources(x, y, sources, spec, distance_fn):
    if not sources:
        return 0.0
    weight = float(spec.get("weight", 0.0))
    if weight == 0.0:
        return 0.0
    decay = float(spec.get("decay", 1.0))
    max_distance = spec.get("max_distance", None)
    if max_distance is not None:
        try:
            max_distance = float(max_distance)
        except (TypeError, ValueError):
            max_distance = None

    total = 0.0
    for sx, sy in sources:
        dist = distance_fn((x, y), (sx, sy))
        if max_distance is not None and dist > max_distance:
            continue
        total += weight / ((dist + 1.0) ** decay)
    return total


def build_static_obstacle_field(width, height, obstacles, spec, distance_fn):
    field = [[0.0 for _ in range(width)] for _ in range(height)]
    if not obstacles:
        return field
    for y in range(height):
        for x in range(width):
            field[y][x] = potential_from_sources(x, y, obstacles, spec, distance_fn)
    return field


class PotentialFieldAI:
    MOVES = [
        ("up", 0, -1),
        ("down", 0, 1),
        ("left", -1, 0),
        ("right", 1, 0),
    ]

    def __init__(self):
        self.world = GameMap(show_gap_in_msec=1000.0)
        self.config = load_config(CONFIG_PATH)
        self.distance_fn = get_distance_fn(self.config.get("distance_metric"))
        self.map_width = 0
        self.map_height = 0
        self.blocked = set()
        self.obstacle_field = []
        self.tick_count = 0

    def reload_config(self):
        self.config = load_config(CONFIG_PATH)
        self.distance_fn = get_distance_fn(self.config.get("distance_metric"))

    def start_game(self, req):
        self.reload_config()
        self.world.init(req)
        self.tick_count = 0

        map_info = req.get("map", {})
        self.map_width = map_info.get("width", 0)
        self.map_height = map_info.get("height", 0)
        obstacles = normalize_positions(map_info.get("obstacles", []))
        walls = normalize_positions(map_info.get("walls", []))
        self.blocked = set(obstacles + walls)

        obstacle_spec = self.config["potential"]["obstacle"]
        self.obstacle_field = build_static_obstacle_field(
            self.map_width, self.map_height, list(self.blocked), obstacle_spec, self.distance_fn
        )
        print("Potential-field AI started.")

    def game_over(self, _req):
        print("Game Over!")

    def in_bounds(self, x, y):
        return 0 <= x < self.map_width and 0 <= y < self.map_height

    def cell_potential(self, x, y, enemy_positions, attract_positions, attract_spec):
        potential = self.obstacle_field[y][x]
        enemy_spec = self.config["potential"]["enemy"]
        potential += potential_from_sources(x, y, enemy_positions, enemy_spec, self.distance_fn)
        potential += potential_from_sources(x, y, attract_positions, attract_spec, self.distance_fn)
        return potential

    def plan_next_actions(self, req):
        if not self.world.update(req):
            return

        self.tick_count += 1
        log_every = self.config.get("debug", {}).get("log_every_n_ticks", 0)
        if log_every and self.tick_count % log_every == 0:
            print(f"Tick {self.tick_count}: planning actions")

        my_players = self.world.list_players(mine=True, inPrison=False, hasFlag=None) or []
        opponents = self.world.list_players(mine=False, inPrison=False, hasFlag=None) or []
        enemy_flags = self.world.list_flags(mine=False, canPickup=True) or []
        my_targets = list(self.world.list_targets(mine=True) or [])

        enemy_positions = normalize_positions(opponents)
        enemy_flag_positions = normalize_positions(enemy_flags)
        my_target_positions = normalize_positions(my_targets)

        allow_stay = bool(self.config["movement"].get("allow_stay", True))
        random_tie_break = bool(self.config["movement"].get("random_tie_break", True))

        actions = {}
        for player in my_players:
            start_x = to_grid(player["posX"])
            start_y = to_grid(player["posY"])
            if not self.in_bounds(start_x, start_y):
                continue

            if player["hasFlag"]:
                attract_positions = my_target_positions
                attract_spec = self.config["potential"]["home"]
            else:
                attract_positions = enemy_flag_positions
                attract_spec = self.config["potential"]["flag"]

            candidates = []
            best_potential = None

            if allow_stay:
                stay_potential = self.cell_potential(
                    start_x, start_y, enemy_positions, attract_positions, attract_spec
                )
                candidates.append((None, stay_potential))
                best_potential = stay_potential

            for direction, dx, dy in self.MOVES:
                nx = start_x + dx
                ny = start_y + dy
                if not self.in_bounds(nx, ny):
                    continue
                if (nx, ny) in self.blocked:
                    continue
                potential = self.cell_potential(nx, ny, enemy_positions, attract_positions, attract_spec)
                if best_potential is None or potential < best_potential - 1e-6:
                    best_potential = potential
                    candidates = [(direction, potential)]
                elif abs(potential - best_potential) <= 1e-6:
                    candidates.append((direction, potential))

            if not candidates:
                continue

            chosen_direction = None
            if random_tie_break and len(candidates) > 1:
                chosen_direction = random.choice(candidates)[0]
            else:
                chosen_direction = candidates[0][0]

            if chosen_direction:
                actions[player["name"]] = chosen_direction

        return actions


AI = PotentialFieldAI()


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
