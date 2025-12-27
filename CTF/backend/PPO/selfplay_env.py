import json
import random
from pathlib import Path


MOVE_DELTAS = {
    0: (0, 0),
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


class CTFFrontendRulesEnv:
    def __init__(self, config):
        self.width = config.get("width", 20)
        self.height = config.get("height", 20)
        self.num_players = config.get("num_players", 3)
        self.num_flags = config.get("num_flags", 9)
        self.use_random_flags = bool(config.get("use_random_flags", False))
        self.num_obstacles_1 = config.get("num_obstacles_1", 8)
        self.num_obstacles_2 = config.get("num_obstacles_2", 4)
        self.prison_duration = config.get("prison_duration", 20)
        self.max_steps = config.get("max_steps", 400)
        self.rng = random.Random(config.get("seed", 0))

        init_path = config.get("init_path")
        if init_path:
            self._load_init(init_path)
        else:
            self._generate_map()

        self._build_init_requests()
        self.reset()

    def _load_init(self, init_path):
        path = Path(__file__).resolve().parent / init_path
        with open(path, "r", encoding="utf-8") as handle:
            self.init_data = json.load(handle)

        map_data = self.init_data.get("map", {})
        self.width = map_data.get("width", self.width)
        self.height = map_data.get("height", self.height)
        self.midline = self.width / 2
        self.walls = {(w["x"], w["y"]) for w in map_data.get("walls", [])}
        obstacles = map_data.get("obstacles", [])
        self.obstacles = {(o["x"], o["y"]) for o in obstacles}
        self.obstacles1 = []
        self.obstacles2 = []
        self.targets = {
            "L": [(t["x"], t["y"]) for t in self.init_data.get("myteamTarget", [])],
            "R": [(t["x"], t["y"]) for t in self.init_data.get("opponentTarget", [])],
        }
        self.prisons = {
            "L": [(t["x"], t["y"]) for t in self.init_data.get("myteamPrison", [])],
            "R": [(t["x"], t["y"]) for t in self.init_data.get("opponentPrison", [])],
        }

    def _generate_map(self):
        self.midline = self.width / 2
        self.walls = set()
        self.obstacles = set()

        walls = []
        walls.append({"x": 0, "y": 0})
        walls.append({"x": self.width - 1, "y": 0})
        walls.append({"x": 0, "y": self.height - 1})
        walls.append({"x": self.width - 1, "y": self.height - 1})

        for i in range(1, self.width - 1):
            walls.append({"x": i, "y": 0})
            walls.append({"x": i, "y": self.height - 1})
        for i in range(1, self.height - 1):
            walls.append({"x": 0, "y": i})
            walls.append({"x": self.width - 1, "y": i})

        self.walls = {(w["x"], w["y"]) for w in walls}

        obstacles1 = []
        obstacles2 = []

        def not_contains(arr, x, y):
            return all(o["x"] != x or o["y"] != y for o in arr)

        for _ in range(self.num_obstacles_1):
            while True:
                x = self.rng.randint(4, self.width - 5)
                y = self.rng.randint(1, self.height - 2)
                if not_contains(obstacles1, x, y):
                    obstacles1.append({"x": x, "y": y})
                    break

        for _ in range(self.num_obstacles_2):
            while True:
                x = self.rng.randint(4, self.width - 5)
                y = self.rng.randint(1, self.height - 3)
                if (
                    not_contains(obstacles1, x, y)
                    and not_contains(obstacles1, x, y + 1)
                    and not_contains(obstacles2, x, y - 1)
                    and not_contains(obstacles2, x, y)
                ):
                    obstacles2.append({"x": x, "y": y})
                    break

        obstacle_tiles = []
        obstacle_tiles.extend(obstacles1)
        obstacle_tiles.extend(obstacles2)
        obstacle_tiles.extend([{"x": o["x"], "y": o["y"] + 1} for o in obstacles2])
        self.obstacles = {(o["x"], o["y"]) for o in obstacle_tiles}
        self.obstacles1 = obstacles1
        self.obstacles2 = obstacles2

        self.targets = {
            "L": self._create_3x3_grid(2, self.height // 2),
            "R": self._create_3x3_grid(self.width - 3, self.height // 2),
        }
        self.prisons = {
            "L": self._create_3x3_grid(2, self.height - 3),
            "R": self._create_3x3_grid(self.width - 3, self.height - 3),
        }

        self.init_data = {
            "map": {
                "width": self.width,
                "height": self.height,
                "walls": [{"x": x, "y": y} for (x, y) in sorted(self.walls)],
                "obstacles": [{"x": x, "y": y} for (x, y) in sorted(self.obstacles)],
            },
            "myteamPrison": [{"x": x, "y": y} for (x, y) in self.prisons["L"]],
            "myteamTarget": [{"x": x, "y": y} for (x, y) in self.targets["L"]],
            "opponentPrison": [{"x": x, "y": y} for (x, y) in self.prisons["R"]],
            "opponentTarget": [{"x": x, "y": y} for (x, y) in self.targets["R"]],
        }

    def _create_3x3_grid(self, x, y):
        return [
            (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
            (x - 1, y),     (x, y),     (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1),
        ]

    def _build_init_requests(self):
        map_data = self.init_data["map"]
        left = {
            "action": "init",
            "map": map_data,
            "numPlayers": self.num_players,
            "numFlags": self.num_flags,
            "myteamName": "L",
            "myteamPrison": [{"x": x, "y": y} for (x, y) in self.prisons["L"]],
            "myteamTarget": [{"x": x, "y": y} for (x, y) in self.targets["L"]],
            "opponentPrison": [{"x": x, "y": y} for (x, y) in self.prisons["R"]],
            "opponentTarget": [{"x": x, "y": y} for (x, y) in self.targets["R"]],
        }
        right = {
            "action": "init",
            "map": map_data,
            "numPlayers": self.num_players,
            "numFlags": self.num_flags,
            "myteamName": "R",
            "myteamPrison": [{"x": x, "y": y} for (x, y) in self.prisons["R"]],
            "myteamTarget": [{"x": x, "y": y} for (x, y) in self.targets["R"]],
            "opponentPrison": [{"x": x, "y": y} for (x, y) in self.prisons["L"]],
            "opponentTarget": [{"x": x, "y": y} for (x, y) in self.targets["L"]],
        }
        self.init_req = {"L": left, "R": right}

    def reset(self):
        self.time = 0.0
        self.score = {"L": 0, "R": 0}
        self.players = {
            "L": self._spawn_players("L"),
            "R": self._spawn_players("R"),
        }
        self.flags = {
            "L": self._spawn_flags("L"),
            "R": self._spawn_flags("R"),
        }
        return self.get_status("L"), self.get_status("R")

    def _spawn_players(self, team):
        players = []
        for idx in range(self.num_players):
            if self.use_random_flags:
                x = 1 if team == "L" else self.width - 2
            else:
                x = 2 if team == "L" else self.width - 3
            y = idx + 1
            players.append({
                "name": f"{team}{idx}",
                "team": team,
                "posX": x,
                "posY": y,
                "hasFlag": False,
                "inPrison": False,
                "inPrisonTimeLeft": 0,
                "inPrisonDuration": self.prison_duration,
            })
        return players

    def _spawn_flags(self, team):
        flags = []
        if not self.use_random_flags:
            x = 1 if team == "L" else self.width - 2
            for i in range(self.num_flags):
                flags.append({
                    "posX": x,
                    "posY": i + 1,
                    "canPickup": True,
                })
            return flags

        obstacles1 = {(o["x"], o["y"]) for o in self.obstacles1}
        obstacles2 = {(o["x"], o["y"]) for o in self.obstacles2}
        existing = set()
        for _ in range(self.num_flags):
            while True:
                if team == "L":
                    x = self.rng.randint(2, int(self.width / 2) - 1)
                else:
                    x = self.rng.randint(int(self.width / 2), self.width - 2)
                y = self.rng.randint(1, self.height - 3)

                if (x, y) in existing:
                    continue
                if (x, y) in self.walls or (x, y) in self.obstacles:
                    continue
                if (x, y) in obstacles1:
                    continue
                if (x, y) in obstacles2 or (x, y - 1) in obstacles2:
                    continue

                existing.add((x, y))
                flags.append({
                    "posX": x,
                    "posY": y,
                    "canPickup": True,
                })
                break
        return flags

    def step(self, actions_L, actions_R):
        self.time += 1.0
        self._tick_prisoners()
        self._apply_moves("L", actions_L)
        self._apply_moves("R", actions_R)
        self._handle_captures()
        self._handle_pickups("L")
        self._handle_pickups("R")
        self._handle_flag_drop("L")
        self._handle_flag_drop("R")
        self._handle_prison_frees("L")
        self._handle_prison_frees("R")

        done = (
            self.score["L"] >= self.num_flags
            or self.score["R"] >= self.num_flags
            or self.time >= self.max_steps
        )
        return self.get_status("L"), self.get_status("R"), done

    def _tick_prisoners(self):
        for team in ("L", "R"):
            for p in self.players[team]:
                if p["inPrison"]:
                    p["inPrisonTimeLeft"] -= 1
                    if p["inPrisonTimeLeft"] <= 0:
                        p["inPrison"] = False
                        p["inPrisonTimeLeft"] = 0

    def _apply_moves(self, team, actions):
        for idx, p in enumerate(self.players[team]):
            if p["inPrison"]:
                continue
            action = actions[idx] if idx < len(actions) else 0
            dx, dy = MOVE_DELTAS.get(action, (0, 0))
            nx = p["posX"] + dx
            ny = p["posY"] + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            if (nx, ny) in self.walls or (nx, ny) in self.obstacles:
                continue
            p["posX"] = nx
            p["posY"] = ny

    def _handle_captures(self):
        positions = {}
        for team in ("L", "R"):
            for p in self.players[team]:
                if p["inPrison"]:
                    continue
                positions.setdefault((p["posX"], p["posY"]), []).append(p)

        for (x, _), occupants in positions.items():
            teams_present = {p["team"] for p in occupants}
            if len(teams_present) < 2:
                continue
            if x < self.midline:
                self._capture_team("R", occupants, territory="L")
            else:
                self._capture_team("L", occupants, territory="R")

    def _capture_team(self, team, occupants, territory):
        for p in occupants:
            if p["team"] != team or p["inPrison"]:
                continue
            if p["hasFlag"]:
                self.flags[territory].append({
                    "posX": p["posX"],
                    "posY": p["posY"],
                    "canPickup": True,
                })
                p["hasFlag"] = False
            tile = self._find_prison_tile(team)
            p["posX"] = tile[0]
            p["posY"] = tile[1]
            p["inPrison"] = True
            p["inPrisonTimeLeft"] = self.prison_duration

    def _find_prison_tile(self, team):
        occupied = {(p["posX"], p["posY"]) for p in self.players[team] if p["inPrison"]}
        for tile in self.prisons[team]:
            if tile not in occupied:
                return tile
        return self.prisons[team][0] if self.prisons[team] else (0, 0)

    def _handle_pickups(self, team):
        opponent = "L" if team == "R" else "R"
        for p in self.players[team]:
            if p["inPrison"] or p["hasFlag"]:
                continue
            for flag in list(self.flags[opponent]):
                if not flag["canPickup"]:
                    continue
                if flag["posX"] == p["posX"] and flag["posY"] == p["posY"]:
                    self.flags[opponent].remove(flag)
                    p["hasFlag"] = True
                    break

    def _handle_flag_drop(self, team):
        opponent = "L" if team == "R" else "R"
        for p in self.players[team]:
            if not p["hasFlag"]:
                continue
            if (p["posX"], p["posY"]) not in set(self.targets[team]):
                continue
            p["hasFlag"] = False
            self.score[team] += 1
            spot = self._find_available_flag_tile(opponent, self.targets[team])
            self.flags[opponent].append({
                "posX": spot[0],
                "posY": spot[1],
                "canPickup": False,
            })

    def _find_available_flag_tile(self, opponent_team, targets):
        occupied = {
            (f["posX"], f["posY"])
            for f in self.flags[opponent_team]
            if not f["canPickup"]
        }
        for (x, y) in targets:
            if (x, y) not in occupied:
                return (x, y)
        return targets[0] if targets else (0, 0)

    def _handle_prison_frees(self, team):
        freed = False
        for p in self.players[team]:
            if p["inPrison"]:
                continue
            if (p["posX"], p["posY"]) in set(self.prisons[team]):
                freed = True
                break
        if freed:
            for p in self.players[team]:
                if p["inPrison"]:
                    p["inPrison"] = False
                    p["inPrisonTimeLeft"] = 0

    def get_status(self, team):
        opponent = "L" if team == "R" else "R"
        return {
            "action": "status",
            "time": float(self.time),
            "myteamPlayer": [p.copy() for p in self.players[team]],
            "opponentPlayer": [p.copy() for p in self.players[opponent]],
            "myteamFlag": [f.copy() for f in self.flags[team]],
            "opponentFlag": [f.copy() for f in self.flags[opponent]],
            "myteamScore": self.score[team],
            "opponentScore": self.score[opponent],
        }
