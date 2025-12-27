from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Directions are identical to frontend/src/gameObjects/Player.js
DIR_UP = "up"
DIR_DOWN = "down"
DIR_LEFT = "left"
DIR_RIGHT = "right"

_DIR_TO_DELTA: Dict[Optional[str], Tuple[int, int]] = {
    None: (0, 0),
    "": (0, 0),
    DIR_UP: (0, -1),
    DIR_DOWN: (0, 1),
    DIR_LEFT: (-1, 0),
    DIR_RIGHT: (1, 0),
}

_INT_TO_DIR: Dict[int, Optional[str]] = {
    0: None,
    1: DIR_UP,
    2: DIR_DOWN,
    3: DIR_LEFT,
    4: DIR_RIGHT,
}


@dataclass(slots=True)
class FlagState:
    team: str  # "L" or "R" (same as frontend Flag.team)
    pos_x: int  # tile X
    pos_y: int  # tile Y
    can_pickup: bool

    def to_status(self) -> Dict[str, Any]:
        return {"canPickup": self.can_pickup, "posX": self.pos_x, "posY": self.pos_y}


@dataclass(slots=True)
class PlayerState:
    name: str
    team: str  # "L" or "R"
    x: int  # world X (pixels, centered on tile)
    y: int  # world Y (pixels, centered on tile)
    target_x: int  # world X target (pixels)
    target_y: int  # world Y target (pixels)
    sprite_choice: int
    in_prison: bool = False
    in_prison_time_left: float = 0.0
    in_prison_duration: float = 20000.0
    has_flag: bool = False
    remote_control: Optional[str] = None
    can_go_next_tile: bool = False

    def at_target(self) -> bool:
        return self.x == self.target_x and self.y == self.target_y

    def set_remote_control(self, direction: Optional[str]) -> None:
        self.remote_control = direction

    def collect_flag(self) -> None:
        self.has_flag = True

    def drop_flag(self) -> None:
        self.has_flag = False

    def to_prison(self, env: "MockEnvVNew", prison_tile_x: int, prison_tile_y: int) -> None:
        self.target_x, self.target_y = env.tile_to_world(prison_tile_x, prison_tile_y)
        self.x = self.target_x
        self.y = self.target_y
        self.in_prison = True
        self.in_prison_time_left = float(self.in_prison_duration)

    def choose_next_target(self, env: "MockEnvVNew") -> None:
        # Mirrors Player.checkInput() (remoteControl path only).
        if not (self.can_go_next_tile and self.at_target()):
            return
        self.can_go_next_tile = False

        dx_tile, dy_tile = env.direction_to_delta(self.remote_control)
        next_x = self.x + (dx_tile * env.tile_size_px)
        next_y = self.y + (dy_tile * env.tile_size_px)
        if not env.is_wall_world(next_x, next_y):
            self.target_x = next_x
            self.target_y = next_y

    def microstep_update(self, env: "MockEnvVNew") -> None:
        # Mirrors Player.update() inner while-loop body with a fixed dt=frame_duration_ms.
        if self.in_prison:
            self.in_prison_time_left -= env.frame_duration_ms
            if self.in_prison_time_left <= 0:
                self.in_prison = False
                self.in_prison_time_left = 0.0
            else:
                return

        # In frontend, once prison ends (timer hits 0), the same update loop will
        # immediately run checkInput() + move() in that same frameDuration tick.
        self.choose_next_target(env)

        if self.x < self.target_x:
            self.x += 1
        elif self.x > self.target_x:
            self.x -= 1

        if self.y < self.target_y:
            self.y += 1
        elif self.y > self.target_y:
            self.y -= 1

    def to_status(self, env: "MockEnvVNew") -> Dict[str, Any]:
        # Frontend sends the tile coords derived from target.{x,y}.
        pos_x, pos_y = env.world_to_tile(self.target_x, self.target_y)
        return {
            "name": self.name,
            "team": self.team,
            "hasFlag": self.has_flag,
            "posX": pos_x,
            "posY": pos_y,
            "inPrison": self.in_prison,
            "inPrisonTimeLeft": self.in_prison_time_left,
            "inPrisonDuration": self.in_prison_duration,
        }


class MockEnvVNew:
    """
    A faithful Python re-simulation of the Phaser frontend game rules:
    - CTF/frontend/src/scenes/Game.js
    - CTF/frontend/src/gameObjects/Player.js
    - CTF/frontend/src/gameObjects/Flag.js

    This is designed as a self-play / adversarial training foundation and emits
    the same init/status payload shapes as the WebSocket frontend.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        num_players: int = 3,
        num_flags: int = 9,
        *,
        use_random_flags: bool = False,
        num_obstacles_1: int = 8,
        num_obstacles_2: int = 4,
        tile_size_px: int = 32,
        move_speed_ms: float = 300.0,
        prison_duration_ms: float = 20000.0,
        seed: Optional[int] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.num_players = int(num_players)
        self.num_flags = int(num_flags)
        self.use_random_flags = bool(use_random_flags)

        self.num_obstacles_1 = int(num_obstacles_1)
        self.num_obstacles_2 = int(num_obstacles_2)

        self.tile_size_px = int(tile_size_px)
        self.half_tile_px = self.tile_size_px // 2
        self.move_speed_ms = float(move_speed_ms)
        self.frame_duration_ms = float(self.move_speed_ms) / float(self.tile_size_px)
        self.prison_duration_ms = float(prison_duration_ms)

        self._rng = random.Random(seed)

        # World origin (simplified vs Phaser's centered mapX/mapY). Only relative positions matter.
        self.map_x_px = 0
        self.map_y_px = 0
        self.map_offset_x_px = self.map_x_px + self.half_tile_px
        self.map_offset_y_px = self.map_y_px + self.half_tile_px
        self.center_x_px = self.map_x_px + (self.width * self.tile_size_px * 0.5)

        # Static map (generated on reset)
        self.walls: List[Tuple[int, int]] = []
        self.obstacles1: List[Tuple[int, int]] = []
        self.obstacles2: List[Tuple[int, int]] = []
        self._wall_set: set[Tuple[int, int]] = set()
        self._obstacle_set: set[Tuple[int, int]] = set()
        self._blocked_set: set[Tuple[int, int]] = set()

        # Zones (tiles)
        self.targets: Dict[str, List[Tuple[int, int]]] = {"L": [], "R": []}
        self.prisons: Dict[str, List[Tuple[int, int]]] = {"L": [], "R": []}

        # Zone rectangles in world coords: (cx, cy, half_w, half_h)
        self._target_zone: Dict[str, Tuple[float, float, float, float]] = {}
        self._prison_zone: Dict[str, Tuple[float, float, float, float]] = {}

        # Dynamic entities
        self.players: Dict[str, List[PlayerState]] = {"L": [], "R": []}
        self.flags: Dict[str, List[FlagState]] = {"L": [], "R": []}
        self.scores: Dict[str, int] = {"L": 0, "R": 0}
        self.time_ms: float = 0.0

        self.reset(seed=seed)

    @staticmethod
    def direction_to_delta(direction: Optional[str]) -> Tuple[int, int]:
        return _DIR_TO_DELTA.get(direction, (0, 0))

    @staticmethod
    def _normalize_action(action: Any) -> Optional[str]:
        # Frontend accepts only "up/down/left/right" (or undefined for no-op).
        # For training convenience, also accept ints 0..4.
        if action is None:
            return None
        if isinstance(action, int):
            return _INT_TO_DIR.get(action)
        if isinstance(action, str):
            return action
        return None

    def tile_to_world(self, tile_x: int, tile_y: int) -> Tuple[int, int]:
        return (
            int(self.map_offset_x_px + (tile_x * self.tile_size_px)),
            int(self.map_offset_y_px + (tile_y * self.tile_size_px)),
        )

    def world_to_tile(self, world_x: int, world_y: int) -> Tuple[int, int]:
        # Mirrors tilemapLayer.getTileAtWorldXY(..., true) behaviour closely (floor-based).
        tx = int(math.floor((world_x - self.map_x_px) / self.tile_size_px))
        ty = int(math.floor((world_y - self.map_y_px) / self.tile_size_px))
        return tx, ty

    def is_wall_world(self, world_x: int, world_y: int) -> bool:
        tile = self.world_to_tile(world_x, world_y)
        return tile in self._blocked_set

    @staticmethod
    def _create_3x3_grid(center_x: int, center_y: int) -> List[Tuple[int, int]]:
        # Mirrors Game.create3x3grid()
        return [
            (center_x - 1, center_y - 1),
            (center_x, center_y - 1),
            (center_x + 1, center_y - 1),
            (center_x - 1, center_y),
            (center_x, center_y),
            (center_x + 1, center_y),
            (center_x - 1, center_y + 1),
            (center_x, center_y + 1),
            (center_x + 1, center_y + 1),
        ]

    def _set_zone_rects(self) -> None:
        half_w = 1.5 * self.tile_size_px
        half_h = 1.5 * self.tile_size_px

        l_target_cx, l_target_cy = self.tile_to_world(2, self.height // 2)
        r_target_cx, r_target_cy = self.tile_to_world(self.width - 3, self.height // 2)
        l_prison_cx, l_prison_cy = self.tile_to_world(2, self.height - 3)
        r_prison_cx, r_prison_cy = self.tile_to_world(self.width - 3, self.height - 3)

        self._target_zone = {
            "L": (float(l_target_cx), float(l_target_cy), half_w, half_h),
            "R": (float(r_target_cx), float(r_target_cy), half_w, half_h),
        }
        self._prison_zone = {
            "L": (float(l_prison_cx), float(l_prison_cy), half_w, half_h),
            "R": (float(r_prison_cx), float(r_prison_cy), half_w, half_h),
        }

    @staticmethod
    def _aabb_overlap(
        ax: float,
        ay: float,
        ahw: float,
        ahh: float,
        bx: float,
        by: float,
        bhw: float,
        bhh: float,
    ) -> bool:
        return abs(ax - bx) < (ahw + bhw) and abs(ay - by) < (ahh + bhh)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)

        self.time_ms = 0.0
        self.scores = {"L": 0, "R": 0}

        self._generate_static_map()
        self._generate_zones()
        self._generate_entities()

        return self.get_full_state()

    def _generate_static_map(self) -> None:
        # Mirrors Game.initVariables() wall + obstacle generation logic.
        w, h = self.width, self.height

        self.walls = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        self.walls += [(x, 0) for x in range(1, w - 1)]
        self.walls += [(x, h - 1) for x in range(1, w - 1)]
        self.walls += [(0, y) for y in range(1, h - 1)]
        self.walls += [(w - 1, y) for y in range(1, h - 1)]

        self.obstacles1 = []
        for _ in range(self.num_obstacles_1):
            while True:
                x = self._rng.randint(4, w - 5)
                y = self._rng.randint(1, h - 2)
                if (x, y) not in self.obstacles1:
                    self.obstacles1.append((x, y))
                    break

        self.obstacles2 = []
        for _ in range(self.num_obstacles_2):
            while True:
                x = self._rng.randint(4, w - 5)
                y = self._rng.randint(1, h - 3)
                if (x, y) in self.obstacles1:
                    continue
                if (x, y + 1) in self.obstacles1:
                    continue
                if (x, y - 1) in self.obstacles2:
                    continue
                if (x, y) in self.obstacles2:
                    continue
                self.obstacles2.append((x, y))
                break

        self._wall_set = set(self.walls)
        self._obstacle_set = set(self.obstacles1)
        for ox, oy in self.obstacles2:
            self._obstacle_set.add((ox, oy))
            self._obstacle_set.add((ox, oy + 1))
        self._blocked_set = self._wall_set | self._obstacle_set

        self.center_x_px = self.map_x_px + (self.width * self.tile_size_px * 0.5)

    def _generate_zones(self) -> None:
        # Mirrors Game.initVariables() lteamState/rteamState target/prison definitions.
        l_target_center = (2, self.height // 2)
        r_target_center = (self.width - 3, self.height // 2)
        l_prison_center = (2, self.height - 3)
        r_prison_center = (self.width - 3, self.height - 3)

        self.targets = {
            "L": self._create_3x3_grid(*l_target_center),
            "R": self._create_3x3_grid(*r_target_center),
        }
        self.prisons = {
            "L": self._create_3x3_grid(*l_prison_center),
            "R": self._create_3x3_grid(*r_prison_center),
        }
        self._set_zone_rects()

    def _generate_entities(self) -> None:
        # Mirrors Game.initVariables() lteamState/rteamState flags/players definitions.
        w, h = self.width, self.height

        l_flags: List[Tuple[int, int]] = []
        r_flags: List[Tuple[int, int]] = []

        if self.use_random_flags:
            # Random flags match Game.js constraints.
            obstacles2_tops = set(self.obstacles2)
            for _ in range(self.num_flags):
                while True:
                    x = self._rng.randint(2, int(w / 2) - 1)
                    y = self._rng.randint(1, h - 3)
                    if (x, y) in self.obstacles1:
                        continue
                    if (x, y - 1) in obstacles2_tops or (x, y) in obstacles2_tops:
                        continue
                    if (x, y) in l_flags:
                        continue
                    l_flags.append((x, y))
                    break
            for _ in range(self.num_flags):
                while True:
                    x = self._rng.randint(int(w / 2), w - 2)
                    y = self._rng.randint(1, h - 3)
                    if (x, y) in self.obstacles1:
                        continue
                    if (x, y - 1) in obstacles2_tops or (x, y) in obstacles2_tops:
                        continue
                    if (x, y) in r_flags:
                        continue
                    r_flags.append((x, y))
                    break
        else:
            l_flags = [(1, i + 1) for i in range(self.num_flags)]
            r_flags = [(w - 2, i + 1) for i in range(self.num_flags)]

        self.flags = {
            "L": [FlagState(team="L", pos_x=x, pos_y=y, can_pickup=True) for (x, y) in l_flags],
            "R": [FlagState(team="R", pos_x=x, pos_y=y, can_pickup=True) for (x, y) in r_flags],
        }

        if self.use_random_flags:
            l_player_start_x = 1
            r_player_start_x = w - 2
        else:
            l_player_start_x = 2
            r_player_start_x = w - 3

        self.players = {"L": [], "R": []}
        for i in range(self.num_players):
            xw, yw = self.tile_to_world(l_player_start_x, i + 1)
            self.players["L"].append(
                PlayerState(
                    name=f"L{i}",
                    team="L",
                    x=xw,
                    y=yw,
                    target_x=xw,
                    target_y=yw,
                    sprite_choice=1,
                    in_prison_duration=self.prison_duration_ms,
                )
            )
        for i in range(self.num_players):
            xw, yw = self.tile_to_world(r_player_start_x, i + 1)
            self.players["R"].append(
                PlayerState(
                    name=f"R{i}",
                    team="R",
                    x=xw,
                    y=yw,
                    target_x=xw,
                    target_y=yw,
                    sprite_choice=4,
                    in_prison_duration=self.prison_duration_ms,
                )
            )

    def get_map_payload(self) -> Dict[str, Any]:
        obstacles_payload = list(self.obstacles1) + list(self.obstacles2) + [
            (x, y + 1) for (x, y) in self.obstacles2
        ]
        return {
            "width": self.width,
            "height": self.height,
            "walls": [{"x": x, "y": y} for (x, y) in self.walls],
            "obstacles": [{"x": x, "y": y} for (x, y) in obstacles_payload],
        }

    def get_init_payload(self, team_name: str) -> Dict[str, Any]:
        if team_name not in ("L", "R"):
            raise ValueError(f"Invalid team_name: {team_name!r}")
        opponent = "R" if team_name == "L" else "L"
        return {
            "action": "init",
            "map": self.get_map_payload(),
            "numPlayers": self.num_players,
            "numFlags": self.num_flags,
            "myteamName": team_name,
            "myteamPrison": [{"x": x, "y": y} for (x, y) in self.prisons[team_name]],
            "myteamTarget": [{"x": x, "y": y} for (x, y) in self.targets[team_name]],
            "opponentPrison": [{"x": x, "y": y} for (x, y) in self.prisons[opponent]],
            "opponentTarget": [{"x": x, "y": y} for (x, y) in self.targets[opponent]],
        }

    def load_from_init_payload(self, init_payload: Dict[str, Any]) -> None:
        """
        Optional helper: rebuild static map/zones from a frontend-style init payload.
        """
        m = init_payload["map"]
        self.width = int(m["width"])
        self.height = int(m["height"])
        self.center_x_px = self.map_x_px + (self.width * self.tile_size_px * 0.5)

        self.walls = [(w["x"], w["y"]) for w in m.get("walls", [])]
        self.obstacles1 = [(o["x"], o["y"]) for o in m.get("obstacles", [])]
        self.obstacles2 = []

        self._wall_set = set(self.walls)
        self._obstacle_set = set(self.obstacles1)
        self._blocked_set = self._wall_set | self._obstacle_set

        myteam = init_payload.get("myteamName", "L")
        opponent = "R" if myteam == "L" else "L"

        my_prison = [(p["x"], p["y"]) for p in init_payload.get("myteamPrison", [])]
        opp_prison = [(p["x"], p["y"]) for p in init_payload.get("opponentPrison", [])]
        my_target = [(t["x"], t["y"]) for t in init_payload.get("myteamTarget", [])]
        opp_target = [(t["x"], t["y"]) for t in init_payload.get("opponentTarget", [])]

        self.prisons = {myteam: my_prison, opponent: opp_prison}
        self.targets = {myteam: my_target, opponent: opp_target}
        self._set_zone_rects()

    def _find_available_prison_tile(self, team: str) -> Tuple[int, int]:
        prisons = self.prisons[team]
        occupied = set()
        for p in self.players[team]:
            if not p.in_prison:
                continue
            tx, ty = self.world_to_tile(p.x, p.y)
            occupied.add((tx, ty))
        for tile in prisons:
            if tile not in occupied:
                return tile
        return prisons[0]

    def _find_available_flag_tile(self, flag_team: str, targets: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
        # Mirrors Game.findAvailableFlagTile(flags, targets)
        for tx, ty in targets:
            is_available = True
            for f in self.flags[flag_team]:
                if f.can_pickup:
                    continue
                if (f.pos_x, f.pos_y) == (tx, ty):
                    is_available = False
                    break
            if is_available:
                return tx, ty
        return targets[0]

    def _remove_flag_item(self, flag: FlagState) -> None:
        self.flags[flag.team] = [f for f in self.flags[flag.team] if f is not flag]

    def _collect_flag(self, player: PlayerState, flag: FlagState) -> None:
        # Mirrors Game.collectFlag(player, flag)
        if player.team == flag.team:
            return
        if player.in_prison:
            return
        if player.has_flag:
            return
        if not flag.can_pickup:
            return
        self._remove_flag_item(flag)
        player.collect_flag()

    def _drop_flag_at_home(self, player: PlayerState) -> None:
        # Mirrors Game.dropFlag(player) when overlapping own target zone.
        if not player.has_flag:
            return
        player.drop_flag()

        if player.team == "L":
            spot_x, spot_y = self._find_available_flag_tile("R", self.targets["L"])
            self.flags["R"].append(FlagState(team="R", pos_x=spot_x, pos_y=spot_y, can_pickup=False))
            self._update_team_score("L")
        else:
            spot_x, spot_y = self._find_available_flag_tile("L", self.targets["R"])
            self.flags["L"].append(FlagState(team="L", pos_x=spot_x, pos_y=spot_y, can_pickup=False))
            self._update_team_score("R")

    def _update_team_score(self, team: str) -> None:
        self.scores[team] += 1

    def _hit_player(self, player1: PlayerState, player2: PlayerState) -> None:
        # Mirrors Game.hitPlayer(player1, player2)
        if player1.team == player2.team:
            return
        if player1.in_prison or player2.in_prison:
            return

        player_center_x = (player1.x + player2.x) / 2.0

        # In L team's side
        if player_center_x < self.center_x_px:
            caught = player1 if player1.team == "R" else player2
            prison_x, prison_y = self._find_available_prison_tile("R")
            if caught.has_flag:
                drop_tile_x, drop_tile_y = self.world_to_tile(caught.x, caught.y)
                self.flags["L"].append(FlagState(team="L", pos_x=drop_tile_x, pos_y=drop_tile_y, can_pickup=True))
                caught.has_flag = False
            caught.to_prison(self, prison_x, prison_y)
        else:
            caught = player1 if player1.team == "L" else player2
            prison_x, prison_y = self._find_available_prison_tile("L")
            if caught.has_flag:
                drop_tile_x, drop_tile_y = self.world_to_tile(caught.x, caught.y)
                self.flags["R"].append(FlagState(team="R", pos_x=drop_tile_x, pos_y=drop_tile_y, can_pickup=True))
                caught.has_flag = False
            caught.to_prison(self, prison_x, prison_y)

    def _free_players(self, team: str) -> None:
        # Mirrors Game.freePlayer(player): frees all teammates if a free player overlaps own prison zone.
        for p in self.players[team]:
            if p.in_prison:
                p.in_prison = False

    def step(
        self,
        actions_l: Dict[str, Any],
        actions_r: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, Any]], bool, Dict[str, Any]]:
        """
        Advance one decision "turn" (one tile move duration, 300ms by default).

        actions_l/actions_r map player name -> direction string ("up"/"down"/"left"/"right")
        Missing keys mean "no movement" (remoteControl becomes undefined in frontend).
        """
        # 1) Apply remote controls (frontend updatePlayerInfo behaviour).
        for p in self.players["L"]:
            p.set_remote_control(self._normalize_action(actions_l.get(p.name)))
        for p in self.players["R"]:
            p.set_remote_control(self._normalize_action(actions_r.get(p.name)))

        # 2) Allow each player to choose next tile (frontend can_go_next_tile gating).
        for p in self.players["L"] + self.players["R"]:
            p.can_go_next_tile = True

        # 3) Simulate microsteps (1px per frameDuration), processing overlaps in the same order as initPhysics().
        microsteps = self.tile_size_px
        player_half = self.tile_size_px / 2.0

        for _ in range(microsteps):
            for p in self.players["L"] + self.players["R"]:
                p.microstep_update(self)

            # 3.1) Player-player overlaps -> hitPlayer
            for lp in self.players["L"]:
                for rp in self.players["R"]:
                    if lp.in_prison or rp.in_prison:
                        continue
                    if self._aabb_overlap(
                        lp.x,
                        lp.y,
                        player_half,
                        player_half,
                        rp.x,
                        rp.y,
                        player_half,
                        player_half,
                    ):
                        self._hit_player(lp, rp)

            # 3.2) Player-flag overlaps -> collectFlag
            for lp in self.players["L"]:
                if lp.in_prison or lp.has_flag:
                    continue
                for f in list(self.flags["R"]):
                    if self._aabb_overlap(
                        lp.x,
                        lp.y,
                        player_half,
                        player_half,
                        *self.tile_to_world(f.pos_x, f.pos_y),
                        player_half,
                        player_half,
                    ):
                        self._collect_flag(lp, f)
                        if lp.has_flag:
                            break

            for rp in self.players["R"]:
                if rp.in_prison or rp.has_flag:
                    continue
                for f in list(self.flags["L"]):
                    if self._aabb_overlap(
                        rp.x,
                        rp.y,
                        player_half,
                        player_half,
                        *self.tile_to_world(f.pos_x, f.pos_y),
                        player_half,
                        player_half,
                    ):
                        self._collect_flag(rp, f)
                        if rp.has_flag:
                            break

            # 3.3) Player-target overlaps -> dropFlag
            l_zone = self._target_zone["L"]
            r_zone = self._target_zone["R"]
            for lp in self.players["L"]:
                if lp.in_prison or not lp.has_flag:
                    continue
                if self._aabb_overlap(lp.x, lp.y, player_half, player_half, *l_zone):
                    self._drop_flag_at_home(lp)

            for rp in self.players["R"]:
                if rp.in_prison or not rp.has_flag:
                    continue
                if self._aabb_overlap(rp.x, rp.y, player_half, player_half, *r_zone):
                    self._drop_flag_at_home(rp)

            # 3.4) Player-prison overlaps -> freePlayer
            l_prison_zone = self._prison_zone["L"]
            r_prison_zone = self._prison_zone["R"]
            for lp in self.players["L"]:
                if lp.in_prison:
                    continue
                if self._aabb_overlap(lp.x, lp.y, player_half, player_half, *l_prison_zone):
                    self._free_players("L")
                    break

            for rp in self.players["R"]:
                if rp.in_prison:
                    continue
                if self._aabb_overlap(rp.x, rp.y, player_half, player_half, *r_prison_zone):
                    self._free_players("R")
                    break

            self.time_ms += self.frame_duration_ms

        done = self.scores["L"] >= self.num_flags or self.scores["R"] >= self.num_flags
        winner = None
        if done:
            winner = "L" if self.scores["L"] >= self.num_flags else "R"

        return self.get_full_state(), done, {"winner": winner}

    def get_team_status(self, team: str) -> Dict[str, Any]:
        if team not in ("L", "R"):
            raise ValueError(f"Invalid team: {team!r}")
        opponent = "R" if team == "L" else "L"
        return {
            "action": "status",
            "time": self.time_ms,
            "myteamPlayer": [p.to_status(self) for p in self.players[team]],
            "myteamFlag": [f.to_status() for f in self.flags[team]],
            "myteamScore": self.scores[team],
            "opponentPlayer": [p.to_status(self) for p in self.players[opponent]],
            "opponentFlag": [f.to_status() for f in self.flags[opponent]],
            "opponentScore": self.scores[opponent],
        }

    def get_full_state(self) -> Dict[str, Dict[str, Any]]:
        return {"L": self.get_team_status("L"), "R": self.get_team_status("R")}


if __name__ == "__main__":
    env = MockEnvVNew(num_flags=6, seed=0)
    init_l = env.get_init_payload("L")
    print("Init(L) map size:", init_l["map"]["width"], init_l["map"]["height"])
    state = env.get_full_state()
    print("t=0 L0:", state["L"]["myteamPlayer"][0])

    for _ in range(5):
        state, done, info = env.step(actions_l={"L0": DIR_RIGHT}, actions_r={"R0": DIR_LEFT})
        print("t=", int(state["L"]["time"]), "L0:", state["L"]["myteamPlayer"][0]["posX"], state["L"]["myteamPlayer"][0]["posY"])
        if done:
            print("winner:", info["winner"])
            break
