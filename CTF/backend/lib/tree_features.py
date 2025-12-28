from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Action = str  # "", "up", "down", "left", "right"


@dataclass(frozen=True, slots=True)
class Geometry:
    width: int
    height: int
    my_side_is_left: bool
    left_max_x: int
    blocked: frozenset[Tuple[int, int]]
    my_targets: Tuple[Tuple[int, int], ...]
    my_prisons: Tuple[Tuple[int, int], ...]

    @staticmethod
    def from_init(req: Mapping[str, Any]) -> "Geometry":
        map_data = req.get("map") or {}
        width = int(map_data.get("width", 0))
        height = int(map_data.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid map size: {width}x{height}")

        my_targets = tuple((int(t["x"]), int(t["y"])) for t in (req.get("myteamTarget") or []))
        my_prisons = tuple((int(p["x"]), int(p["y"])) for p in (req.get("myteamPrison") or []))
        if not my_targets:
            raise ValueError("Missing myteamTarget in init payload")

        my_side_is_left = int(my_targets[0][0]) < (width / 2.0)
        left_max_x = (width - 1) // 2

        blocked_list: List[Tuple[int, int]] = []
        for key in ("walls", "obstacles"):
            for w in map_data.get(key, []) or []:
                blocked_list.append((int(w["x"]), int(w["y"])))

        return Geometry(
            width=width,
            height=height,
            my_side_is_left=my_side_is_left,
            left_max_x=left_max_x,
            blocked=frozenset(blocked_list),
            my_targets=my_targets,
            my_prisons=my_prisons,
        )

    def normalize_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        x, y = pos
        if self.my_side_is_left:
            return x, y
        return (self.width - 1 - x), y

    def denormalize_pos(self, npos: Tuple[int, int]) -> Tuple[int, int]:
        x, y = npos
        if self.my_side_is_left:
            return x, y
        return (self.width - 1 - x), y

    def normalize_action(self, action: Action) -> Action:
        if self.my_side_is_left:
            return action
        if action == "left":
            return "right"
        if action == "right":
            return "left"
        return action

    def denormalize_action(self, action: Action) -> Action:
        # left/right swap is its own inverse
        return self.normalize_action(action)

    def is_safe_normalized(self, npos: Tuple[int, int]) -> bool:
        return npos[0] <= self.left_max_x

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, pos: Tuple[int, int]) -> bool:
        return pos in self.blocked

    def is_blocked_normalized(self, npos: Tuple[int, int]) -> bool:
        return self.denormalize_pos(npos) in self.blocked


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _min_manhattan(src: Tuple[int, int], targets: Sequence[Tuple[int, int]]) -> float:
    if not targets:
        return 999.0
    return float(min(_manhattan(src, t) for t in targets))


def _players_from_status(req: Mapping[str, Any], *, mine: bool) -> List[Mapping[str, Any]]:
    if mine:
        return list(req.get("myteamPlayer") or [])
    return list(req.get("opponentPlayer") or [])


def _flags_from_status(req: Mapping[str, Any], *, mine: bool, can_pickup: Optional[bool] = None) -> List[Mapping[str, Any]]:
    flags = list((req.get("myteamFlag") if mine else req.get("opponentFlag")) or [])
    if can_pickup is None:
        return flags
    return [f for f in flags if bool(f.get("canPickup")) == bool(can_pickup)]


def _pos(entity: Mapping[str, Any]) -> Tuple[int, int]:
    return (int(entity.get("posX", 0)), int(entity.get("posY", 0)))


def _count_where(items: Iterable[Mapping[str, Any]], key: str, value: bool) -> int:
    want = bool(value)
    return sum(1 for it in items if bool(it.get(key)) == want)


def _nearest_dist(src: Tuple[int, int], positions: Sequence[Tuple[int, int]]) -> float:
    if not positions:
        return 999.0
    return float(min(_manhattan(src, p) for p in positions))


def _next_pos(npos: Tuple[int, int], action: Action) -> Tuple[int, int]:
    x, y = npos
    if action == "up":
        return x, y - 1
    if action == "down":
        return x, y + 1
    if action == "left":
        return x - 1, y
    if action == "right":
        return x + 1, y
    return x, y


def extract_player_features(req: Mapping[str, Any], geometry: Geometry, player: Mapping[str, Any]) -> Dict[str, float]:
    pos = _pos(player)
    npos = geometry.normalize_pos(pos)

    my_players = _players_from_status(req, mine=True)
    opp_players = _players_from_status(req, mine=False)
    opp_flags = _flags_from_status(req, mine=False, can_pickup=True)

    opp_positions = [geometry.normalize_pos(_pos(p)) for p in opp_players if not bool(p.get("inPrison"))]
    opp_carriers = [
        geometry.normalize_pos(_pos(p))
        for p in opp_players
        if bool(p.get("hasFlag")) and not bool(p.get("inPrison"))
    ]
    intruders_with_flag = [p for p in opp_carriers if geometry.is_safe_normalized(p)]

    enemy_flag_positions = [geometry.normalize_pos(_pos(f)) for f in opp_flags]
    my_target_positions = [geometry.normalize_pos(t) for t in geometry.my_targets]
    my_prison_positions = [geometry.normalize_pos(p) for p in geometry.my_prisons]

    is_safe = geometry.is_safe_normalized(npos)
    num_my_prisoners = _count_where(my_players, "inPrison", True)
    num_opp_prisoners = _count_where(opp_players, "inPrison", True)

    features: Dict[str, float] = {
        "x": float(npos[0]),
        "y": float(npos[1]),
        "is_safe": 1.0 if is_safe else 0.0,
        "has_flag": 1.0 if bool(player.get("hasFlag")) else 0.0,
        "in_prison": 1.0 if bool(player.get("inPrison")) else 0.0,
        "num_my_prisoners": float(num_my_prisoners),
        "num_opp_prisoners": float(num_opp_prisoners),
        "num_intruders_with_flag": float(len(intruders_with_flag)),
        "dist_enemy_flag": _min_manhattan(npos, enemy_flag_positions),
        "dist_home_target": _min_manhattan(npos, my_target_positions),
        "dist_home_prison": _min_manhattan(npos, my_prison_positions),
        "dist_nearest_opp": _nearest_dist(npos, opp_positions),
        "dist_nearest_opp_carrier": _nearest_dist(npos, opp_carriers),
    }

    for action in ("up", "down", "left", "right"):
        nxt = _next_pos(npos, action)
        key = f"blocked_{action}"
        if not geometry.in_bounds(nxt) or geometry.is_blocked_normalized(nxt):
            features[key] = 1.0
        else:
            features[key] = 0.0

    return features


def allowed_actions(req: Mapping[str, Any], geometry: Geometry, player: Mapping[str, Any]) -> List[Action]:
    if bool(player.get("inPrison")):
        return [""]

    pos = _pos(player)
    npos = geometry.normalize_pos(pos)

    actions: List[Action] = [""]
    for action in ("up", "down", "left", "right"):
        nxt = _next_pos(npos, action)
        if not geometry.in_bounds(nxt):
            continue
        if geometry.is_blocked_normalized(nxt):
            continue
        actions.append(action)
    return actions
