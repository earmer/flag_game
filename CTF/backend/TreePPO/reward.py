from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence, Tuple, Union


MOVE_DELTAS_ENV = {
    0: (0, 0),   # stay
    1: (0, -1),  # up
    2: (0, 1),   # down
    3: (-1, 0),  # left
    4: (1, 0),   # right
}


def _sorted_players(players: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(players, key=lambda p: str(p.get("name", "")))


def _count_where(players: Sequence[Mapping[str, Any]], key: str, value: bool) -> int:
    want = bool(value)
    return sum(1 for p in players if bool(p.get(key)) == want)


def _player_by_name(players: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(p.get("name", "")): p for p in players}


@dataclass
class RewardTracker:
    width: int = 20
    height: int = 20
    max_players: int = 3
    my_side_is_left: bool = True
    blocked: set[Tuple[int, int]] | None = None

    initial_opponent_flag_x_dist: float = 0.0
    my_prison_tiles: set[Tuple[int, int]] | None = None

    def start_game(self, init_req: Union[Mapping[str, Any], str]) -> None:
        if isinstance(init_req, str):
            init_req = json.loads(init_req)
        map_data = init_req.get("map") or {}
        self.width = int(map_data.get("width", self.width))
        self.height = int(map_data.get("height", self.height))

        my_targets = list(init_req.get("myteamTarget") or [])
        if not my_targets:
            raise ValueError("init payload missing myteamTarget")
        self.my_side_is_left = int(my_targets[0].get("x", 0)) < (self.width / 2.0)

        blocked: set[Tuple[int, int]] = set()
        for key in ("walls", "obstacles"):
            for item in map_data.get(key, []) or []:
                blocked.add((int(item.get("x", 0)), int(item.get("y", 0))))
        self.blocked = blocked
        self.my_prison_tiles = {
            (int(p.get("x", 0)), int(p.get("y", 0))) for p in (init_req.get("myteamPrison") or [])
        }
        self.initial_opponent_flag_x_dist = float(self.width - 1)

    def reset_episode(self, initial_status: Union[Mapping[str, Any], str]) -> None:
        if isinstance(initial_status, str):
            initial_status = json.loads(initial_status)
        opp_flags = [
            f
            for f in (initial_status.get("opponentFlag") or [])
            if bool(f.get("canPickup", True))
        ]
        if not opp_flags:
            self.initial_opponent_flag_x_dist = float(self.width - 1)
            return
        self.initial_opponent_flag_x_dist = float(
            max(self._x_dist(int(f.get("posX", 0))) for f in opp_flags)
        )

    def _x_dist(self, x: int) -> int:
        if self.my_side_is_left:
            return int(x)
        return int(self.width - 1 - x)

    def _predict_next_pos(self, prev_x: int, prev_y: int, action_env: int) -> Tuple[int, int]:
        dx, dy = MOVE_DELTAS_ENV.get(int(action_env), (0, 0))
        nx = prev_x + dx
        ny = prev_y + dy
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return prev_x, prev_y
        if self.blocked and (nx, ny) in self.blocked:
            return prev_x, prev_y
        return nx, ny

    def compute(
        self,
        prev_status: Union[Mapping[str, Any], str],
        curr_status: Union[Mapping[str, Any], str],
        *,
        done: bool,
        actions_env: Sequence[int] | None = None,
    ) -> float:
        actions_env = list(actions_env or [])
        if isinstance(prev_status, str):
            prev_status = json.loads(prev_status)
        if isinstance(curr_status, str):
            curr_status = json.loads(curr_status)

        prev_my = _sorted_players(prev_status.get("myteamPlayer") or [])[: self.max_players]
        curr_my = _sorted_players(curr_status.get("myteamPlayer") or [])[: self.max_players]
        prev_opp = list(prev_status.get("opponentPlayer") or [])
        curr_opp = list(curr_status.get("opponentPlayer") or [])

        prev_my_map = _player_by_name(prev_my)
        curr_my_map = _player_by_name(curr_my)

        reward = 0.0

        # 被对方拿到旗子 -1
        prev_opp_carriers = _count_where(prev_opp, "hasFlag", True)
        curr_opp_carriers = _count_where(curr_opp, "hasFlag", True)
        reward += -1.0 * max(0, curr_opp_carriers - prev_opp_carriers)

        # 被对方将旗子送还 -3
        prev_opp_score = float(prev_status.get("opponentScore", 0.0))
        curr_opp_score = float(curr_status.get("opponentScore", prev_opp_score))
        reward += -3.0 * max(0.0, curr_opp_score - prev_opp_score)

        # 拿到旗子 +5
        prev_my_carriers = _count_where(prev_my, "hasFlag", True)
        curr_my_carriers = _count_where(curr_my, "hasFlag", True)
        reward += 5.0 * max(0, curr_my_carriers - prev_my_carriers)

        # 送还旗子 +10
        prev_my_score = float(prev_status.get("myteamScore", 0.0))
        curr_my_score = float(curr_status.get("myteamScore", prev_my_score))
        reward += 10.0 * max(0.0, curr_my_score - prev_my_score)

        # 被关进监狱 -1
        prev_my_prison = _count_where(prev_my, "inPrison", True)
        curr_my_prison = _count_where(curr_my, "inPrison", True)
        reward += -1.0 * max(0, curr_my_prison - prev_my_prison)

        # 解救队友 +2
        # In the real frontend rules, a rescue happens when a free teammate overlaps the prison zone
        # and frees all jailed teammates. Detect this by:
        # - at least one player transitions inPrison True->False, and
        # - at least one currently-free teammate is standing on a prison tile.
        freed_names = [
            name
            for name, prev_p in prev_my_map.items()
            if bool(prev_p.get("inPrison"))
            and (curr_my_map.get(name) is not None)
            and (not bool(curr_my_map[name].get("inPrison")))
        ]
        rescued = 0
        if freed_names and self.my_prison_tiles:
            rescuer_present = any(
                (int(p.get("posX", 0)), int(p.get("posY", 0))) in self.my_prison_tiles
                and (not bool(p.get("inPrison")))
                for p in curr_my
            )
            if rescuer_present:
                rescued = len(freed_names)
        reward += 2.0 * float(rescued)

        # 被关进监狱时：若(己方带着对方旗子)的旗在 x 轴上离我方更近了 k 格，则 +k/2
        prev_opp_flags = [
            f for f in (prev_status.get("opponentFlag") or []) if bool(f.get("canPickup", True))
        ]
        curr_opp_flags = [
            f for f in (curr_status.get("opponentFlag") or []) if bool(f.get("canPickup", True))
        ]
        prev_flag_positions = {(int(f.get("posX", 0)), int(f.get("posY", 0))) for f in prev_opp_flags}
        new_flag_positions = [
            (int(f.get("posX", 0)), int(f.get("posY", 0)))
            for f in curr_opp_flags
            if (int(f.get("posX", 0)), int(f.get("posY", 0))) not in prev_flag_positions
        ]

        progress_reward = 0.0
        for _idx, prev_p in enumerate(prev_my):
            name = str(prev_p.get("name", ""))
            curr_p = curr_my_map.get(name)
            if curr_p is None:
                continue
            if bool(prev_p.get("inPrison")) or (not bool(curr_p.get("inPrison"))):
                continue
            if not bool(prev_p.get("hasFlag")):
                continue
            if not new_flag_positions:
                continue
            prev_x = int(prev_p.get("posX", 0))
            prev_y = int(prev_p.get("posY", 0))
            drop_x, _drop_y = min(
                new_flag_positions,
                key=lambda xy: abs(int(xy[0]) - prev_x) + abs(int(xy[1]) - prev_y),
            )
            new_flag_positions.remove((drop_x, _drop_y))
            k = float(self.initial_opponent_flag_x_dist - self._x_dist(int(drop_x)))
            if k > 0:
                progress_reward += k / 2.0
        reward += progress_reward

        # 游戏胜利 +15
        if done and (curr_my_score > curr_opp_score):
            reward += 15.0

        return float(reward)
