from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from typing import Any, List, Mapping, Sequence, Tuple, Union

import numpy as np


# Matches AI_Design.md "Situations" (00-11).
SITUATION_ID = {
    "PLAYER": 0,
    "PLAYER_WITH_FLAG": 1,
    "OPPONENT_PLAYER": 2,
    "OPPONENT_PLAYER_WITH_FLAG": 3,
    "PRISON": 4,
    "HOME": 5,
    "HOME_WITH_FLAG": 6,
    "OPPONENT_HOME": 7,
    "BARRIER": 8,
    "BLANK": 9,
    "FLAG": 10,
    "OPPONENT_FLAG": 11,
}


@dataclass(frozen=True, slots=True)
class EncoderGeometry:
    width: int
    height: int
    mirror_x: bool

    def norm_pos(self, x: int, y: int) -> Tuple[int, int]:
        if not self.mirror_x:
            return x, y
        return (self.width - 1 - x), y


class TeamHistoryStateEncoder:
    def __init__(
        self,
        *,
        width: int = 20,
        height: int = 20,
        max_players: int = 3,
        history_len: int = 3,
        normalize_side: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.max_players = max_players
        self.history_len = history_len
        self.normalize_side = normalize_side

        self._geometry: EncoderGeometry | None = None
        self._static_grid: np.ndarray | None = None  # (H, W) int8 situation IDs
        self._histories: List[deque[np.ndarray]] = []

    @property
    def mirror_actions(self) -> bool:
        geom = self._geometry
        return bool(geom and geom.mirror_x)

    def start_game(self, init_req: Union[Mapping[str, Any], str]) -> None:
        if isinstance(init_req, str):
            init_req = json.loads(init_req)
        map_data = init_req.get("map") or {}
        self.width = int(map_data.get("width", self.width))
        self.height = int(map_data.get("height", self.height))

        my_targets = list(init_req.get("myteamTarget") or [])
        if not my_targets:
            raise ValueError("init payload missing myteamTarget")
        my_side_is_left = int(my_targets[0].get("x", 0)) < (self.width / 2.0)
        mirror_x = bool(self.normalize_side and (not my_side_is_left))

        self._geometry = EncoderGeometry(width=self.width, height=self.height, mirror_x=mirror_x)
        self._static_grid = self._build_static_grid(init_req)
        self._histories = [deque(maxlen=self.history_len) for _ in range(self.max_players)]

    def reset_history(self) -> None:
        self._histories = [deque(maxlen=self.history_len) for _ in range(self.max_players)]

    def denormalize_action(self, action_idx: int) -> int:
        if not self.mirror_actions:
            return int(action_idx)
        if int(action_idx) == 2:
            return 3
        if int(action_idx) == 3:
            return 2
        return int(action_idx)

    def _build_static_grid(self, init_req: Mapping[str, Any]) -> np.ndarray:
        geom = self._require_geom()
        map_data = init_req.get("map") or {}
        grid = np.full((geom.height, geom.width), SITUATION_ID["BLANK"], dtype=np.int8)

        for key in ("walls", "obstacles"):
            for item in map_data.get(key, []) or []:
                x, y = geom.norm_pos(int(item.get("x", 0)), int(item.get("y", 0)))
                if 0 <= x < geom.width and 0 <= y < geom.height:
                    grid[y, x] = SITUATION_ID["BARRIER"]

        for item in init_req.get("myteamPrison") or []:
            x, y = geom.norm_pos(int(item.get("x", 0)), int(item.get("y", 0)))
            if 0 <= x < geom.width and 0 <= y < geom.height:
                grid[y, x] = SITUATION_ID["PRISON"]

        for item in init_req.get("myteamTarget") or []:
            x, y = geom.norm_pos(int(item.get("x", 0)), int(item.get("y", 0)))
            if 0 <= x < geom.width and 0 <= y < geom.height:
                grid[y, x] = SITUATION_ID["HOME"]

        for item in init_req.get("opponentTarget") or []:
            x, y = geom.norm_pos(int(item.get("x", 0)), int(item.get("y", 0)))
            if 0 <= x < geom.width and 0 <= y < geom.height:
                grid[y, x] = SITUATION_ID["OPPONENT_HOME"]

        return grid

    def encode_team(
        self, status_req: Union[Mapping[str, Any], str]
    ) -> Tuple[np.ndarray, List[Mapping[str, Any]], np.ndarray]:
        if isinstance(status_req, str):
            status_req = json.loads(status_req)
        if self._static_grid is None:
            return np.zeros((0, 0, 0, 0), dtype=np.float32), [], np.zeros((0,), dtype=np.float32)

        my_players = _sorted_players(status_req.get("myteamPlayer") or [])
        opp_players = list(status_req.get("opponentPlayer") or [])

        base_dynamic = self._static_grid.copy()
        self._overlay_flags(base_dynamic, status_req)
        self._overlay_opponents(base_dynamic, opp_players)

        obs_list: List[np.ndarray] = []
        active_mask = np.zeros((self.max_players,), dtype=np.float32)

        for idx in range(self.max_players):
            if idx >= len(my_players):
                obs_list.append(np.zeros((self.history_len * 12, self.height, self.width), dtype=np.float32))
                active_mask[idx] = 0.0
                continue

            player = my_players[idx]
            grid = base_dynamic.copy()
            self._overlay_focus_player(grid, player)

            planes = _one_hot_12(grid)
            hist = self._histories[idx]
            if not hist:
                for _ in range(self.history_len):
                    hist.append(planes)
            else:
                hist.append(planes)
                while len(hist) < self.history_len:
                    hist.append(planes)

            stacked = np.stack(list(hist), axis=0)  # (T, 12, H, W)
            obs = stacked.reshape(self.history_len * 12, self.height, self.width).astype(np.float32, copy=False)
            obs_list.append(obs)

            active_mask[idx] = 0.0 if bool(player.get("inPrison")) else 1.0

        obs_batch = np.stack(obs_list, axis=0)  # (P, C, H, W)
        return obs_batch, my_players[: self.max_players], active_mask

    def _overlay_flags(self, grid: np.ndarray, status_req: Mapping[str, Any]) -> None:
        geom = self._require_geom()

        for f in status_req.get("myteamFlag") or []:
            x, y = geom.norm_pos(int(f.get("posX", 0)), int(f.get("posY", 0)))
            if 0 <= x < geom.width and 0 <= y < geom.height:
                grid[y, x] = SITUATION_ID["FLAG"]

        for f in status_req.get("opponentFlag") or []:
            x, y = geom.norm_pos(int(f.get("posX", 0)), int(f.get("posY", 0)))
            if not (0 <= x < geom.width and 0 <= y < geom.height):
                continue
            if bool(f.get("canPickup", True)):
                grid[y, x] = SITUATION_ID["OPPONENT_FLAG"]
            else:
                grid[y, x] = SITUATION_ID["HOME_WITH_FLAG"]

    def _overlay_opponents(self, grid: np.ndarray, opp_players: Sequence[Mapping[str, Any]]) -> None:
        geom = self._require_geom()
        for p in opp_players:
            x, y = geom.norm_pos(int(p.get("posX", 0)), int(p.get("posY", 0)))
            if not (0 <= x < geom.width and 0 <= y < geom.height):
                continue
            if bool(p.get("hasFlag")):
                grid[y, x] = SITUATION_ID["OPPONENT_PLAYER_WITH_FLAG"]
            else:
                grid[y, x] = SITUATION_ID["OPPONENT_PLAYER"]

    def _overlay_focus_player(self, grid: np.ndarray, player: Mapping[str, Any]) -> None:
        geom = self._require_geom()
        x, y = geom.norm_pos(int(player.get("posX", 0)), int(player.get("posY", 0)))
        if not (0 <= x < geom.width and 0 <= y < geom.height):
            return
        if bool(player.get("hasFlag")):
            grid[y, x] = SITUATION_ID["PLAYER_WITH_FLAG"]
        else:
            grid[y, x] = SITUATION_ID["PLAYER"]

    def _require_geom(self) -> EncoderGeometry:
        if self._geometry is None:
            raise RuntimeError("encoder not initialized; call start_game(init_req) first")
        return self._geometry


def _sorted_players(players: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(players, key=lambda p: str(p.get("name", "")))


def _one_hot_12(grid: np.ndarray) -> np.ndarray:
    height, width = grid.shape
    planes = np.zeros((12, height, width), dtype=np.float32)
    for idx in range(12):
        planes[idx] = (grid == idx).astype(np.float32)
    return planes
