import os
import sys
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from lib.matrix_util import CTFMatrixConverter  # noqa: E402


MAX_PLAYERS = 3

PLAYER_FEATURE_INDEX = {
    "x": 0,
    "y": 1,
    "has_flag": 2,
    "in_prison": 3,
    "team": 4,
    "index": 5,
    "valid": 6,
}


def _sorted_players(players):
    return sorted(players, key=lambda p: p.get("name", ""))


class StateEncoder:
    def __init__(self, width=20, height=20, max_players=MAX_PLAYERS):
        self.width = width
        self.height = height
        self.max_players = max_players
        self.converter = CTFMatrixConverter(width=width, height=height)
        self.initialized = False

    def start_game(self, init_req):
        map_data = init_req.get("map", {})
        width = map_data.get("width", self.width)
        height = map_data.get("height", self.height)
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            self.converter = CTFMatrixConverter(width=width, height=height)
        self.converter.initialize_static_map(init_req)
        self.initialized = True

    def encode(self, status_req):
        if not self.initialized:
            return None, None

        matrix = self.converter.convert_to_matrix(status_req)
        if matrix is None:
            return None, None

        grid_ids = matrix.astype(np.int64)
        player_features = self._build_player_features(status_req)
        return grid_ids, player_features

    def _build_player_features(self, status_req):
        my_players = _sorted_players(status_req.get("myteamPlayer", []))
        opp_players = _sorted_players(status_req.get("opponentPlayer", []))

        features = []
        features.extend(self._players_to_features(my_players, team_id=0))
        features.extend(self._players_to_features(opp_players, team_id=1))
        return np.array(features, dtype=np.int64)

    def _players_to_features(self, players, team_id):
        rows = []
        for idx in range(self.max_players):
            if idx < len(players):
                p = players[idx]
                rows.append([
                    int(round(p.get("posX", 0))),
                    int(round(p.get("posY", 0))),
                    1 if p.get("hasFlag", False) else 0,
                    1 if p.get("inPrison", False) else 0,
                    team_id,
                    idx,
                    1,
                ])
            else:
                rows.append([0, 0, 0, 0, team_id, idx, 0])
        return rows
