from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class StateNormalizer:
    """
    Normalize the observation so that the learning agent is always on the left side.

    Notes:
    - `CTFMatrixConverter` already encodes entities as "my team" vs "opponent", so for
      the default mode we only need to mirror the map for team "R".
    - The optional `swap_ids` mode exists to match older plans that used absolute "L/R"
      IDs (not currently used by `CTFMatrixConverter`).
    """

    def __init__(self, *, swap_ids: bool = False) -> None:
        self.swap_ids = bool(swap_ids)

        # Only used when swap_ids=True (see docstring).
        self._id_swap_map = {
            # my players <-> opponent players
            0: 6,
            1: 7,
            2: 8,
            3: 9,
            4: 10,
            5: 11,
            6: 0,
            7: 1,
            8: 2,
            9: 3,
            10: 4,
            11: 5,
            # home / opponent home
            13: 15,
            14: 15,
            15: 13,
            # flags
            18: 19,
            19: 18,
        }

    def normalize_state(self, state_matrix: np.ndarray, team_name: str) -> np.ndarray:
        """
        Args:
            state_matrix: (H, W) integer ID matrix (typically 20x20).
            team_name: "L" or "R" (which side this matrix is from).
        """
        if team_name not in ("L", "R"):
            raise ValueError(f"Invalid team_name: {team_name!r}")

        if team_name == "L":
            return state_matrix

        mirrored = np.flip(state_matrix, axis=1)
        if not self.swap_ids:
            return mirrored

        new_matrix = mirrored.copy()
        for old_id, new_id in self._id_swap_map.items():
            new_matrix[mirrored == old_id] = new_id
        return new_matrix

    def denormalize_actions(
        self, actions: Dict[str, Optional[str]], team_name: str
    ) -> Dict[str, Optional[str]]:
        """
        Map actions from normalized coordinates back to the real environment coordinates.

        For team "R", horizontal mirroring flips left/right; up/down is unchanged.
        """
        if team_name not in ("L", "R"):
            raise ValueError(f"Invalid team_name: {team_name!r}")

        if team_name == "L":
            return actions

        action_map = {
            "left": "right",
            "right": "left",
            "up": "up",
            "down": "down",
            None: None,
            "": None,
        }
        return {name: action_map.get(act, None) for name, act in actions.items()}

