from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from lib.matrix_util import CTFMatrixConverter  # noqa: E402

from mock_env_vnew import MockEnvVNew  # noqa: E402
from state_normalizer import StateNormalizer  # noqa: E402


class PPOEnvAdapter:
    """Wrap `MockEnvVNew` with a PPO-friendly API."""

    def __init__(
        self,
        *,
        team: str = "L",
        num_flags: int = 9,
        seed: Optional[int] = None,
        swap_ids_on_normalize: bool = False,
    ) -> None:
        if team not in ("L", "R"):
            raise ValueError(f"Invalid team: {team!r}")

        self.env = MockEnvVNew(num_flags=num_flags, seed=seed)
        self.team = team
        self.opponent_team = "R" if team == "L" else "L"

        self.converter = CTFMatrixConverter(width=self.env.width, height=self.env.height)
        self.normalizer = StateNormalizer(swap_ids=swap_ids_on_normalize)

        self.num_players = self.env.num_players
        self.num_actions = 5  # [noop, up, down, left, right]

        self._action_id_to_dir = {0: None, 1: "up", 2: "down", 3: "left", 4: "right"}

    def reset(self) -> np.ndarray:
        """Reset env and return normalized observation matrix (H, W)."""
        full_state = self.env.reset()

        init_payload = self.env.get_init_payload(self.team)
        self.converter.initialize_static_map(init_payload)

        status = full_state[self.team]
        state_matrix = self.converter.convert_to_matrix(status)
        if state_matrix is None:
            raise RuntimeError("CTFMatrixConverter returned None; did you initialize_static_map()?")

        return self.normalizer.normalize_state(state_matrix, self.team)

    def step(
        self,
        actions: np.ndarray,
        opponent_actions: Dict[str, Optional[str]],
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one turn.

        Args:
            actions: (num_players,) int array with values [0..4].
            opponent_actions: dict {player_name: direction or None} for opponent team.
        """
        if actions.shape != (self.num_players,):
            raise ValueError(f"Expected actions shape ({self.num_players},), got {actions.shape}")

        my_actions: Dict[str, Optional[str]] = {}
        for i, action_id in enumerate(actions.tolist()):
            if int(action_id) not in self._action_id_to_dir:
                raise ValueError(f"Invalid action id: {action_id!r}")
            player_name = f"{self.team}{i}"
            my_actions[player_name] = self._action_id_to_dir[int(action_id)]

        my_actions = self.normalizer.denormalize_actions(my_actions, self.team)

        if self.team == "L":
            full_state, done, info = self.env.step(my_actions, opponent_actions)
        else:
            full_state, done, info = self.env.step(opponent_actions, my_actions)

        status = full_state[self.team]
        state_matrix = self.converter.convert_to_matrix(status)
        if state_matrix is None:
            raise RuntimeError("CTFMatrixConverter returned None; did you initialize_static_map()?")

        normalized = self.normalizer.normalize_state(state_matrix, self.team)
        reward = self.compute_reward(full_state, done, info)

        return normalized, reward, done, info

    def compute_reward(self, full_state: Dict[str, Any], done: bool, info: Dict[str, Any]) -> float:
        """Simple shaped reward; replace with a richer `reward.py` later."""
        status = full_state[self.team]
        my_score = float(status["myteamScore"])
        opp_score = float(status["opponentScore"])

        if done:
            return 100.0 if info.get("winner") == self.team else -100.0
        return (my_score - opp_score) * 10.0

