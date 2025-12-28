from __future__ import annotations

import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from lib.tree_features import Geometry, allowed_actions, extract_player_features

Action = str


@dataclass
class RewardState:
    has_flag: bool
    in_prison: bool
    dist_enemy_flag: float
    dist_home_target: float


class QLearningCore:
    def __init__(
        self,
        *,
        alpha: float = 0.4,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        persistence_path: Optional[Path] = None,
        save_every: int = 500,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.save_every = max(1, save_every)
        self.updates_since_save = 0

        self.q_table: Dict[Tuple[Any, ...], float] = {}
        self.last_state_action: Dict[str, Tuple[Tuple[int, ...], Action]] = {}
        self.last_reward_state: Dict[str, RewardState] = {}

        self._load_q_table()

    def reset(self, *, clear_table: bool = False) -> None:
        if clear_table:
            self.q_table.clear()
            self.clear_persistence()
        self.last_state_action.clear()
        self.last_reward_state.clear()
        self.updates_since_save = 0
        self.epsilon = 1.0

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def pick_action(self, req: Mapping[str, Any], geometry: Geometry, player: Mapping[str, Any]) -> Action:
        features = extract_player_features(req, geometry, player)
        state = self._state_from_features(features)
        actions_norm = self._filtered_actions(req, geometry, player)

        reward = self._compute_reward(player.get("name", ""), features)
        self._update_q(player.get("name", ""), reward, state, actions_norm)

        chosen_norm = self._choose_action(state, actions_norm)
        self.last_state_action[str(player.get("name", ""))] = (state, chosen_norm)
        return geometry.denormalize_action(chosen_norm)

    def _filtered_actions(self, req: Mapping[str, Any], geometry: Geometry, player: Mapping[str, Any]) -> List[Action]:
        actions = allowed_actions(req, geometry, player)
        if len(actions) > 1:
            actions = [a for a in actions if a]
        return actions or [""]

    def _choose_action(self, state: Tuple[int, ...], actions: List[Action]) -> Action:
        if not actions:
            return ""
        if random.random() < self.epsilon:
            return random.choice(actions)
        best_action = actions[0]
        best_value = self._q_value(state, best_action)
        for act in actions[1:]:
            val = self._q_value(state, act)
            if val > best_value:
                best_value = val
                best_action = act
        return best_action

    def _update_q(
        self,
        player_name: str,
        reward: float,
        next_state: Tuple[int, ...],
        next_actions: List[Action],
    ) -> None:
        last = self.last_state_action.get(str(player_name))
        if not last:
            return
        last_state, last_action = last
        max_next = 0.0
        if next_actions:
            max_next = max(self._q_value(next_state, act) for act in next_actions)
        key = last_state + (last_action,)
        old = self.q_table.get(key, 0.0)
        self.q_table[key] = old + self.alpha * (reward + self.gamma * max_next - old)
        self._maybe_save()

    def _q_value(self, state: Tuple[int, ...], action: Action) -> float:
        return self.q_table.get(state + (action,), 0.0)

    def _compute_reward(self, player_name: str, features: Dict[str, float]) -> float:
        reward = -0.05
        has_flag = bool(features.get("has_flag", 0.0))
        in_prison = bool(features.get("in_prison", 0.0))
        dist_enemy = float(features.get("dist_enemy_flag", 999.0))
        dist_home = float(features.get("dist_home_target", 999.0))

        last = self.last_reward_state.get(str(player_name))
        if last is not None:
            if not last.has_flag and has_flag:
                reward += 100.0
            if not last.in_prison and in_prison:
                reward -= 100.0
            if last.has_flag and not has_flag and last.dist_home_target <= 1.0:
                reward += 50.0

            if has_flag:
                reward += max(0.0, last.dist_home_target - dist_home) * 1.0
                reward -= max(0.0, dist_home - last.dist_home_target) * 0.2
            else:
                reward += max(0.0, last.dist_enemy_flag - dist_enemy) * 0.5
                reward -= max(0.0, dist_enemy - last.dist_enemy_flag) * 0.1

        self.last_reward_state[str(player_name)] = RewardState(
            has_flag=has_flag,
            in_prison=in_prison,
            dist_enemy_flag=dist_enemy,
            dist_home_target=dist_home,
        )
        return reward

    @staticmethod
    def _bin_distance(value: float) -> int:
        if value <= 0:
            return 0
        if value <= 1:
            return 1
        if value <= 2:
            return 2
        if value <= 3:
            return 3
        if value <= 4:
            return 4
        if value <= 6:
            return 5
        if value <= 8:
            return 6
        if value <= 11:
            return 7
        if value <= 15:
            return 8
        return 9

    def _state_from_features(self, features: Dict[str, float]) -> Tuple[int, ...]:
        def _clip(value: float, cap: int = 3) -> int:
            return int(min(cap, max(0, int(value))))

        state = (
            int(features.get("x", 0.0)),
            int(features.get("y", 0.0)),
            int(features.get("is_safe", 0.0)),
            int(features.get("has_flag", 0.0)),
            int(features.get("in_prison", 0.0)),
            _clip(features.get("num_my_prisoners", 0.0)),
            _clip(features.get("num_opp_prisoners", 0.0)),
            _clip(features.get("num_intruders_with_flag", 0.0)),
            self._bin_distance(features.get("dist_enemy_flag", 999.0)),
            self._bin_distance(features.get("dist_home_target", 999.0)),
            self._bin_distance(features.get("dist_home_prison", 999.0)),
            self._bin_distance(features.get("dist_nearest_opp", 999.0)),
            self._bin_distance(features.get("dist_nearest_opp_carrier", 999.0)),
            int(features.get("blocked_up", 0.0)),
            int(features.get("blocked_down", 0.0)),
            int(features.get("blocked_left", 0.0)),
            int(features.get("blocked_right", 0.0)),
        )
        return tuple(int(v) for v in state)

    def _maybe_save(self) -> None:
        if not self.persistence_path:
            return
        self.updates_since_save += 1
        if self.updates_since_save >= self.save_every:
            self.save()
            self.updates_since_save = 0

    def _load_q_table(self) -> None:
        if not self.persistence_path:
            return
        try:
            if self.persistence_path.is_file():
                with self.persistence_path.open("rb") as handle:
                    data = pickle.load(handle)
                if isinstance(data, dict):
                    self.q_table = data
                else:
                    print(f"Unexpected Q-table format in {self.persistence_path}")
        except Exception as exc:  # pragma: no cover
            print(f"Unable to load Q-table from {self.persistence_path}: {exc}")

    def save(self) -> None:
        if not self.persistence_path:
            return
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with self.persistence_path.open("wb") as handle:
                pickle.dump(self.q_table, handle)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to write Q-table to {self.persistence_path}: {exc}")

    def clear_persistence(self) -> None:
        if self.persistence_path and self.persistence_path.exists():
            try:
                self.persistence_path.unlink()
            except OSError as exc:
                print(f"Failed to remove {self.persistence_path}: {exc}")
