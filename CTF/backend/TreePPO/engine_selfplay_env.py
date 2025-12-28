from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from PBT.mock_env_vnew import MockEnvVNew


_ACTION_ID_TO_DIR = {0: None, 1: "up", 2: "down", 3: "left", 4: "right"}


def _deep_json(payload: Any) -> Any:
    """
    Force payload through JSON roundtrip to match WebSocket shapes exactly and avoid
    accidental non-JSON types leaking into training (e.g. tuples, numpy scalars).
    """
    return json.loads(json.dumps(payload))


def _build_actions(team: str, actions: list[int], *, num_players: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for idx in range(int(num_players)):
        a = int(actions[idx]) if idx < len(actions) else 0
        # Match frontend semantics: "no move" means remoteControl undefined.
        if a == 0:
            continue
        out[f"{team}{idx}"] = _ACTION_ID_TO_DIR.get(a)
    return out


class CTFGameEngineSelfPlayEnv:
    """
    Self-play env backed by the "game engine" (faithful Phaser re-sim) in
    `CTF/backend/PBT/mock_env_vnew.py`.

    Public API matches `PPO/selfplay_env.CTFFrontendRulesEnv` used by the trainers:
    - `.init_req["L"/"R"]` init payloads
    - `.reset() -> (status_L_json, status_R_json)`
    - `.step(actions_L_env, actions_R_env) -> (status_L_json, status_R_json, done)`
    """

    def __init__(self, config: dict) -> None:
        self.max_steps = int(config.get("max_steps", 400))
        self._step_count = 0

        init_path = config.get("init_path")
        self._init_payload: dict | None = None
        if init_path:
            path = Path(__file__).resolve().parent / str(init_path)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and payload.get("action") == "init":
                self._init_payload = payload

        width = int(config.get("width", 20))
        height = int(config.get("height", 20))
        num_players = int(config.get("num_players", 3))
        num_flags = int(config.get("num_flags", 9))
        use_random_flags = bool(config.get("use_random_flags", False))
        num_obstacles_1 = int(config.get("num_obstacles_1", 8))
        num_obstacles_2 = int(config.get("num_obstacles_2", 4))
        move_speed_ms = float(config.get("move_speed_ms", 300.0))
        prison_duration_ms = float(config.get("prison_duration_ms", 20000.0))
        seed = config.get("seed", None)

        self._engine = MockEnvVNew(
            width=width,
            height=height,
            num_players=num_players,
            num_flags=num_flags,
            use_random_flags=use_random_flags,
            num_obstacles_1=num_obstacles_1,
            num_obstacles_2=num_obstacles_2,
            move_speed_ms=move_speed_ms,
            prison_duration_ms=prison_duration_ms,
            seed=seed,
        )

        if self._init_payload is not None:
            # Keep static map/zones fixed for every reset.
            self._engine.load_from_init_payload(self._init_payload)

        self.init_req = {
            "L": _deep_json(self._engine.get_init_payload("L")),
            "R": _deep_json(self._engine.get_init_payload("R")),
        }

        self.reset()

    @property
    def width(self) -> int:
        return int(self._engine.width)

    @property
    def height(self) -> int:
        return int(self._engine.height)

    @property
    def num_players(self) -> int:
        return int(self._engine.num_players)

    @property
    def num_flags(self) -> int:
        return int(self._engine.num_flags)

    @property
    def time(self) -> float:
        return float(self._engine.time_ms)

    @property
    def score(self) -> Dict[str, int]:
        return dict(self._engine.scores)

    def reset(self) -> Tuple[dict, dict]:
        self._step_count = 0
        # Keep static map/zones fixed: only reset time/scores and respawn entities.
        self._engine.time_ms = 0.0
        self._engine.scores = {"L": 0, "R": 0}
        self._engine._generate_entities()  # noqa: SLF001
        state = self._engine.get_full_state()
        return _deep_json(state["L"]), _deep_json(state["R"])

    def step(self, actions_L: list[int], actions_R: list[int]) -> Tuple[dict, dict, bool]:
        self._step_count += 1

        act_L = _build_actions("L", list(actions_L), num_players=self.num_players)
        act_R = _build_actions("R", list(actions_R), num_players=self.num_players)

        state, done, info = self._engine.step(act_L, act_R)
        done = bool(done or (self._step_count >= self.max_steps))
        if done and info.get("winner") is None:
            # When max_steps ends the episode, treat score as the tiebreaker.
            if self._engine.scores["L"] > self._engine.scores["R"]:
                info["winner"] = "L"
            elif self._engine.scores["R"] > self._engine.scores["L"]:
                info["winner"] = "R"

        return _deep_json(state["L"]), _deep_json(state["R"]), done
