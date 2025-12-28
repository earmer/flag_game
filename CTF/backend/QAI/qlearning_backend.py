from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from lib.game_engine import run_game_server
from lib.tree_features import Geometry

from qlearning_core import QLearningCore


class QLearningBackend:
    def __init__(self, *, persistence_path: Optional[Path], save_every: int) -> None:
        self.geometry: Optional[Geometry] = None
        self.core = QLearningCore(persistence_path=persistence_path, save_every=save_every)
        self.last_time: float = -1.0

    def start_game(self, req: Dict[str, Any]) -> None:
        self.geometry = Geometry.from_init(req)
        self.last_time = -1.0
        self.core.reset()
        side = "Left" if self.geometry.my_side_is_left else "Right"
        print(f"Q-Learning AI started. Side: {side}")

    def game_over(self, _req: Dict[str, Any]) -> None:
        print("Game Over!")
        self.core.save()

    def plan_next_actions(self, req: Dict[str, Any]) -> Dict[str, str]:
        if self.geometry is None:
            return {}
        now = float(req.get("time", 0.0))
        if now < self.last_time:
            return {}
        self.last_time = now

        actions: Dict[str, str] = {}
        for player in list(req.get("myteamPlayer") or []):
            action = self.core.pick_action(req, self.geometry, player)
            if action:
                actions[str(player.get("name", ""))] = action
        self.core.decay_epsilon()
        return actions


BACKEND: Optional[QLearningBackend] = None


def _get_backend() -> QLearningBackend:
    if BACKEND is None:
        raise RuntimeError("QLearningBackend not initialized")
    return BACKEND


def start_game(req: Dict[str, Any]) -> None:
    _get_backend().start_game(req)


def plan_next_actions(req: Dict[str, Any]) -> Dict[str, str]:
    return _get_backend().plan_next_actions(req)


def game_over(req: Dict[str, Any]) -> None:
    _get_backend().game_over(req)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parent / "qlearning_model.pkl",
        help="Path where the Q-table snapshot is read/written.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="How many updates between auto-saves.",
    )
    parser.add_argument(
        "--clear-model",
        action="store_true",
        help="Discard any persisted Q-table before starting.",
    )
    args = parser.parse_args()

    if args.save_every <= 0:
        raise SystemExit("--save-every must be greater than zero")

    backend = QLearningBackend(persistence_path=args.model_path, save_every=args.save_every)
    if args.clear_model:
        backend.core.reset(clear_table=True)

    global BACKEND
    BACKEND = backend

    port = int(args.port)
    print(f"AI backend running on port {port} ...")

    try:
        await run_game_server(port, start_game, plan_next_actions, game_over)
    except Exception as exc:
        print(f"Server stopped: {exc}")
        raise SystemExit(1)
    finally:
        backend.core.save()


if __name__ == "__main__":
    asyncio.run(main())
