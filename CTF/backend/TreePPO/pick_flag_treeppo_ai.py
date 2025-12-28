from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from model import TreePPOPolicy  # noqa: E402
from state_encoder import TeamHistoryStateEncoder  # noqa: E402


CONFIG_PATH = Path(__file__).resolve().with_name("config.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER_INF = logging.getLogger("TreePPO.infer")


ACTION_INDEX_TO_MOVE = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: None,  # stay
}


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


class TreePPOBackend:
    def __init__(self, config: dict) -> None:
        self.device = select_device()
        model_cfg = config.get("model", {})

        self.encoder = TeamHistoryStateEncoder(
            width=20,
            height=20,
            max_players=3,
            history_len=int(model_cfg.get("history_len", 3)),
            normalize_side=bool(model_cfg.get("normalize_side", True)),
        )

        in_channels = int(model_cfg.get("history_len", 3)) * 12
        self.model = TreePPOPolicy(
            in_channels=in_channels,
            feature_dim=int(model_cfg.get("feature_dim", 128)),
            depth=int(model_cfg.get("depth", 10)),
        ).to(self.device)
        self._load_checkpoint(config.get("checkpoint_path", "checkpoints/latest.pt"))
        self.model.eval()
        LOGGER_INF.info(
            "Inference ready (device=%s depth=%s history=%s)",
            self.device,
            model_cfg.get("depth"),
            model_cfg.get("history_len"),
        )

    def _load_checkpoint(self, rel_path: str) -> None:
        ckpt_path = Path(__file__).resolve().parent / rel_path
        if not ckpt_path.exists():
            print(f"[TreePPO] checkpoint not found: {ckpt_path} (using random weights)")
            return
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state.get("model", state))
        print(f"[TreePPO] loaded checkpoint: {ckpt_path}")

    def start_game(self, req: dict) -> None:
        self.encoder.start_game(req)
        LOGGER_INF.info(
            "start_game team=%s mirror=%s",
            req.get("myteamName"),
            self.encoder.mirror_actions,
        )

    def plan_next_actions(self, req: dict) -> dict[str, str]:
        obs_batch, my_players, _active = self.encoder.encode_team(req)
        if obs_batch.size == 0:
            return {}
        LOGGER_INF.debug("plan_next_actions players=%d mirror=%s", len(my_players), self.encoder.mirror_actions)

        obs_t = torch.from_numpy(obs_batch).to(self.device)
        with torch.no_grad():
            logits, _values = self.model(obs_t)
        actions = torch.argmax(logits, dim=-1).cpu().tolist()

        moves: dict[str, str] = {}
        for idx, player in enumerate(my_players):
            if bool(player.get("inPrison")):
                continue
            act = int(actions[idx]) if idx < len(actions) else 4
            act = self.encoder.denormalize_action(act)
            move = ACTION_INDEX_TO_MOVE.get(act)
            if move:
                moves[str(player.get("name", ""))] = move
        return moves

    def game_over(self, _req: dict) -> None:
        print("Game Over!")


BACKEND = TreePPOBackend(load_config(CONFIG_PATH))


def start_game(req: dict) -> None:
    BACKEND.start_game(req)


def plan_next_actions(req: dict) -> dict[str, str]:
    return BACKEND.plan_next_actions(req)


def game_over(req: dict) -> None:
    BACKEND.game_over(req)


async def main() -> None:
    from lib.game_engine import run_game_server

    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <port>")
        raise SystemExit(1)

    port = int(sys.argv[1])
    print(f"AI backend running on port {port} ...")
    try:
        await run_game_server(port, start_game, plan_next_actions, game_over)
    except Exception as exc:
        print(f"Server stopped: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
