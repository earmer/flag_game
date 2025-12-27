import asyncio
import json
import os
import sys
from pathlib import Path

import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from model import PPOTransformerPolicy
from state_encoder import StateEncoder, MAX_PLAYERS


CONFIG_PATH = Path(__file__).resolve().with_name("config.json")

ACTION_INDEX_TO_MOVE = {
    0: None,
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}


def load_config(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


class PPOBackend:
    def __init__(self, config):
        env_cfg = config.get("env", {})
        self.width = env_cfg.get("width", 20)
        self.height = env_cfg.get("height", 20)
        self.max_players = env_cfg.get("num_players", MAX_PLAYERS)
        self.vector_dim = config.get("vector_dim", 64)
        t_cfg = config.get("transformer", {})

        self.device = torch.device(os.environ.get("CTF_PPO_DEVICE", "cpu"))
        self.encoder = StateEncoder(width=self.width, height=self.height, max_players=self.max_players)
        self.model = PPOTransformerPolicy(
            width=self.width,
            height=self.height,
            max_players=self.max_players,
            vector_dim=self.vector_dim,
            num_layers=t_cfg.get("num_layers", 4),
            num_heads=t_cfg.get("num_heads", 4),
            dropout=t_cfg.get("dropout", 0.1),
            mlp_dim=t_cfg.get("mlp_dim", 128),
        ).to(self.device)
        self._load_checkpoint(config.get("checkpoint_path", "checkpoints/latest.pt"))
        self.model.eval()

    def _load_checkpoint(self, rel_path):
        ckpt_path = Path(__file__).resolve().parent / rel_path
        if not ckpt_path.exists():
            print(f"[PPO] checkpoint not found: {ckpt_path} (using random weights)")
            return
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        print(f"[PPO] loaded checkpoint: {ckpt_path}")

    def start_game(self, req):
        self.encoder.start_game(req)

    def plan_next_actions(self, req):
        grid_ids, player_features = self.encoder.encode(req)
        if grid_ids is None:
            return {}

        grid_tensor = torch.from_numpy(grid_ids).unsqueeze(0).to(self.device)
        player_tensor = torch.from_numpy(player_features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(grid_tensor, player_tensor)

        logits = logits.squeeze(0)
        actions = torch.argmax(logits, dim=-1).cpu().tolist()

        my_players = sorted(req.get("myteamPlayer", []), key=lambda p: p.get("name", ""))
        moves = {}
        for idx, player in enumerate(my_players[: self.max_players]):
            if player.get("inPrison"):
                continue
            action_idx = actions[idx]
            move = ACTION_INDEX_TO_MOVE.get(action_idx)
            if move:
                moves[player["name"]] = move
        return moves

    def game_over(self, _req):
        print("Game Over!")


BACKEND = PPOBackend(load_config(CONFIG_PATH))


def start_game(req):
    BACKEND.start_game(req)


def plan_next_actions(req):
    return BACKEND.plan_next_actions(req)


def game_over(req):
    BACKEND.game_over(req)


async def main():
    from lib.game_engine import run_game_server

    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <port>")
        print(f"Example: python3 {sys.argv[0]} 8080")
        sys.exit(1)

    port = int(sys.argv[1])
    print(f"AI backend running on port {port} ...")
    try:
        await run_game_server(port, start_game, plan_next_actions, game_over)
    except Exception as exc:
        print(f"Server stopped: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
