import argparse
import asyncio
import json
import os
from typing import Any, Dict, Optional

from lib.game_engine import run_game_server
from lib.tiny_decision_tree import TinyDecisionTreeClassifier
from lib.tree_features import Geometry, allowed_actions, extract_player_features


class TreeCTFAI:
    def __init__(self) -> None:
        self.model: Optional[TinyDecisionTreeClassifier] = None
        self.model_team: str = ""
        self.geometry: Optional[Geometry] = None
        self.last_time: float = -1.0

    def load_model(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.model = TinyDecisionTreeClassifier.from_dict(data)
        self.model_team = str(data.get("team", "") or "")
        print(f"Loaded decision tree model: {path}")

    def start_game(self, req: Dict[str, Any]) -> None:
        self.geometry = Geometry.from_init(req)
        self.last_time = -1.0
        side = "Left" if self.geometry.my_side_is_left else "Right"
        runtime_team = str(req.get("myteamName", "") or "")
        if self.model_team and runtime_team and self.model_team != runtime_team:
            print(
                f"Tree AI started. Side: {side} (runtime_team={runtime_team}, model_team={self.model_team}; "
                "OK: inputs/outputs are normalized via board mirroring)"
            )
        else:
            print(f"Tree AI started. Side: {side}")

    def game_over(self, _req: Dict[str, Any]) -> None:
        print("Game Over!")

    def plan_next_actions(self, req: Dict[str, Any]) -> Dict[str, str]:
        if self.model is None or self.geometry is None:
            return {}

        now = float(req.get("time", 0.0))
        if now < self.last_time:
            return {}
        self.last_time = now

        actions: Dict[str, str] = {}
        for p in list(req.get("myteamPlayer") or []):
            feats = extract_player_features(req, self.geometry, p)
            allowed = allowed_actions(req, self.geometry, p)
            allowed_moves = [a for a in allowed if a]

            proba = self.model.predict_proba_one(feats)
            chosen_norm = ""
            if proba:
                ranked = sorted(proba.items(), key=lambda item: item[1], reverse=True)
                for act, _prob in ranked:
                    if not act:
                        continue
                    if act in allowed_moves:
                        chosen_norm = act
                        break
                if not chosen_norm:
                    for act, _prob in ranked:
                        if act in allowed:
                            chosen_norm = act
                            break
            else:
                pred = self.model.predict_one(feats, default="")
                if pred in allowed_moves:
                    chosen_norm = pred
                elif pred in allowed:
                    chosen_norm = pred

            if not chosen_norm and allowed_moves:
                chosen_norm = allowed_moves[0]

            chosen_world = self.geometry.denormalize_action(chosen_norm)
            if chosen_world:
                actions[p["name"]] = chosen_world
        return actions


AI = TreeCTFAI()


def start_game(req: Dict[str, Any]) -> None:
    AI.start_game(req)


def plan_next_actions(req: Dict[str, Any]) -> Dict[str, str]:
    return AI.plan_next_actions(req)


def game_over(req: Dict[str, Any]) -> None:
    AI.game_over(req)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument(
        "--model",
        default=os.environ.get("CTF_TREE_MODEL", ""),
        help="Path to a decision tree model JSON (or set CTF_TREE_MODEL)",
    )
    args = parser.parse_args()

    if not args.model:
        raise SystemExit("Missing --model (or env CTF_TREE_MODEL)")

    AI.load_model(args.model)
    port = int(args.port)
    print(f"AI backend running on port {port} ...")

    try:
        await run_game_server(port, start_game, plan_next_actions, game_over)
    except Exception as exc:
        print(f"Server stopped: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
