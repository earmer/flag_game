from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from PBT.mock_env_vnew import MockEnvVNew
from lib.tiny_decision_tree import TinyDecisionTreeClassifier
from lib.tree_features import Geometry, allowed_actions, extract_player_features
from pick_flag_elite_ai import EliteCTFAI


ACTIONS = ["", "up", "down", "left", "right"]

POLICY_BEGINNER = "beginner"
POLICY_ELITE = "elite"
POLICY_MODEL = "model"


def _pos(entity: Mapping[str, Any]) -> Tuple[int, int]:
    return (int(entity.get("posX", 0)), int(entity.get("posY", 0)))


def _bfs_next_step(
    start: Tuple[int, int],
    goals: List[Tuple[int, int]],
    *,
    width: int,
    height: int,
    blocked: set[Tuple[int, int]],
) -> str:
    if not goals:
        return ""
    if start in goals:
        return ""

    from collections import deque

    queue = deque([start])
    prev: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
    seen = {start}

    while queue:
        x, y = queue.popleft()
        for action, dx, dy in (("up", 0, -1), ("down", 0, 1), ("left", -1, 0), ("right", 1, 0)):
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if nxt in blocked:
                continue
            if nxt in seen:
                continue
            prev[nxt] = ((x, y), action)
            if nxt in goals:
                cur = nxt
                first_action = action
                while cur != start:
                    parent, a = prev[cur]
                    first_action = a
                    cur = parent
                return first_action
            seen.add(nxt)
            queue.append(nxt)

    return ""


def _beginner_actions_norm(team_status: Mapping[str, Any], geometry: Geometry) -> Dict[str, str]:
    players = list(team_status.get("myteamPlayer") or [])
    enemy_flags = [f for f in (team_status.get("opponentFlag") or []) if bool(f.get("canPickup"))]

    enemy_flag_positions = [geometry.normalize_pos(_pos(f)) for f in enemy_flags]
    my_target_positions = [geometry.normalize_pos(t) for t in geometry.my_targets]

    actions: Dict[str, str] = {}
    blocked_norm = {geometry.normalize_pos(pos) for pos in geometry.blocked}
    for p in players:
        if bool(p.get("inPrison")):
            continue
        npos = geometry.normalize_pos(_pos(p))
        goals = my_target_positions if bool(p.get("hasFlag")) else enemy_flag_positions
        action = _bfs_next_step(
            npos,
            list(goals),
            width=geometry.width,
            height=geometry.height,
            blocked=blocked_norm,
        )
        if action:
            actions[str(p["name"])] = action
    return actions


def _elite_actions_norm(team_status: Mapping[str, Any], geometry: Geometry, elite: EliteCTFAI) -> Dict[str, str]:
    actions_world = elite.plan_next_actions(team_status) or {}
    actions_norm: Dict[str, str] = {}
    for name, action_world in actions_world.items():
        act_norm = geometry.normalize_action(str(action_world))
        if act_norm:
            actions_norm[str(name)] = act_norm
    return actions_norm


def _model_actions_norm(team_status: Mapping[str, Any], geometry: Geometry, model: TinyDecisionTreeClassifier) -> Dict[str, str]:
    actions: Dict[str, str] = {}
    for p in list(team_status.get("myteamPlayer") or []):
        feats = extract_player_features(team_status, geometry, p)
        allowed = allowed_actions(team_status, geometry, p)
        proba = model.predict_proba_one(feats)
        chosen = ""
        if proba:
            for act, _prob in sorted(proba.items(), key=lambda item: item[1], reverse=True):
                if act in allowed:
                    chosen = act
                    break
        else:
            pred = model.predict_one(feats, default="")
            if pred in allowed:
                chosen = pred
        if chosen:
            actions[str(p["name"])] = chosen
    return actions


def _policy_actions_norm(
    team_status: Mapping[str, Any],
    geometry: Geometry,
    *,
    policy: str,
    model: Optional[TinyDecisionTreeClassifier],
    elite: Optional[EliteCTFAI],
) -> Dict[str, str]:
    if policy == POLICY_BEGINNER:
        return _beginner_actions_norm(team_status, geometry)
    if policy == POLICY_ELITE:
        if elite is None:
            raise ValueError("policy=elite requires an EliteCTFAI instance")
        return _elite_actions_norm(team_status, geometry, elite)
    if policy == POLICY_MODEL:
        if model is None:
            raise ValueError("policy=model requires a trained/loaded model")
        return _model_actions_norm(team_status, geometry, model)
    raise ValueError(f"Unknown policy: {policy!r}")


def _reward_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    def score_diff(st: Mapping[str, Any]) -> int:
        return int(st.get("myteamScore", 0)) - int(st.get("opponentScore", 0))

    def carriers_diff(st: Mapping[str, Any]) -> int:
        my_carriers = sum(
            1 for p in (st.get("myteamPlayer") or []) if bool(p.get("hasFlag")) and not bool(p.get("inPrison"))
        )
        opp_carriers = sum(
            1 for p in (st.get("opponentPlayer") or []) if bool(p.get("hasFlag")) and not bool(p.get("inPrison"))
        )
        return my_carriers - opp_carriers

    def prisoners_diff(st: Mapping[str, Any]) -> int:
        my_prisoners = sum(1 for p in (st.get("myteamPlayer") or []) if bool(p.get("inPrison")))
        opp_prisoners = sum(1 for p in (st.get("opponentPlayer") or []) if bool(p.get("inPrison")))
        return my_prisoners - opp_prisoners

    delta_score = score_diff(after) - score_diff(before)
    delta_carriers = carriers_diff(after) - carriers_diff(before)
    delta_prisoners = prisoners_diff(after) - prisoners_diff(before)

    return (10.0 * delta_score) + (0.5 * delta_carriers) - (0.3 * delta_prisoners)


def _load_dataset(path: str) -> Tuple[List[Dict[str, float]], List[str]]:
    X: List[Dict[str, float]] = []
    y: List[str] = []
    p = Path(path)
    if not p.exists():
        return X, y
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        X.append({str(k): float(v) for k, v in (item.get("x") or {}).items()})
        y.append(str(item.get("y", "")))
    return X, y


def _append_sample(dataset_path: str, x: Dict[str, float], y: str, *, stage: str) -> None:
    if not dataset_path:
        return
    Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "x": x, "y": y}, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny decision tree policy with a 4-stage curriculum:\n"
            "  S1: imitate beginner AI (beginner vs beginner)\n"
            "  S2: imitate elite AI (elite vs beginner)\n"
            "  S3: adversarial improvement vs elite (1-step rollout labels)\n"
            "  S4: self-play improvement (1-step rollout labels)\n\n"
            "Note: if --team=R, features/actions are normalized by flipping the board via Geometry."
        )
    )
    parser.add_argument("--out", required=True, help="Output model JSON path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--team", choices=["L", "R"], default="L")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--sample-rate", type=float, default=0.35, help="Chance to label a player on a tick")
    parser.add_argument("--epsilon", type=float, default=0.15, help="Chance to replace a team action with random")

    parser.add_argument("--s1-episodes", type=int, default=10, help="S1: beginner vs beginner imitation")
    parser.add_argument("--s2-episodes", type=int, default=10, help="S2: elite vs beginner imitation")
    parser.add_argument("--s3-episodes", type=int, default=10, help="S3: adversarial vs elite (rollout labels)")
    parser.add_argument("--s4-episodes", type=int, default=10, help="S4: self-play (rollout labels)")

    parser.add_argument("--s1-dataset", default="", help="Optional JSONL dataset for S1 (beginner vs beginner)")
    parser.add_argument("--s2-dataset", default="", help="Optional JSONL dataset for S2 (elite vs beginner)")
    parser.add_argument("--resume-model", default="", help="Optional model JSON to start from (useful for S3/S4)")

    parser.add_argument("--checkpoint-dir", default="", help="Write intermediate model checkpoints here")
    parser.add_argument("--checkpoint-every-episodes", type=int, default=5)

    # max_depth=10 => up to 1024 leaves (if data supports it).
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=30)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    opponent = "R" if args.team == "L" else "L"

    X: List[Dict[str, float]] = []
    y: List[str] = []

    if args.s1_dataset:
        X1, y1 = _load_dataset(args.s1_dataset)
        X.extend(X1)
        y.extend(y1)
        if X1:
            print(f"Loaded S1 dataset: {args.s1_dataset} (samples={len(X1)})")

    if args.s2_dataset:
        X2, y2 = _load_dataset(args.s2_dataset)
        X.extend(X2)
        y.extend(y2)
        if X2:
            print(f"Loaded S2 dataset: {args.s2_dataset} (samples={len(X2)})")

    model: Optional[TinyDecisionTreeClassifier] = None
    if args.resume_model:
        with open(args.resume_model, "r", encoding="utf-8") as handle:
            model = TinyDecisionTreeClassifier.from_dict(json.load(handle))
        print(f"Resumed model: {args.resume_model}")

    def train_and_checkpoint(tag: str) -> TinyDecisionTreeClassifier:
        nonlocal model
        if not X:
            raise SystemExit("No samples collected/loaded")

        indices = list(range(len(X)))
        rng.shuffle(indices)
        split = int(0.9 * len(indices))
        train_idx = indices[:split]
        val_idx = indices[split:]

        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = TinyDecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split,
        ).fit(X_train, y_train)

        correct = 0
        for row, label in zip(X_val, y_val):
            pred = model.predict_one(row, default="")
            if pred == label:
                correct += 1
        acc = correct / max(1, len(y_val))
        print(f"{tag}: val accuracy: {acc:.3f} ({correct}/{len(y_val)}) samples={len(X)}")

        if args.checkpoint_dir:
            ckpt_dir = Path(args.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"tree_{args.team.lower()}_{tag}.json"
            payload = model.to_dict()
            payload["team"] = args.team
            payload["labels"] = ACTIONS
            ckpt_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(f"wrote checkpoint: {ckpt_path}")

        return model

    def _noisy_team_actions(actions: Dict[str, str], team_status: Mapping[str, Any]) -> Dict[str, str]:
        noisy = dict(actions)
        for p in list(team_status.get("myteamPlayer") or []):
            if bool(p.get("inPrison")):
                continue
            if rng.random() < args.epsilon:
                noisy[str(p["name"])] = rng.choice(ACTIONS)
        return noisy

    def collect_imitation(
        episodes: int,
        *,
        stage: str,
        team_policy: str,
        opp_policy: str,
        dataset_path: str,
    ) -> None:
        nonlocal model
        if episodes <= 0:
            return

        offsets = {"s1": 0, "s2": 20000}
        for ep in range(episodes):
            env = MockEnvVNew(seed=args.seed + offsets.get(stage, 0) + ep)
            init_team = env.get_init_payload(args.team)
            init_opp = env.get_init_payload(opponent)
            geom_team = Geometry.from_init(init_team)
            geom_opp = Geometry.from_init(init_opp)

            elite_team: Optional[EliteCTFAI] = None
            elite_opp: Optional[EliteCTFAI] = None
            if team_policy == POLICY_ELITE:
                elite_team = EliteCTFAI(show_gap_in_msec=0.0)
                elite_team.start_game(init_team)
            if opp_policy == POLICY_ELITE:
                elite_opp = EliteCTFAI(show_gap_in_msec=0.0)
                elite_opp.start_game(init_opp)

            state = env.get_full_state()
            for _t in range(args.max_steps):
                team_status = state[args.team]
                opp_status = state[opponent]

                team_actions = _policy_actions_norm(
                    team_status, geom_team, policy=team_policy, model=model, elite=elite_team
                )
                opp_actions = _policy_actions_norm(
                    opp_status, geom_opp, policy=opp_policy, model=model, elite=elite_opp
                )

                for p in list(team_status.get("myteamPlayer") or []):
                    if rng.random() > args.sample_rate:
                        continue
                    feats = extract_player_features(team_status, geom_team, p)
                    candidates = allowed_actions(team_status, geom_team, p)
                    label = team_actions.get(str(p["name"]), "")
                    if label not in candidates:
                        continue
                    X.append(feats)
                    y.append(label)
                    _append_sample(dataset_path, feats, label, stage=stage)

                step_team_norm = _noisy_team_actions(team_actions, team_status)
                step_opp_norm = _noisy_team_actions(opp_actions, opp_status) if stage == "s1" else dict(opp_actions)

                step_team_world = {name: geom_team.denormalize_action(act) for name, act in step_team_norm.items()}
                step_opp_world = {name: geom_opp.denormalize_action(act) for name, act in step_opp_norm.items()}

                if args.team == "L":
                    state, done, _info = env.step(actions_l=step_team_world, actions_r=step_opp_world)
                else:
                    state, done, _info = env.step(actions_l=step_opp_world, actions_r=step_team_world)
                if done:
                    break

            print(f"{stage} episode {ep + 1}/{episodes}: samples={len(X)}")
            if args.checkpoint_dir and args.checkpoint_every_episodes > 0 and (ep + 1) % args.checkpoint_every_episodes == 0:
                train_and_checkpoint(f"{stage}_ep{ep+1}")

    def _ensure_model(tag: str) -> TinyDecisionTreeClassifier:
        nonlocal model
        if model is not None:
            return model
        if X:
            return train_and_checkpoint(tag)
        raise SystemExit("No model and no samples; run S1/S2 or provide --resume-model")

    def collect_rollout_improve(
        episodes: int,
        *,
        stage: str,
        opp_policy: str,
    ) -> None:
        nonlocal model
        if episodes <= 0:
            return

        _ensure_model(f"{stage}_bootstrap")

        offsets = {"s3": 40000, "s4": 60000}
        for ep in range(episodes):
            env = MockEnvVNew(seed=args.seed + offsets.get(stage, 0) + ep)
            init_team = env.get_init_payload(args.team)
            init_opp = env.get_init_payload(opponent)
            geom_team = Geometry.from_init(init_team)
            geom_opp = Geometry.from_init(init_opp)

            elite_opp: Optional[EliteCTFAI] = None
            if opp_policy == POLICY_ELITE:
                elite_opp = EliteCTFAI(show_gap_in_msec=0.0)
                elite_opp.start_game(init_opp)

            state = env.get_full_state()
            for _t in range(args.max_steps):
                team_status = state[args.team]
                opp_status = state[opponent]

                base_team = _policy_actions_norm(
                    team_status, geom_team, policy=POLICY_MODEL, model=model, elite=None
                )
                base_opp = _policy_actions_norm(
                    opp_status, geom_opp, policy=opp_policy, model=model, elite=elite_opp
                )

                for p in list(team_status.get("myteamPlayer") or []):
                    if rng.random() > args.sample_rate:
                        continue

                    feats = extract_player_features(team_status, geom_team, p)
                    candidates = allowed_actions(team_status, geom_team, p)
                    if not candidates:
                        continue

                    best_action = ""
                    best_reward = None
                    for cand in candidates:
                        env_c = copy.deepcopy(env)

                        actions_team_norm = dict(base_team)
                        actions_team_norm[str(p["name"])] = cand
                        actions_team_world = {
                            name: geom_team.denormalize_action(act) for name, act in actions_team_norm.items()
                        }

                        actions_opp_world = {name: geom_opp.denormalize_action(act) for name, act in base_opp.items()}

                        if args.team == "L":
                            next_state, _done, _info = env_c.step(actions_l=actions_team_world, actions_r=actions_opp_world)
                        else:
                            next_state, _done, _info = env_c.step(actions_l=actions_opp_world, actions_r=actions_team_world)

                        reward = _reward_delta(team_status, next_state[args.team])
                        if best_reward is None or reward > best_reward:
                            best_reward = reward
                            best_action = cand

                    X.append(feats)
                    y.append(best_action)

                step_team_norm = _noisy_team_actions(base_team, team_status)
                if stage == "s4":
                    step_opp_norm = _noisy_team_actions(base_opp, opp_status)
                else:
                    step_opp_norm = dict(base_opp)

                step_team_world = {name: geom_team.denormalize_action(act) for name, act in step_team_norm.items()}
                step_opp_world = {name: geom_opp.denormalize_action(act) for name, act in step_opp_norm.items()}

                if args.team == "L":
                    state, done, _info = env.step(actions_l=step_team_world, actions_r=step_opp_world)
                else:
                    state, done, _info = env.step(actions_l=step_opp_world, actions_r=step_team_world)
                if done:
                    break

            print(f"{stage} episode {ep + 1}/{episodes}: samples={len(X)}")
            train_and_checkpoint(f"{stage}_ep{ep+1}")

    collect_imitation(
        args.s1_episodes,
        stage="s1",
        team_policy=POLICY_BEGINNER,
        opp_policy=POLICY_BEGINNER,
        dataset_path=args.s1_dataset,
    )
    if args.s1_episodes > 0:
        train_and_checkpoint("s1_final")

    collect_imitation(
        args.s2_episodes,
        stage="s2",
        team_policy=POLICY_ELITE,
        opp_policy=POLICY_BEGINNER,
        dataset_path=args.s2_dataset,
    )
    if args.s2_episodes > 0:
        train_and_checkpoint("s2_final")

    collect_rollout_improve(args.s3_episodes, stage="s3", opp_policy=POLICY_ELITE)
    if args.s3_episodes > 0:
        train_and_checkpoint("s3_final")

    collect_rollout_improve(args.s4_episodes, stage="s4", opp_policy=POLICY_MODEL)
    if args.s4_episodes > 0:
        train_and_checkpoint("s4_final")

    model = _ensure_model("final_bootstrap")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.to_dict()
    payload["team"] = args.team
    payload["labels"] = ACTIONS
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
