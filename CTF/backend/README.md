# Backend AIs

Existing AI entrypoints live in `CTF/backend/` (e.g. `pick_flag_ai.py`, `pick_flag_elite_ai.py`, `pick_flag_potential_ai.py`).

## Decision Tree (Naive ML)

This repo includes a tiny, dependency-free decision tree classifier and a simple training script that labels actions with 1-step rollouts in `PBT/mock_env_vnew.py`.

- Train a model (writes JSON):
  - `python3 CTF/backend/train_tree_from_rollouts.py --out CTF/backend/tree_models/tree_l.json`
- Run the tree-based backend:
  - `python3 CTF/backend/pick_flag_tree_ai.py 8081 --model CTF/backend/tree_models/tree_l.json`

Notes:
- The model predicts actions in a “normalized” coordinate system (your home side always treated as left); `pick_flag_tree_ai.py` converts them back at runtime.
- This is intentionally naive: it’s a quick baseline to iterate on without installing `scikit-learn`.
- Model capacity: `--max-depth 10` allows up to 1024 leaves (data permitting).

### Staged Training (S1–S4)

`train_tree_from_rollouts.py` supports a 4-stage curriculum:
- S1: imitate beginner AI (beginner vs beginner) → optional dataset `--s1-dataset`
- S2: imitate elite AI (elite vs beginner) → optional dataset `--s2-dataset`
- S3: adversarial improvement vs elite (1-step rollout labels)
- S4: self-play improvement (1-step rollout labels)

Example (writes S1/S2 datasets + periodic checkpoints):
- `python3 CTF/backend/train_tree_from_rollouts.py --out CTF/backend/tree_models/tree_l.json --s1-dataset CTF/backend/tree_models/s1.jsonl --s2-dataset CTF/backend/tree_models/s2.jsonl --checkpoint-dir CTF/backend/tree_models/ckpt --s1-episodes 20 --s2-episodes 20 --s3-episodes 40 --s4-episodes 40`
