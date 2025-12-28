# Q-Learning Backend (Python)

Lightweight Q-Learning backend adapted to the current capture-the-flag game. It uses the same WebSocket API as
the other Python AIs (init/status/finished payloads) and runs directly in the backend server, so no Node/stdio
bridge is needed.

This implementation uses a shared Q-table across all of your players, with a small feature vector derived from
the current game state (player position, nearest flags, nearest opponents, prison counts, blocked moves, etc.).
It applies simple reward shaping based on flag pickup/return, imprisonment, and distance deltas.

## Running it locally

1. Start the backend:

   `python3 CTF/backend/QAI/qlearning_backend.py 8080`

   Add `--model-path <path>` to use a custom checkpoint, `--save-every <n>` to change the autosave cadence, and include `--clear-model` to discard any previous Q-table.

2. Point the frontend at it by editing `CTF/frontend/game_config.json`:

   Set your team entry (e.g. `user1-11`) to `ws://localhost:8080`.

3. Run the frontend:

   `cd CTF/frontend && python3 -m http.server 8000`

   Then open `http://localhost:8000/index.html`.

## Persistence

- The default snapshot is stored at `CTF/backend/QAI/qlearning_model.pkl`. Every `--save-every` updates the dictionary is persisted, and `game_over`/shutdown also trigger a save.
- Remove this file or run the backend with `--clear-model` to restart training from scratch; otherwise each launch continues from the previous run automatically.

## Files

- `CTF/backend/QAI/qlearning_backend.py`: WebSocket entrypoint that plugs into the game server.
- `CTF/backend/QAI/qlearning_core.py`: Q-learning core, feature discretization, and reward shaping.

## Legacy

The old C++/Node version is still in `CTF/backend/QAI/ai/` and `CTF/backend/QAI/controller/` for reference, but
the current project uses the Python backend above.

  
