# Repository Guidelines

## Project Structure & Module Organization

- `CTF/frontend/`: Phaser-based web client (scenes in `src/scenes/`, game objects in `src/gameObjects/`, art + maps in `assets/`).
- `CTF/backend/`: Python backends and utilities (AI entrypoints like `pick_flag_ai.py`, shared helpers in `lib/`, notebooks in `*.ipynb`).
- Docs: `README.md` (quick notes), `CTF/README.md` (game overview), `AI_Design.md` (state IDs / control flow notes).

## Build, Test, and Development Commands

- Run the frontend locally:
  - `cd CTF/frontend && python3 -m http.server 8000`
  - Open `http://localhost:8000/index.html`
- Run a Python backend (WebSocket server) on a port:
  - `cd CTF/backend && python3 pick_flag_ai.py 8081` (also: `pick_closest_flag.py`, `pick_flag_potential_ai.py`)
- Point the game at your backend:
  - Edit `CTF/frontend/game_config.json` and set `servers` entries to `ws://localhost:<port>`.
- Dependencies aren’t pinned in-repo; typical installs include `websockets`, `numpy`, and `ipython`.

## Coding Style & Naming Conventions

- Python: 4-space indentation, `snake_case` for functions/vars, keep async entrypoints compatible with `asyncio`.
- JavaScript: ES modules (`import … from …`), `PascalCase` for classes (e.g., scenes), `camelCase` for variables.
- Prefer small, focused diffs; keep assets in `CTF/frontend/assets/` (don’t inline large binaries into code).

## Testing Guidelines

- No formal automated test suite currently.
- Quick sanity checks:
  - `python3 -m compileall CTF/backend`
  - `python3 CTF/backend/get_matrix.py` (converts example game JSON into a 2D matrix)

## Commit & Pull Request Guidelines

- Commit messages commonly use short, prefixed subjects (e.g., `feat: …`, `docs: …`, `fix: …`). Keep subjects imperative and scoped.
- PRs should include: what changed, how to run/verify (ports + config), and screenshots/GIFs for gameplay/UI changes.

## Configuration & Security Notes

- `CTF/frontend/index.html` loads Phaser from a CDN by default; use the local `CTF/frontend/phaser.js` if you need offline runs.
- Avoid committing secrets, tokens, or machine-specific paths; don’t add virtualenvs (`.venv/`) or generated outputs.
