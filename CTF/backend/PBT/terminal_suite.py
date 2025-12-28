from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import random
import select
import signal
import subprocess
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from PBT.mock_env_vnew import MockEnvVNew  # noqa: E402


ActionMap = Dict[str, Optional[str]]


def _clear_screen() -> None:
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class Ansi:
    RESET = "\x1b[0m"
    DIM = "\x1b[2m"
    BOLD = "\x1b[1m"
    FG_BLUE = "\x1b[34m"
    FG_RED = "\x1b[31m"
    FG_YELLOW = "\x1b[33m"
    FG_CYAN = "\x1b[36m"
    FG_WHITE = "\x1b[37m"
    FG_GRAY = "\x1b[90m"


@dataclass(frozen=True, slots=True)
class RenderOptions:
    show_coords: bool = True
    color: bool = True
    flag_over_target: bool = True
    player_over_prison: bool = True


class KeyReader:
    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._old: Optional[Sequence[int]] = None

    def __enter__(self) -> "KeyReader":
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._fd is not None and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll_key(self) -> Optional[str]:
        if self._fd is None:
            return None
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return None
        ch = sys.stdin.read(1)
        return ch


def _normalize_controller_reply(reply: Any) -> ActionMap:
    if reply is None:
        return {}
    if isinstance(reply, dict) and "players" in reply and isinstance(reply["players"], dict):
        return {str(k): (v if v is None else str(v)) for k, v in reply["players"].items()}
    if isinstance(reply, dict):
        return {str(k): (v if v is None else str(v)) for k, v in reply.items()}
    raise TypeError(f"Invalid controller reply: {type(reply).__name__}")


class Controller:
    def start_game(self, init_payload: Dict[str, Any]) -> None:
        return None

    def plan_next_actions(self, status_payload: Dict[str, Any]) -> ActionMap:
        return {}

    def game_over(self, finished_payload: Dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


class NullController(Controller):
    pass


class RandomController(Controller):
    def __init__(self, team: str, seed: Optional[int] = None) -> None:
        self.team = team
        self._rng = random.Random(seed)

    def plan_next_actions(self, status_payload: Dict[str, Any]) -> ActionMap:
        actions: ActionMap = {}
        for p in status_payload.get("myteamPlayer", []):
            name = p.get("name")
            if not isinstance(name, str):
                continue
            actions[name] = self._rng.choice([None, "up", "down", "left", "right"])
        return actions


class ModuleController(Controller):
    def __init__(self, spec: str) -> None:
        self._module = self._load(spec)
        self._start = getattr(self._module, "start_game", None)
        self._plan = getattr(self._module, "plan_next_actions", None)
        self._end = getattr(self._module, "game_over", None)
        if not callable(self._start) or not callable(self._plan):
            raise TypeError(f"{spec!r} must define callables start_game(req) and plan_next_actions(req)")

    @staticmethod
    def _load(spec: str):
        if spec.endswith(".py") or "/" in spec or "\\" in spec:
            path = Path(spec).expanduser().resolve()
            mod_name = f"pbt_ext_{path.stem}"
            module_spec = importlib.util.spec_from_file_location(mod_name, path)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"Failed to load module from {path}")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module
        return importlib.import_module(spec)

    def start_game(self, init_payload: Dict[str, Any]) -> None:
        self._start(init_payload)

    def plan_next_actions(self, status_payload: Dict[str, Any]) -> ActionMap:
        return _normalize_controller_reply(self._plan(status_payload))

    def game_over(self, finished_payload: Dict[str, Any]) -> None:
        if callable(self._end):
            self._end(finished_payload)


class SubprocessController(Controller):
    def __init__(self, argv: Sequence[str]) -> None:
        self._p = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )

    def _send(self, payload: Dict[str, Any]) -> None:
        assert self._p.stdin is not None
        self._p.stdin.write(json.dumps(payload) + "\n")
        self._p.stdin.flush()

    def _recv(self) -> Any:
        assert self._p.stdout is not None
        line = self._p.stdout.readline()
        if not line:
            raise RuntimeError("Controller subprocess closed stdout")
        return json.loads(line)

    def start_game(self, init_payload: Dict[str, Any]) -> None:
        self._send(init_payload)

    def plan_next_actions(self, status_payload: Dict[str, Any]) -> ActionMap:
        self._send(status_payload)
        return _normalize_controller_reply(self._recv())

    def game_over(self, finished_payload: Dict[str, Any]) -> None:
        self._send(finished_payload)

    def close(self) -> None:
        if self._p.poll() is None:
            self._p.terminate()


class KeyboardController(Controller):
    """
    Controls a single player with keys:
      - L team: WASD for L0
      - R team: IJKL for R0
    """

    def __init__(self, team: str) -> None:
        self.team = team
        self._last_dir: Optional[str] = None

    def feed_key(self, ch: str) -> None:
        ch = ch.lower()
        if self.team == "L":
            mapping = {"w": "up", "s": "down", "a": "left", "d": "right"}
        else:
            mapping = {"i": "up", "k": "down", "j": "left", "l": "right"}
        if ch in mapping:
            self._last_dir = mapping[ch]

    def plan_next_actions(self, status_payload: Dict[str, Any]) -> ActionMap:
        target_player = f"{self.team}0"
        actions: ActionMap = {target_player: self._last_dir}
        self._last_dir = None
        return actions


def _make_controller(team: str, spec: str, seed: Optional[int]) -> Controller:
    if spec == "none":
        return NullController()
    if spec == "random":
        return RandomController(team=team, seed=seed)
    if spec == "keyboard":
        return KeyboardController(team=team)
    if spec.startswith("module:"):
        return ModuleController(spec[len("module:") :])
    if spec.startswith("cmd:"):
        argv = json.loads(spec[len("cmd:") :])
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError("cmd: expects JSON array argv, e.g. cmd:[\"python3\",\"bot.py\"]")
        return SubprocessController(argv)
    raise ValueError(f"Unknown controller spec: {spec!r}")


def _team_color(team: str, *, enable: bool) -> str:
    if not enable:
        return ""
    return Ansi.FG_BLUE if team == "L" else Ansi.FG_RED


def _render(
    env: MockEnvVNew,
    full_state: Dict[str, Dict[str, Any]],
    *,
    opts: RenderOptions,
    last_actions_l: ActionMap,
    last_actions_r: ActionMap,
    turn: int,
) -> str:
    blocked = env._blocked_set
    prisons = {"L": set(env.prisons["L"]), "R": set(env.prisons["R"])}
    targets = {"L": set(env.targets["L"]), "R": set(env.targets["R"])}

    players: Dict[Tuple[int, int], Tuple[str, str, bool]] = {}
    for team in ("L", "R"):
        for p in full_state[team]["myteamPlayer"]:
            x = int(p["posX"])
            y = int(p["posY"])
            name = str(p["name"])
            in_prison = bool(p.get("inPrison", False))
            players[(x, y)] = (name, team, in_prison)

    flags: Dict[Tuple[int, int], Tuple[str, bool]] = {}
    for team in ("L", "R"):
        for f in full_state[team]["myteamFlag"]:
            x = int(f["posX"])
            y = int(f["posY"])
            can_pickup = bool(f.get("canPickup", False))
            flags[(x, y)] = (team, can_pickup)

    def cell_at(x: int, y: int) -> str:
        pos = (x, y)
        if pos in blocked:
            return f"{Ansi.FG_WHITE}██{Ansi.RESET}" if opts.color else "██"

        prison_team = None
        if pos in prisons["L"]:
            prison_team = "L"
        elif pos in prisons["R"]:
            prison_team = "R"

        target_team = None
        if pos in targets["L"]:
            target_team = "L"
        elif pos in targets["R"]:
            target_team = "R"

        player = players.get(pos)
        flag = flags.get(pos)

        if opts.flag_over_target and flag is not None:
            team, can_pickup = flag
            label = f"{team}F" if can_pickup else f"{team}f"
            return f"{_team_color(team, enable=opts.color)}{label}{Ansi.RESET}" if opts.color else label

        if opts.player_over_prison and player is not None:
            name, team, in_prison = player
            label = name[:2].ljust(2)
            color = _team_color(team, enable=opts.color)
            dim = Ansi.DIM if (opts.color and in_prison) else ""
            reset = Ansi.RESET if opts.color else ""
            return f"{dim}{color}{label}{reset}"

        if prison_team is not None:
            label = "PP"
            return f"{Ansi.FG_CYAN}{label}{Ansi.RESET}" if opts.color else label

        if target_team is not None:
            label = "TT"
            return f"{Ansi.FG_YELLOW}{label}{Ansi.RESET}" if opts.color else label

        if player is not None:
            name, team, in_prison = player
            label = name[:2].ljust(2)
            color = _team_color(team, enable=opts.color)
            dim = Ansi.DIM if (opts.color and in_prison) else ""
            reset = Ansi.RESET if opts.color else ""
            return f"{dim}{color}{label}{reset}"

        if flag is not None:
            team, can_pickup = flag
            label = f"{team}F" if can_pickup else f"{team}f"
            return f"{_team_color(team, enable=opts.color)}{label}{Ansi.RESET}" if opts.color else label

        if opts.player_over_prison and prison_team is not None:
            label = "PP"
            return f"{Ansi.FG_CYAN}{label}{Ansi.RESET}" if opts.color else label

        if opts.flag_over_target and target_team is not None:
            label = "TT"
            return f"{Ansi.FG_YELLOW}{label}{Ansi.RESET}" if opts.color else label

        return f"{Ansi.FG_GRAY}..{Ansi.RESET}" if opts.color else ".."

    lines = []
    l_score = full_state["L"]["myteamScore"]
    r_score = full_state["R"]["myteamScore"]
    t_ms = int(full_state["L"]["time"])

    if opts.color:
        title = (
            f"{Ansi.BOLD}PBT MockEnvVNew Terminal Suite{Ansi.RESET}  "
            f"turn={turn}  time={t_ms}ms  "
            f"{Ansi.FG_BLUE}L{Ansi.RESET}:{l_score}  {Ansi.FG_RED}R{Ansi.RESET}:{r_score}"
        )
    else:
        title = f"PBT MockEnvVNew Terminal Suite  turn={turn}  time={t_ms}ms  L:{l_score}  R:{r_score}"
    lines.append(title)

    if opts.color:
        lines.append(
            f"actions L: {Ansi.FG_BLUE}{json.dumps(last_actions_l, ensure_ascii=False)}{Ansi.RESET}  "
            f"R: {Ansi.FG_RED}{json.dumps(last_actions_r, ensure_ascii=False)}{Ansi.RESET}"
        )
    else:
        lines.append(f"actions L: {json.dumps(last_actions_l, ensure_ascii=False)}  R: {json.dumps(last_actions_r, ensure_ascii=False)}")

    if opts.show_coords:
        header = "   " + " ".join([f"{x:2}" for x in range(env.width)])
        lines.append(header)

    for y in range(env.height):
        row = []
        for x in range(env.width):
            row.append(cell_at(x, y))
        if opts.show_coords:
            lines.append(f"{y:2} " + " ".join(row))
        else:
            lines.append(" ".join(row))

    lines.append("keys: space=play/pause  n=step  q=quit  (WASD->L0, IJKL->R0 when using keyboard)")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Terminal wrapper for PBT/mock_env_vnew.py (frontend-like init/status loop).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-flags", type=int, default=9)
    parser.add_argument("--num-players", type=int, default=3)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--delay-ms", type=int, default=0, help="Delay between turns when auto-playing.")
    parser.add_argument("--no-coords", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--l", default="keyboard", help="L team controller: keyboard|random|none|module:<mod>|cmd:<json_argv>")
    parser.add_argument("--r", default="random", help="R team controller: keyboard|random|none|module:<mod>|cmd:<json_argv>")
    args = parser.parse_args(list(argv) if argv is not None else None)

    env = MockEnvVNew(
        width=args.width,
        height=args.height,
        num_players=args.num_players,
        num_flags=args.num_flags,
        seed=args.seed,
    )

    l_ctrl = _make_controller("L", args.l, seed=args.seed)
    r_ctrl = _make_controller("R", args.r, seed=args.seed + 1)

    init_l = env.get_init_payload("L")
    init_r = env.get_init_payload("R")
    l_ctrl.start_game(init_l)
    r_ctrl.start_game(init_r)

    opts = RenderOptions(
        show_coords=not args.no_coords,
        color=(not args.no_color) and _supports_color(),
    )

    stop = False

    def _handle_sigint(_sig: int, _frm: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    playing = args.delay_ms > 0
    last_actions_l: ActionMap = {}
    last_actions_r: ActionMap = {}

    key_reader = KeyReader()
    with key_reader:
        for turn in range(args.max_turns + 1):
            full_state = env.get_full_state()
            _clear_screen()
            sys.stdout.write(
                _render(
                    env,
                    full_state,
                    opts=opts,
                    last_actions_l=last_actions_l,
                    last_actions_r=last_actions_r,
                    turn=turn,
                )
                + "\n"
            )
            sys.stdout.flush()

            if stop:
                break

            key = key_reader.poll_key()
            if key is not None:
                if key == "q":
                    break
                if key == " ":
                    playing = not playing
                if key == "n":
                    playing = False
                if isinstance(l_ctrl, KeyboardController):
                    l_ctrl.feed_key(key)
                if isinstance(r_ctrl, KeyboardController):
                    r_ctrl.feed_key(key)

            if not playing and key != "n":
                # Idle a bit so key polling doesn't spin.
                select.select([], [], [], 0.03)
                continue

            status_l = full_state["L"]
            status_r = full_state["R"]

            try:
                last_actions_l = l_ctrl.plan_next_actions(status_l)
                last_actions_r = r_ctrl.plan_next_actions(status_r)
            except Exception as exc:
                _clear_screen()
                print("Controller error:", exc)
                raise

            _, done, info = env.step(last_actions_l, last_actions_r)
            if done:
                finished_payload = {"action": "finished", "winner": info.get("winner"), "time": env.time_ms}
                l_ctrl.game_over(finished_payload)
                r_ctrl.game_over(finished_payload)
                full_state = env.get_full_state()
                _clear_screen()
                print(
                    _render(
                        env,
                        full_state,
                        opts=opts,
                        last_actions_l=last_actions_l,
                        last_actions_r=last_actions_r,
                        turn=turn,
                    )
                )
                print(f"game over: winner={info.get('winner')}")
                break

            if args.delay_ms > 0:
                select.select([], [], [], args.delay_ms / 1000.0)

    l_ctrl.close()
    r_ctrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

