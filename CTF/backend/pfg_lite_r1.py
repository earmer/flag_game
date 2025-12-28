import asyncio

from lib.game_engine import run_game_server
from ultra.ai import UltraCTFAI
from ultra.config import UltraConfig
from ultra.weights import (
    BoundaryPathWeights,
    CarrierChaseWeights,
    DoubleTeamWeights,
    InterceptWeights,
    TrapWeights,
    UltraWeights,
)


class _LiteBase(UltraCTFAI):
    def __init__(self, *, config, show_gap_in_msec=1000.0):
        super().__init__(show_gap_in_msec=show_gap_in_msec, config=config)
        self._pos_hist = {}
        self._last_move = {}

    def start_game(self, req):
        super().start_game(req)
        self._pos_hist = {}
        self._last_move = {}

    def _role_bias(self, name, lane_y):
        if self.world.height <= 0:
            return 0.0
        seed = sum(ord(c) for c in name) % self.world.height
        return 0.02 * abs(int(seed) - int(lane_y))

    def _closest_runner_to_cell(self, runners, cell):
        best_runner = None
        best_path = []
        best_score = None
        for p in runners:
            path = self._route(self._pos(p), cell, restrict_safe=True)
            if not path:
                continue
            dist = len(path) - 1
            score = dist + self._role_bias(p["name"], cell[1])
            if best_score is None or score < best_score:
                best_score = score
                best_path = path
                best_runner = p
        return best_runner, best_path

    def _apply_move(self, start, move):
        x, y = start
        if move == "up":
            return (x, y - 1)
        if move == "down":
            return (x, y + 1)
        if move == "left":
            return (x - 1, y)
        if move == "right":
            return (x + 1, y)
        return start

    def _score_alt_move(self, start, next_pos):
        safe_penalty = 0 if self._is_safe(next_pos) else 1
        guard_bias = 0
        if self.guard_post is not None:
            guard_bias = abs(int(next_pos[1]) - int(self.guard_post[1])) * 0.01
        return safe_penalty + guard_bias

    def _choose_alternate_move(self, start, avoid_cells):
        moves = [("up", (0, -1)), ("down", (0, 1)), ("left", (-1, 0)), ("right", (1, 0))]
        best = None
        best_score = None
        for move, (dx, dy) in moves:
            nx, ny = start[0] + dx, start[1] + dy
            if not (0 <= nx < self.world.width and 0 <= ny < self.world.height):
                continue
            nxt = (nx, ny)
            if nxt in self.world.walls or nxt in avoid_cells:
                continue
            score = self._score_alt_move(start, nxt)
            if best_score is None or score < best_score:
                best_score = score
                best = move
        return best

    def _track_positions(self, players):
        for p in players:
            name = p["name"]
            pos = self._pos(p)
            hist = self._pos_hist.setdefault(name, [])
            if not hist or hist[-1] != pos:
                hist.append(pos)
                if len(hist) > 4:
                    hist.pop(0)

    def _apply_motion_guards(self, actions, players):
        if not actions:
            return actions

        positions = {p["name"]: self._pos(p) for p in players}
        reserved = set()
        new_actions = {}

        for name in sorted(actions):
            start = positions.get(name)
            if start is None:
                continue
            move = actions[name]
            next_pos = self._apply_move(start, move)
            hist = self._pos_hist.get(name, [])
            avoid = set(reserved)
            if len(hist) >= 2:
                avoid.add(hist[-2])
            oscillating = len(hist) >= 2 and next_pos == hist[-2]
            if oscillating or next_pos in reserved:
                alt = self._choose_alternate_move(start, avoid)
                if alt:
                    move = alt
                    next_pos = self._apply_move(start, move)
            new_actions[name] = move
            reserved.add(next_pos)
            self._last_move[name] = move

        return new_actions

    def plan_next_actions(self, req):
        actions = super().plan_next_actions(req)
        my_free = self.world.list_players(mine=True, inPrison=False, hasFlag=None) or []
        self._track_positions(my_free)
        return self._apply_motion_guards(actions, my_free)


def _config_defensive():
    weights = UltraWeights(
        boundary=BoundaryPathWeights(steps=1.0, guard_bias=0.15),
        intercept=InterceptWeights(steps=1.0, margin=1.0, index=0.2),
        carrier=CarrierChaseWeights(
            intercept_steps=1.0,
            detour=1.6,
            threat=0.7,
            margin=0.9,
            direct=0.9,
        ),
        trap=TrapWeights(
            intercept_steps=0.9,
            escape_steps=0.8,
            margin=1.0,
            index=0.2,
            secondary_penalty=0.6,
        ),
        double_team=DoubleTeamWeights(advantage=1.2, urgency=1.1, risk=1.1),
    )
    return UltraConfig(
        intercept_horizon_trap=14,
        intercept_horizon_carrier=7,
        intercept_arrival_slack=1,
        carrier_direct_target_threshold=1,
        carrier_chase_detour_slack=0,
        carrier_chase_score_slack=0.2,
        carrier_chase_min_ticks=12,
        trap_intruder_close_steps=3,
        trap_urgent_escape_steps=3,
        double_team_score_threshold=0.2,
        double_team_advantage_margin=1,
        weights=weights,
    )


class PfgLiteR1(_LiteBase):
    def __init__(self, show_gap_in_msec=1000.0):
        super().__init__(config=_config_defensive(), show_gap_in_msec=show_gap_in_msec)


AI = PfgLiteR1()


def start_game(req):
    AI.start_game(req)


def plan_next_actions(req):
    return AI.plan_next_actions(req)


def game_over(req):
    AI.game_over(req)


async def main():
    import sys

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
