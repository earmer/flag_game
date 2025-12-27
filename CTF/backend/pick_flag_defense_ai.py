import asyncio
import itertools

from lib.game_engine import GameMap, run_game_server


class DefensiveCTFAI:
    def __init__(self, show_gap_in_msec=1000.0):
        self.world = GameMap(show_gap_in_msec=show_gap_in_msec)
        self.my_side_is_left = True
        self.left_max_x = 0
        self.right_min_x = 0
        self.our_boundary_x = 0
        self.enemy_boundary_x = 0
        self.safe_cells = set()
        self.enemy_cells = set()
        self.boundary_ys = []
        self.guard_post = None
        self.attack_phase = False
        self.attackers = set()
        self.defender_name = None
        self.attack_progress = {}
        self.avoid_radius = 1

    def start_game(self, req):
        self.world.init(req)
        self.attack_phase = False
        self.attackers = set()
        self.defender_name = None
        self.attack_progress = {}
        self._init_geometry()
        side = "Left" if self.my_side_is_left else "Right"
        print(f"Defense AI started. Side: {side}; guard_post={self.guard_post}")

    def game_over(self, _req):
        print("Game Over!")

    def _init_geometry(self):
        width = self.world.width
        height = self.world.height

        targets = list(self.world.list_targets(mine=True) or [])
        if targets:
            self.my_side_is_left = self.world.is_on_left(next(iter(targets)))
        else:
            self.my_side_is_left = self.world.my_team_name == "L"

        self.left_max_x = (width - 1) // 2
        self.right_min_x = self.left_max_x + 1

        if self.my_side_is_left:
            self.our_boundary_x = self.left_max_x
            self.enemy_boundary_x = self.right_min_x
        else:
            self.our_boundary_x = self.right_min_x
            self.enemy_boundary_x = self.left_max_x

        self.safe_cells = set()
        self.enemy_cells = set()
        for x in range(width):
            for y in range(height):
                pos = (x, y)
                if self._is_safe(pos):
                    self.safe_cells.add(pos)
                else:
                    self.enemy_cells.add(pos)

        self.boundary_ys = [
            y
            for y in range(height)
            if (self.our_boundary_x, y) not in self.world.walls
        ]
        self.guard_post = self._choose_guard_post()

    def _is_safe(self, pos):
        return self.world.is_on_left(pos) == self.my_side_is_left

    @staticmethod
    def _pos(entity):
        return (int(round(entity["posX"])), int(round(entity["posY"])))

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _choose_guard_post(self):
        if not self.boundary_ys:
            return next(iter(self.safe_cells), (0, 0))
        center_y = self.world.height // 2
        best_y = min(self.boundary_ys, key=lambda y: abs(y - center_y))
        return (self.our_boundary_x, best_y)

    def _closest_boundary_y(self, target_y):
        if not self.boundary_ys:
            return max(0, min(self.world.height - 1, int(round(target_y))))
        return min(self.boundary_ys, key=lambda y: abs(y - target_y))

    def _spread_boundary_y(self, occupied_ys):
        if not self.boundary_ys:
            return self.world.height // 2
        if not occupied_ys:
            return self._closest_boundary_y(self.world.height // 2)
        return max(self.boundary_ys, key=lambda y: min(abs(y - oy) for oy in occupied_ys))

    def _expanded_enemy_obstacles(self, opponents, radius):
        if radius <= 0:
            return {self._pos(o) for o in opponents}
        obstacles = set()
        for o in opponents:
            ex, ey = self._pos(o)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) > radius:
                        continue
                    nx, ny = ex + dx, ey + dy
                    if 0 <= nx < self.world.width and 0 <= ny < self.world.height:
                        obstacles.add((nx, ny))
        return obstacles

    def _route(self, start, goal, *, extra_obstacles=None, restrict_safe=False):
        extras = set(extra_obstacles or [])
        if restrict_safe:
            extras |= self.enemy_cells
        return self.world.route_to(start, goal, extra_obstacles=extras)

    def _route_any(self, start, goals, *, extra_obstacles=None, restrict_safe=False):
        best = []
        for goal in goals or []:
            path = self._route(start, goal, extra_obstacles=extra_obstacles, restrict_safe=restrict_safe)
            if path and (not best or len(path) < len(best)):
                best = path
        return best

    def _next_move(self, start, path):
        if not path or len(path) < 2:
            return None
        return GameMap.get_direction(start, path[1])

    def _assign_defense_targets(self, defenders, opponents):
        targets = {}
        if not defenders:
            return targets

        if not opponents:
            fallback = self.guard_post or (self.our_boundary_x, self.world.height // 2)
            for p in defenders:
                targets[p["name"]] = fallback
            return targets

        defenders = list(defenders)
        opponents_sorted = sorted(opponents, key=lambda p: p["posY"])

        if len(opponents_sorted) >= len(defenders):
            defenders_sorted = sorted(defenders, key=lambda p: p["posY"])
            for p, opp in zip(defenders_sorted, opponents_sorted):
                target_y = self._closest_boundary_y(opp["posY"])
                targets[p["name"]] = (self.our_boundary_x, target_y)
            return targets

        best_total = None
        best_pairing = None
        for perm in itertools.permutations(defenders, len(opponents_sorted)):
            total = sum(
                abs(perm[idx]["posY"] - opponents_sorted[idx]["posY"])
                for idx in range(len(opponents_sorted))
            )
            if best_total is None or total < best_total:
                best_total = total
                best_pairing = list(zip(perm, opponents_sorted))

        assigned_names = set()
        occupied_ys = []
        for defender, opp in best_pairing:
            target_y = self._closest_boundary_y(opp["posY"])
            targets[defender["name"]] = (self.our_boundary_x, target_y)
            assigned_names.add(defender["name"])
            occupied_ys.append(target_y)

        waiting = [p for p in defenders if p["name"] not in assigned_names]
        for p in waiting:
            wait_y = self._spread_boundary_y(occupied_ys)
            targets[p["name"]] = (self.our_boundary_x, wait_y)
            occupied_ys.append(wait_y)

        return targets

    def _choose_attack_roles(self, players, opponents):
        if not players:
            self.attackers = set()
            self.defender_name = None
            self.attack_progress = {}
            return

        opponent_positions = [self._pos(o) for o in opponents]
        if opponent_positions:
            distances = [
                (min(self._manhattan(self._pos(p), o) for o in opponent_positions), p)
                for p in players
            ]
            distances.sort(key=lambda item: item[0])
            defender = distances[0][1]
            if len(distances) >= 3:
                attackers = [p for _, p in distances[-2:]]
            elif len(distances) == 2:
                attackers = [distances[1][1]]
            else:
                attackers = []
        else:
            guard = self.guard_post or (self.our_boundary_x, self.world.height // 2)
            distances = [(self._manhattan(self._pos(p), guard), p) for p in players]
            distances.sort(key=lambda item: item[0])
            defender = distances[0][1]
            attackers = [p for _, p in distances[1:]]

        attackers = attackers[:2]
        self.attackers = {p["name"] for p in attackers}
        self.defender_name = defender["name"]
        self.attack_progress = {
            name: self.attack_progress.get(name, False) for name in self.attackers
        }

    def _begin_attack_phase(self, players, opponents):
        self.attack_phase = True
        self._choose_attack_roles(players, opponents)
        self.attack_progress = {name: False for name in self.attackers}

    def _update_attack_progress(self, players_by_name):
        for name in self.attackers:
            player = players_by_name.get(name)
            if not player or player.get("inPrison"):
                continue
            if not self._is_safe(self._pos(player)):
                self.attack_progress[name] = True

    def _attackers_done(self, players_by_name):
        if not self.attackers:
            return True
        for name in self.attackers:
            player = players_by_name.get(name)
            if player is None:
                continue
            if player.get("inPrison"):
                continue
            if not self.attack_progress.get(name, False):
                return False
            if self._is_safe(self._pos(player)):
                continue
            return False
        return True

    def _plan_defense(self, my_free, opponents_free, targets):
        actions = {}
        defense_targets = self._assign_defense_targets(my_free, opponents_free)
        for p in my_free:
            if p.get("hasFlag"):
                move = self._plan_attacker(p, opponents_free, [], targets)
                if move:
                    actions[p["name"]] = move
                continue
            dest = defense_targets.get(p["name"])
            if not dest:
                continue
            start = self._pos(p)
            path = self._route(start, dest, restrict_safe=True)
            if not path and self.guard_post:
                path = self._route(start, self.guard_post, restrict_safe=True)
            move = self._next_move(start, path)
            if move:
                actions[p["name"]] = move
        return actions

    def _plan_defender(self, defender, opponents_free):
        if defender is None:
            return None
        if opponents_free:
            opp_pos = min(opponents_free, key=lambda o: abs(o["posY"] - defender["posY"]))
            target_y = self._closest_boundary_y(opp_pos["posY"])
            dest = (self.our_boundary_x, target_y)
        else:
            dest = self.guard_post or (self.our_boundary_x, self.world.height // 2)

        start = self._pos(defender)
        path = self._route(start, dest, restrict_safe=True)
        if not path and self.guard_post:
            path = self._route(start, self.guard_post, restrict_safe=True)
        return self._next_move(start, path)

    def _plan_attacker(self, player, opponents_free, enemy_flags, targets):
        start = self._pos(player)
        avoid1 = self._expanded_enemy_obstacles(opponents_free, self.avoid_radius)
        avoid0 = self._expanded_enemy_obstacles(opponents_free, 0)

        if player.get("hasFlag"):
            goal_positions = list(targets)
            for avoid in (avoid1, avoid0, None):
                path = self._route_any(start, goal_positions, extra_obstacles=avoid, restrict_safe=False)
                move = self._next_move(start, path)
                if move:
                    return move
            return None

        flag_positions = [(int(f["posX"]), int(f["posY"])) for f in enemy_flags if f.get("canPickup")]
        if not flag_positions:
            return None
        for avoid in (avoid1, avoid0, None):
            path = self._route_any(start, flag_positions, extra_obstacles=avoid, restrict_safe=False)
            move = self._next_move(start, path)
            if move:
                return move
        return None

    def _plan_attack(self, my_all, my_free, opponents_free, enemy_flags, targets):
        actions = {}
        players_by_name = {p["name"]: p for p in my_all}

        defender = None
        if self.defender_name:
            candidate = players_by_name.get(self.defender_name)
            if candidate and not candidate.get("inPrison"):
                defender = candidate

        if defender is None:
            candidates = [p for p in my_free if p["name"] not in self.attackers]
            if candidates:
                guard = self.guard_post or (self.our_boundary_x, self.world.height // 2)
                defender = min(candidates, key=lambda p: self._manhattan(self._pos(p), guard))

        if defender:
            if defender.get("hasFlag"):
                move = self._plan_attacker(defender, opponents_free, enemy_flags, targets)
            else:
                move = self._plan_defender(defender, opponents_free)
            if move:
                actions[defender["name"]] = move

        for name in self.attackers:
            player = players_by_name.get(name)
            if not player or player.get("inPrison"):
                continue
            move = self._plan_attacker(player, opponents_free, enemy_flags, targets)
            if move:
                actions[player["name"]] = move

        extra_free = [
            p for p in my_free
            if p["name"] not in self.attackers and (not defender or p["name"] != defender["name"])
        ]
        if extra_free:
            fallback = self.guard_post or (self.our_boundary_x, self.world.height // 2)
            for p in extra_free:
                start = self._pos(p)
                path = self._route(start, fallback, restrict_safe=True)
                move = self._next_move(start, path)
                if move:
                    actions[p["name"]] = move

        return actions

    def plan_next_actions(self, req):
        if not self.world.update(req):
            return {}

        my_all = self.world.list_players(mine=True, inPrison=None, hasFlag=None) or []
        my_free = [p for p in my_all if not p.get("inPrison")]
        opponents_free = self.world.list_players(mine=False, inPrison=False, hasFlag=None) or []
        opponents_prison = self.world.list_players(mine=False, inPrison=True, hasFlag=None) or []
        enemy_flags = self.world.list_flags(mine=False, canPickup=True) or []
        targets = set(self.world.list_targets(mine=True) or [])

        if not self.attack_phase and len(opponents_prison) >= 2:
            if len(my_free) >= 3:
                self._begin_attack_phase(my_free, opponents_free)

        if self.attack_phase:
            players_by_name = {p["name"]: p for p in my_all}
            self._update_attack_progress(players_by_name)
            if self._attackers_done(players_by_name):
                self.attack_phase = False
                self.attackers = set()
                self.defender_name = None
                self.attack_progress = {}
            elif not self.attackers and my_free:
                self._choose_attack_roles(my_free, opponents_free)

        if not self.attack_phase:
            return self._plan_defense(my_free, opponents_free, targets)
        return self._plan_attack(my_all, my_free, opponents_free, enemy_flags, targets)


AI = DefensiveCTFAI()


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
