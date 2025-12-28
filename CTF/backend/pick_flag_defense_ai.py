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
        self.defense_line_x = 0
        self.safe_cells = set()
        self.enemy_cells = set()
        self.boundary_ys = []
        self.defense_ys = []
        self.safe_entry_cells = []
        self.guard_post = None
        self.attack_phase = False
        self.attackers = set()
        self.defender_name = None
        self.attack_progress = {}
        self.avoid_radius = 1
        self.avoid_radius_return = 2

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
            inner_x = self.our_boundary_x - 1
        else:
            self.our_boundary_x = self.right_min_x
            self.enemy_boundary_x = self.left_max_x
            inner_x = self.our_boundary_x + 1

        if 0 <= inner_x < width and self._is_safe((inner_x, 0)):
            self.defense_line_x = inner_x
        else:
            self.defense_line_x = self.our_boundary_x

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
        self.defense_ys = [
            y
            for y in range(height)
            if (self.defense_line_x, y) not in self.world.walls
        ]
        line_ys = self.defense_ys or self.boundary_ys
        line_x = self.defense_line_x if self.defense_ys else self.our_boundary_x
        self.safe_entry_cells = [(line_x, y) for y in line_ys]
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
        line_ys = self.defense_ys or self.boundary_ys
        line_x = self.defense_line_x if self.defense_ys else self.our_boundary_x
        if not line_ys:
            return next(iter(self.safe_cells), (0, 0))
        center_y = self.world.height // 2
        best_y = min(line_ys, key=lambda y: abs(y - center_y))
        return (line_x, best_y)

    def _closest_boundary_y(self, target_y):
        return self._closest_line_y(target_y, self.boundary_ys)

    def _closest_defense_y(self, target_y):
        return self._closest_line_y(target_y, self.defense_ys or self.boundary_ys)

    def _closest_line_y(self, target_y, candidates):
        if not candidates:
            return max(0, min(self.world.height - 1, int(round(target_y))))
        return min(candidates, key=lambda y: abs(y - target_y))

    def _spread_line_y(self, occupied_ys, candidates):
        if not candidates:
            candidates = [self._closest_boundary_y(self.world.height // 2)]
        if not occupied_ys:
            return self._closest_line_y(self.world.height // 2, candidates)
        return max(candidates, key=lambda y: min(abs(y - oy) for oy in occupied_ys))

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

    def _target_cell_for_opponent(self, opponent):
        opp_pos = self._pos(opponent)
        if self._is_safe(opp_pos):
            line_x = self.our_boundary_x
            line_ys = self.boundary_ys
            target_y = self._closest_line_y(opp_pos[1], line_ys)
        else:
            if opp_pos[0] == self.enemy_boundary_x:
                line_x = self.our_boundary_x
                line_ys = self.boundary_ys
            else:
                line_x = self.defense_line_x if self.defense_ys else self.our_boundary_x
                line_ys = self.defense_ys or self.boundary_ys
            target_y = self._closest_line_y(opp_pos[1], line_ys)
        return line_x, target_y

    def _assign_defense_targets(self, defenders, opponents):
        targets = {}
        if not defenders:
            return targets

        if not opponents:
            fallback_x = self.defense_line_x if self.defense_ys else self.our_boundary_x
            fallback_y = self._closest_line_y(self.world.height // 2, self.defense_ys or self.boundary_ys)
            fallback = (fallback_x, fallback_y)
            for p in defenders:
                targets[p["name"]] = fallback
            return targets

        defenders = list(defenders)
        opponent_targets = []
        intruders = []
        border_threats = []
        outsiders = []
        for opp in opponents:
            opp_pos = self._pos(opp)
            line_x, target_y = self._target_cell_for_opponent(opp)
            entry = (opp, line_x, target_y)
            opponent_targets.append(entry)
            if self._is_safe(opp_pos):
                intruders.append(entry)
            elif opp_pos[0] == self.enemy_boundary_x:
                border_threats.append(entry)
            else:
                outsiders.append(entry)

        def cost_for(defender, opp_entry):
            _, tx, ty = opp_entry
            return self._manhattan(self._pos(defender), (tx, ty))

        assigned_names = set()
        occupied_ys = []

        if len(defenders) <= len(opponent_targets):
            priority_targets = intruders + border_threats
            non_priority = [entry for entry in outsiders if entry not in border_threats]
            if priority_targets and len(priority_targets) >= len(defenders):
                candidate_pool = priority_targets
                required_targets = []
                remaining_slots = len(defenders)
            else:
                required_targets = priority_targets
                remaining_slots = len(defenders) - len(required_targets)
                candidate_pool = non_priority

            best_total = None
            best_perm = None
            if remaining_slots <= 0:
                fixed_targets = required_targets[: len(defenders)]
                for perm in itertools.permutations(fixed_targets, len(defenders)):
                    total = sum(cost_for(defenders[idx], perm[idx]) for idx in range(len(defenders)))
                    if best_total is None or total < best_total:
                        best_total = total
                        best_perm = perm
            else:
                for subset in itertools.combinations(candidate_pool, remaining_slots):
                    combined = list(required_targets) + list(subset)
                    for perm in itertools.permutations(combined, len(defenders)):
                        total = sum(cost_for(defenders[idx], perm[idx]) for idx in range(len(defenders)))
                        if best_total is None or total < best_total:
                            best_total = total
                            best_perm = perm

            for idx, defender in enumerate(defenders):
                _, tx, ty = best_perm[idx]
                targets[defender["name"]] = (tx, ty)
                assigned_names.add(defender["name"])
                occupied_ys.append(ty)
            return targets

        best_total = None
        best_perm = None
        for perm in itertools.permutations(defenders, len(opponent_targets)):
            total = sum(cost_for(perm[idx], opponent_targets[idx]) for idx in range(len(opponent_targets)))
            if best_total is None or total < best_total:
                best_total = total
                best_perm = perm

        for idx, opp_entry in enumerate(opponent_targets):
            defender = best_perm[idx]
            _, tx, ty = opp_entry
            targets[defender["name"]] = (tx, ty)
            assigned_names.add(defender["name"])
            occupied_ys.append(ty)

        waiting = [p for p in defenders if p["name"] not in assigned_names]
        line_x = self.defense_line_x if self.defense_ys else self.our_boundary_x
        line_ys = self.defense_ys or self.boundary_ys
        for p in waiting:
            wait_y = self._spread_line_y(occupied_ys, line_ys)
            targets[p["name"]] = (line_x, wait_y)
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
            guard = self.guard_post or (self.defense_line_x, self.world.height // 2)
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

    def _plan_defense(self, my_free, opponents_free, targets, my_prisoners, prisons):
        actions = {}
        intruders = [o for o in opponents_free if self._is_safe(self._pos(o))]
        border_threats = [
            o for o in opponents_free
            if (not self._is_safe(self._pos(o))) and self._pos(o)[0] == self.enemy_boundary_x
        ]
        low_pressure = not intruders and not border_threats

        rescuer_name = None
        if low_pressure and my_prisoners and prisons:
            rescue_candidates = [
                p for p in my_free
                if self._is_safe(self._pos(p)) and not p.get("hasFlag")
            ]
            best_path = None
            best_player = None
            for p in rescue_candidates:
                path = self._route_any(self._pos(p), prisons, restrict_safe=True)
                if path and (best_path is None or len(path) < len(best_path)):
                    best_path = path
                    best_player = p
            if best_player:
                rescuer_name = best_player["name"]
                move = self._next_move(self._pos(best_player), best_path)
                if move:
                    actions[rescuer_name] = move

        defenders = [p for p in my_free if p["name"] != rescuer_name]
        defense_targets = self._assign_defense_targets(defenders, opponents_free)
        for p in defenders:
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
            intruders = [o for o in opponents_free if self._is_safe(self._pos(o))]
            border_threats = [
                o for o in opponents_free
                if (not self._is_safe(self._pos(o))) and self._pos(o)[0] == self.enemy_boundary_x
            ]
            if intruders:
                opp = min(intruders, key=lambda o: self._manhattan(self._pos(defender), self._pos(o)))
                target_y = self._closest_boundary_y(opp["posY"])
                dest = (self.our_boundary_x, target_y)
            elif border_threats:
                opp = min(border_threats, key=lambda o: abs(o["posY"] - defender["posY"]))
                target_y = self._closest_boundary_y(opp["posY"])
                dest = (self.our_boundary_x, target_y)
            else:
                opp = min(opponents_free, key=lambda o: abs(o["posY"] - defender["posY"]))
                target_y = self._closest_defense_y(opp["posY"])
                line_x = self.defense_line_x if self.defense_ys else self.our_boundary_x
                dest = (line_x, target_y)
        else:
            dest = self.guard_post or (self.defense_line_x, self.world.height // 2)

        start = self._pos(defender)
        path = self._route(start, dest, restrict_safe=True)
        if not path and self.guard_post:
            path = self._route(start, self.guard_post, restrict_safe=True)
        return self._next_move(start, path)

    def _plan_attacker(self, player, opponents_free, enemy_flags, targets):
        start = self._pos(player)
        avoid_radius = self.avoid_radius_return if player.get("hasFlag") else self.avoid_radius
        avoid2 = self._expanded_enemy_obstacles(opponents_free, avoid_radius)
        avoid1 = self._expanded_enemy_obstacles(opponents_free, 1)
        avoid0 = self._expanded_enemy_obstacles(opponents_free, 0)

        if player.get("hasFlag"):
            if not self._is_safe(start):
                entry_cells = self.safe_entry_cells or [(self.our_boundary_x, y) for y in self.boundary_ys]
                for avoid in (avoid2, avoid1, avoid0, None):
                    path = self._route_any(start, entry_cells, extra_obstacles=avoid, restrict_safe=False)
                    move = self._next_move(start, path)
                    if move:
                        return move

            goal_positions = list(targets)
            for avoid in (avoid2, avoid1, avoid0, None):
                path = self._route_any(start, goal_positions, extra_obstacles=avoid, restrict_safe=True)
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
                guard = self.guard_post or (self.defense_line_x, self.world.height // 2)
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
            fallback = self.guard_post or (self.defense_line_x, self.world.height // 2)
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
        my_prisoners = self.world.list_players(mine=True, inPrison=True, hasFlag=None) or []
        opponents_free = self.world.list_players(mine=False, inPrison=False, hasFlag=None) or []
        opponents_prison = self.world.list_players(mine=False, inPrison=True, hasFlag=None) or []
        enemy_flags = self.world.list_flags(mine=False, canPickup=True) or []
        targets = set(self.world.list_targets(mine=True) or [])
        prisons = list(self.world.list_prisons(mine=True) or [])

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
            return self._plan_defense(my_free, opponents_free, targets, my_prisoners, prisons)
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
