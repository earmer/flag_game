from .carrier import find_best_carrier_chase
from .config import UltraConfig
from .intercept import best_intercept_on_path, predict_intruder_goal_path
from .loader import load_nt2_module
from .trap import plan_trap_on_carriers

_NT2 = load_nt2_module()
EliteCTFAI_NT2 = _NT2.EliteCTFAI_NT2
BETTER_LURE = bool(getattr(_NT2, "BETTER_LURE", True))


class UltraCTFAI(EliteCTFAI_NT2):
    def __init__(self, show_gap_in_msec=1000.0, config=None):
        super().__init__(show_gap_in_msec=show_gap_in_msec)
        self.config = config or UltraConfig()
        self._apply_config()

    def _apply_config(self):
        self.intercept_horizon_trap = int(self.config.intercept_horizon_trap)
        self.intercept_horizon_carrier = int(self.config.intercept_horizon_carrier)
        self.intercept_arrival_slack = int(self.config.intercept_arrival_slack)
        self.carrier_chase_detour_slack = int(self.config.carrier_chase_detour_slack)
        self.carrier_chase_min_ticks = int(self.config.carrier_chase_min_ticks)
        self.double_team_advantage_margin = int(self.config.double_team_advantage_margin)

    def start_game(self, req):
        super().start_game(req)
        self._apply_config()
        self._carrier_chase = {}

    def _predict_intruder_goal_path(self, intruder, *, my_flags, prisons):
        return predict_intruder_goal_path(
            self,
            intruder,
            my_flags=my_flags,
            prisons=prisons,
            weights=self.config.weights.boundary,
        )

    def _best_intercept_on_path(self, pursuer_start, opponent_path, *, horizon):
        return best_intercept_on_path(
            self,
            pursuer_start,
            opponent_path,
            horizon=horizon,
            arrival_slack=self.intercept_arrival_slack,
            weights=self.config.weights.intercept,
        )

    def _move_carrier(self, player, opponents, targets):
        start = self._pos(player)
        name = player["name"]

        if not self._is_safe(start):
            self._carrier_chase.pop(name, None)
            return super()._move_carrier(player, opponents, targets)

        my_flags = self.world.list_flags(mine=True, canPickup=True) or []
        prisons = self.world.list_prisons(mine=True) or []

        option, direct_steps, direct_score = find_best_carrier_chase(
            self,
            start,
            [o for o in opponents or [] if self._is_safe(self._pos(o))],
            targets,
            my_flags,
            prisons,
            self.config,
        )

        if direct_steps is not None and direct_steps <= int(self.config.carrier_direct_target_threshold):
            self._carrier_chase.pop(name, None)
            direct_path = self._route_any(start, targets, restrict_safe=True) if targets else None
            move = self._next_move(start, direct_path)
            if move:
                return move
            return super()._move_carrier(player, opponents, targets)

        if option and (direct_score is None or option.score <= direct_score + float(self.config.carrier_chase_score_slack)):
            self._carrier_chase[name] = {
                "target": option.target["name"],
                "until": self.tick + int(self.carrier_chase_min_ticks),
                "last_pos": self._pos(option.target),
                "intercept": option.intercept.cell,
            }
            move = self._next_move(start, option.intercept.path)
            if move:
                return move

        chase_state = self._carrier_chase.get(name)
        if chase_state and self.tick <= int(chase_state.get("until", -1)):
            opponents_by_name = {o["name"]: o for o in opponents or [] if self._is_safe(self._pos(o))}
            target = opponents_by_name.get(chase_state.get("target"))
            if target is not None:
                o_path = self._predict_intruder_goal_path(
                    target,
                    my_flags=my_flags,
                    prisons=prisons,
                )
                if o_path:
                    intercept = self._best_intercept_on_path(
                        start,
                        o_path,
                        horizon=self.intercept_horizon_carrier,
                    )
                    if intercept:
                        move = self._next_move(start, intercept.path)
                        if move:
                            chase_state["last_pos"] = self._pos(target)
                            chase_state["intercept"] = intercept.cell
                            return move

            last_pos = chase_state.get("last_pos")
            if last_pos is not None:
                path = self._route(start, last_pos, restrict_safe=True)
                move = self._next_move(start, path)
                if move:
                    return move
        else:
            self._carrier_chase.pop(name, None)

        direct_path = self._route_any(start, targets, restrict_safe=True) if targets else None
        move = self._next_move(start, direct_path)
        if move:
            return move

        return super()._move_carrier(player, opponents, targets)

    def _plan_trap_on_carriers(self, runners, intruder_carriers, intruders, my_flags):
        return plan_trap_on_carriers(
            self,
            runners,
            intruder_carriers,
            intruders,
            my_flags,
            self.config,
        )
