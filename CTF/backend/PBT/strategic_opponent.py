from __future__ import annotations

import collections
from typing import Dict, List, Optional, Tuple

import numpy as np


class StrategicOpponent:
    """智能对手策略 - 基于 pick_flag_ai.py 的逻辑"""

    def __init__(self, team: str, env) -> None:
        """
        Args:
            team: "L" or "R" - 对手所在队伍
            env: MockEnvVNew 实例
        """
        if team not in ("L", "R"):
            raise ValueError(f"Invalid team: {team!r}")

        self.team = team
        self.opponent_team = "R" if team == "L" else "L"
        self.env = env
        self.flag_assignments: Dict[str, Tuple[int, int]] = {}

        # 缓存地图信息
        self.width = env.width
        self.height = env.height
        self.middle_line = env.width / 2

    def get_actions(self, full_state: Dict) -> Dict[str, Optional[str]]:
        """
        生成智能动作

        Returns:
            {player_name: direction} 字典
        """
        status = full_state[self.team]
        opp_status = full_state[self.opponent_team]

        my_players = [p for p in status["myteamPlayer"] if not p["inPrison"]]
        opp_players = [p for p in opp_status["myteamPlayer"] if not p["inPrison"]]
        enemy_flags = [f for f in opp_status["myteamFlag"] if f["canPickup"]]

        # 更新旗帜分配
        self._update_flag_assignments(my_players, enemy_flags)

        # 获取目标区域（从环境中获取）
        my_targets = list(self.env.targets[self.team])

        # 为每个玩家规划动作
        actions = {}
        for player in my_players:
            action = self._plan_player_move(player, opp_players, my_targets)
            if action:
                actions[player["name"]] = action

        return actions

    def _update_flag_assignments(
        self, players: List[Dict], enemy_flags: List[Dict]
    ) -> None:
        """分配旗帜给玩家"""
        active_players = {p["name"] for p in players if not p["hasFlag"]}

        # 清理无效分配
        self.flag_assignments = {
            name: pos
            for name, pos in self.flag_assignments.items()
            if name in active_players
        }

        # 为新玩家分配旗帜
        for player in players:
            if (
                not player["hasFlag"]
                and player["name"] not in self.flag_assignments
                and enemy_flags
            ):
                flag = enemy_flags[np.random.randint(len(enemy_flags))]
                self.flag_assignments[player["name"]] = (flag["posX"], flag["posY"])

    def _plan_player_move(
        self, player: Dict, opponents: List[Dict], my_targets: List[Tuple[int, int]]
    ) -> Optional[str]:
        """规划单个玩家的移动"""
        curr_pos = (player["posX"], player["posY"])

        # 确定目标
        if player["hasFlag"]:
            dest = my_targets[0] if my_targets else curr_pos
        elif player["name"] in self.flag_assignments:
            dest = self.flag_assignments[player["name"]]
        else:
            return None

        # 确定障碍物（在敌方领地避开对手）
        is_safe = self._is_on_my_side(curr_pos)
        obstacles = [] if is_safe else [(o["posX"], o["posY"]) for o in opponents]

        # 路径规划
        path = self._route_to(curr_pos, dest, obstacles)

        if len(path) > 1:
            return self._get_direction(curr_pos, path[1])
        return None

    def _is_on_my_side(self, pos: Tuple[int, int]) -> bool:
        """判断是否在己方领地"""
        if self.team == "L":
            return pos[0] < self.middle_line
        else:
            return pos[0] >= self.middle_line

    def _route_to(
        self,
        src: Tuple[int, int],
        dst: Tuple[int, int],
        extra_obstacles: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """BFS 路径规划"""
        obstacles = set(extra_obstacles) if extra_obstacles else set()
        queue = collections.deque([[src]])
        seen = {src}

        while queue:
            path = queue.popleft()
            curr = path[-1]
            if curr == dst:
                return path

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nxt = (curr[0] + dx, curr[1] + dy)
                if (
                    0 <= nxt[0] < self.width
                    and 0 <= nxt[1] < self.height
                    and nxt not in self.env._blocked_set
                    and nxt not in obstacles
                    and nxt not in seen
                ):
                    queue.append(path + [nxt])
                    seen.add(nxt)
        return []

    @staticmethod
    def _get_direction(
        curr: Tuple[int, int], next_pos: Tuple[int, int]
    ) -> Optional[str]:
        """计算移动方向"""
        dx = next_pos[0] - curr[0]
        dy = next_pos[1] - curr[1]
        if dx == 1:
            return "right"
        if dx == -1:
            return "left"
        if dy == 1:
            return "down"
        if dy == -1:
            return "up"
        return None
