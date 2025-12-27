import asyncio
import json
import random
import traceback
import websockets

ACTIONS = ["up", "down", "left", "right", ""]

class GameMap:

    EMPTY = 0
    OBSTACLE = -1
    FLAG = 2

    ACTIONS = ["up", "down", "left", "right", ""]
    ACTIONS_IN_MOVE = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def __init__(self, map_json):
        self.w = map_json["width"]
        self.h = map_json["height"]
        self.grids = [[GameMap.EMPTY for _ in range(0, self.h)] for _ in range(0, self.w)]
        self.obstacles = map_json["walls"] + map_json["obstacles"]
        for o in self.obstacles:
            self.grids[o["x"]][o["y"]] = GameMap.OBSTACLE

    def show_map(self):
        for y in range(0, self.h):
            for x in range(0, self.w):
                print(self.grids[x][y], end=" ")
            print("")


    def find_closest_goal_from_pos(self, pos_x, pos_y, goals, blockers):
        """
        Find the closest goal(@goal_x, @goal_y) in @goals from (@pos_x, @pos_y).
        Return the moving direction for (@pos_x, @pos_y).
        If none is reachable, return "".
        """
        goals_pos = {(g["x"], g["y"]) for g in goals}
        blocker_pos = set()
        for (bx, by) in blockers:
            for (dx, dy) in GameMap.ACTIONS_IN_MOVE:
                x = bx + dx
                y = by + dy
                if (x, y) not in goals_pos:
                    blocker_pos.add((bx, by))

        # visited != -1 means the grid is reached from (pos_x, pos_y)
        # it stores the previous grid's direction to reach the current grid
        visited = [[-1 for _ in range(0, self.h)] for _ in range(0, self.w)]
        bfs = [(pos_x, pos_y)]
        st = 0
        goal_x = -1
        goal_y = -1
        while st < len(bfs):
            for d, (dx, dy) in enumerate(GameMap.ACTIONS_IN_MOVE):
                x = bfs[st][0] + dx
                y = bfs[st][1] + dy
                if (self.grids[x][y] != GameMap.OBSTACLE and
                      ((x, y) not in blocker_pos) and
                      visited[x][y] < 0):
                    visited[x][y] = d
                    bfs.append((x, y))
                    if (x, y) in goals_pos:
                        goal_x = x
                        goal_y = y
                        break
            st = st + 1

        # we need to find the very first direction taken by (pos_x, pos_y) to reach (goal_x, goal_y)
        if goal_x < 0 or goal_y < 0:
            return GameMap.ACTIONS[-1]
        cur_x = goal_x
        cur_y = goal_y
        first_direction = -1
        while cur_x != pos_x or cur_y != pos_y:
            first_direction = visited[cur_x][cur_y]
            cur_x = cur_x - GameMap.ACTIONS_IN_MOVE[first_direction][0]
            cur_y = cur_y - GameMap.ACTIONS_IN_MOVE[first_direction][1]

        return GameMap.ACTIONS[first_direction]



class Game:
    def __init__(self):
        self.map = None
        self.team_name = None
        self.team_target = None
        self.num_flags = 0
        self.num_players = 0
        self.game_started = False
        self.player_to_flag_assignments = None

    def startGame(self, game_json):
        self.map = GameMap(game_json["map"])
        self.team_name = game_json["myteamName"]
        self.team_target = game_json["myteamTarget"]
        self.num_flags = game_json["numFlags"]
        self.num_players = game_json["numPlayers"]
        self.game_started =  True
        self.middle_line = self.map.w / 2;
        self.my_team_on_the_left = self.team_target[0]['x'] < self.middle_line;
        # playerName -> (flagX, flagY)
        self.player_to_flag_assignments = dict()

    def endGame(self, game_json):
        self.map = None
        self.team_name = None
        self.team_target = None
        self.num_flags = 0
        self.num_players = 0
        self.game_started = False
        self.player_to_flag_assignments = None

    def is_player_safe(self, player):
        return (player["posX"] < self.middle_line) == self.my_team_on_the_left


    def assign_flags_to_players(self, players, flags):
        """
        assign all pickable flags to all eligible players (i.e., !hasFlag and !inPrison)
        """
        # remove the prison player and player with flags
        for p in players:
            if (p["hasFlag"] or p["inPrison"]) and p["name"] in self.player_to_flag_assignments:
                del self.player_to_flag_assignments[p["name"]]
        pickable_flags = {
            (f["posX"], f["posY"]): False for f in flags if f["canPickup"]
        }
        players_wo_flag = set([p["name"] for p in players if (not p["hasFlag"]) and (not p["inPrison"])])
        for p, f in self.player_to_flag_assignments.items():
            if f in pickable_flags:
                pickable_flags[f] = True
                players_wo_flag.remove(p)

        # randomly match unassigned flags and players
        flags_wo_player = [f for (f, m) in pickable_flags.items() if not m]
        if len(flags_wo_player) > 0:
            for i, p in enumerate(players_wo_flag):
                self.player_to_flag_assignments[p] = flags_wo_player[i % len(flags_wo_player)]
        elif len(pickable_flags) > 0:
            pickable_flags_list = list(pickable_flags)
            for i, p in enumerate(players_wo_flag):
                self.player_to_flag_assignments[p] = random.choice(pickable_flags_list)


    def find_next_move(self, player, opponents):
        if player["inPrison"]:
            return ""

        blockers = [] if self.is_player_safe(player) else [(o["posX"], o["posY"]) for o in opponents]
        if player["hasFlag"]:
            return self.map.find_closest_goal_from_pos(
                player["posX"], player["posY"],
                [GAME.team_target[0]],
                blockers)

        if player["name"] not in self.player_to_flag_assignments:
            return ""

        flag = self.player_to_flag_assignments[player["name"]]
        return self.map.find_closest_goal_from_pos(
            player["posX"], player["posY"],
            [{"x": flag[0], "y": flag[1]}],
            blockers
        )


# SINGLETON
GAME = Game()

async def startGame(req):
    print("Start Game")

    global GAME
    GAME.startGame(req)


async def planNextActions(req, websocket):
    global GAME

    player_moves = dict()
    players = req.get("myteamPlayer", [])
    opponents = req.get("opponentPlayer", [])
    flags = req.get("opponentFlag", [])
    GAME.assign_flags_to_players(players, flags)

    for p in players:
        action = GAME.find_next_move(p, opponents)
        if action != "":
            player_moves[p["name"]] = action

    result = {"players": player_moves}
    await websocket.send(json.dumps(result))


async def gameOver(req):
    global GAME
    GAME.endGame(req)


async def handle_client(websocket):
    print("New session started")

    try:
        async for msg in websocket:
            try:
                req = json.loads(msg)

                if req.get("action") == "status":
                    await planNextActions(req, websocket)

                elif req.get("action") == "init":
                    await startGame(req)

                elif req.get("action") == "finished":
                    await gameOver(req)

            except json.JSONDecodeError:
                print("JSON parse error")
                await websocket.send(json.dumps({"error": "invalid json"}))

    except websockets.exceptions.ConnectionClosedOK:
        print("Client closed connection normally")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()


async def main():
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <port>")
        print(f"Example: python3 {sys.argv[0]} 8080")
        return

    port = int(sys.argv[1])
    print(f"AI backend running on port {port} ...")

    async with websockets.serve(handle_client, "0.0.0.0", port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
