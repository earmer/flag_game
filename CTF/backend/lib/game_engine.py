import asyncio
from abc import ABC
import collections
import json
import threading
import random
import websockets

from IPython.display import clear_output

class GameMap(ABC):
    def __init__(self, show_gap_in_msec = 1000.0):
        """
        @show_gap: how many milliseconds when show can be invoked 
        """
        self.width = 0
        self.height = 0
        self.middle_line = self.width / 2
        self.walls = set()
        self.players = []
        self.flags = []
        self.current_time = 0.0
        self.next_show_time = -1.0
        self.my_team_name = ""
        self.show_gap_in_msec = show_gap_in_msec

    def init(self, req):
        map_data = req["map"]
        self.current_time = 0.0
        self.next_show_time = -1.0
        self.width = map_data["width"]
        self.height = map_data["height"]
        self.middle_line = self.width / 2
        self.my_team_name = req.get("myteamName", "")
        
        self.walls = {(w["x"], w["y"]) for w in (map_data.get("walls", []) + map_data.get("obstacles", []))}
        self.my_team_prison = {(w["x"], w["y"]) for w in (req.get("myteamPrison", []))}
        self.opponent_team_prison = {(w["x"], w["y"]) for w in (req.get("opponentPrison", []))}
        self.my_team_target = {(w["x"], w["y"]) for w in (req.get("myteamTarget", []))}
        self.opponent_team_target = {(w["x"], w["y"]) for w in (req.get("opponentTarget", []))}


    def update(self, req):
        if req["time"] < self.current_time:
            return False
        self.current_time = req["time"]
        self.players = []
        # Combine and tag players
        for p in req.get("myteamPlayer", []):
            p['mine'] = True
            self.players.append(p)
        for p in req.get("opponentPlayer", []):
            p['mine'] = False
            self.players.append(p)
            
        self.flags = []
        for f in req.get("myteamFlag", []):
            f['mine'] = True
            self.flags.append(f)
        for f in req.get("opponentFlag", []):
            f['mine'] = False
            self.flags.append(f)
        return True

    def list_players(self, mine, inPrison, hasFlag):
        """
        mine: True or False. If True, return players on my side, otherwise return opponent players;
        inPrison: True or False or None. If True, return players that are in prison; if false, return players that can move around freely; if none, return all of them.
        hasFlag: True or False or None. If True, return players that have flags; if false, return players that do not have flags; if none, return all of them.
        """
        return [p for p in self.players if p['mine'] == mine and (inPrison == None or p["inPrison"] == inPrison) and (hasFlag == None or p["hasFlag"] == hasFlag)]
    
    def list_flags(self, mine, canPickup):
        """
        mine: True or False. If True, return flags on my side (i.e., flags I should protect), otherwise return opponent's flags (i.e., flags I should pick up);
        canPickup: True or False or None. If True, return flags that can be picked up; if false, return flags that are already placed in my camp; if none, return all of them.
        """

        return [f for f in self.flags if f['mine'] == mine and (canPickup == None or f["canPickup"] == canPickup)]

    def list_targets(self, mine):
        if mine:
            return self.my_team_target
        else:
            return self.opponent_team_target

    def list_prisons(self, mine):
        if mine:
            return self.my_team_prison
        else:
            return self.opponent_team_prison

    def get_object_at_XY(self, x, y, flag_over_target=False, player_over_prison=False):
        """
        flags could overlap with targets, and players could overlap with prisons.
        These are controlled by @flag_over_target and @player_over_prison.
        """
        if (x, y) in self.walls: return "██ "
        if not player_over_prison:
            if (x, y) in self.my_team_prison: return "PP "
            if (x, y) in self.opponent_team_prison: return "PP "
        if not flag_over_target:
            if (x, y) in self.my_team_target: return "TT "
            if (x, y) in self.opponent_team_target: return "TT "

        # Check Players
        for p in self.players:
            if p["posX"] == x and p["posY"] == y:
                return p["name"] + " "
        
        # Check Flags
        for f in self.flags:
            if f["posX"] == x and f["posY"] == y:
                if f['mine']:
                    team = self.my_team_name
                else:
                    team = "R" if self.my_team_name == "L" else "L"
                return f"{team}F "

        if player_over_prison:
            if (x, y) in self.my_team_prison: return "PP "
            if (x, y) in self.opponent_team_prison: return "PP "
        if flag_over_target:
            if (x, y) in self.my_team_target: return "TT "
            if (x, y) in self.opponent_team_target: return "TT "
                
        return " . "

    def show(self, force=False, do_not_clear=False, flag_over_target=False, player_over_prison=False):
        """
        Prints the grid with L1, R2, LF, RF labels.
        flags could overlap with targets, and players could overlap with prisons.
        These are controlled by @flag_over_target and @player_over_prison.
        """
        if self.current_time < self.next_show_time and (not force):
            return
        if not do_not_clear:
            clear_output()
        header = "   " + " ".join([f"{x:2}" for x in range(self.width)])
        print(header)
        for y in range(self.height):
            row = f"{y:2} "
            for x in range(self.width):
                row += self.get_object_at_XY(x, y, flag_over_target, player_over_prison)
            print(row)
        self.next_show_time = self.current_time + self.show_gap_in_msec

    def route_to(self, srcXY, dstXY, extra_obstacles=None):
        extras = set(extra_obstacles) if extra_obstacles else set()
        queue = collections.deque([[srcXY]])
        seen = {srcXY}
        
        while queue:
            path = queue.popleft()
            curr = path[-1]
            if curr == dstXY:
                return path

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Up, Down, Left, Right
                nxt = (curr[0] + dx, curr[1] + dy)
                if (0 <= nxt[0] < self.width and 0 <= nxt[1] < self.height and 
                    nxt not in self.walls and nxt not in extras and nxt not in seen):
                    queue.append(path + [nxt])
                    seen.add(nxt)
        return []

    def is_on_left(self, srcXY):
        return srcXY[0] < self.middle_line
    
    @staticmethod
    def get_direction(currentXY, nextXY):
        """Helper to convert two points into a direction string."""
        dx = nextXY[0] - currentXY[0]
        dy = nextXY[1] - currentXY[1]
        if dx == 1: return "right"
        if dx == -1: return "left"
        if dy == 1: return "down"
        if dy == -1: return "up"
        return ""


async def run_game_server(port, start_fn, plan_fn, end_fn):
    lock = threading.Lock()
    async def handler(websocket):
        print(f"Connected on port {port}")
        async for msg in websocket:
            req = json.loads(msg)
            action = req.get("action")
            if action == "init":
                with lock:
                    start_fn(req)
            elif action == "status":
                with lock:
                    moves = plan_fn(req)
                    await websocket.send(json.dumps({"players": moves}))
            elif action == "finished":
                with lock:
                    end_fn(req)

    print(f"Starting server on port {port}...")
    async with websockets.serve(handler, "0.0.0.0", port):
        await asyncio.Future()
