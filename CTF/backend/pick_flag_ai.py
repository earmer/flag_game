import asyncio
import random
from lib.game_engine import GameMap, run_game_server
import threading


# 1. Initialize the global world model
world = GameMap(show_gap_in_msec=1000.0)
lock = threading.Lock()
last_updated_time = -1
update_threshold = 100
player_to_flag_assignments = {}

def start_game(req):
    global player_to_flag_assignments
    world.init(req)
    print("Start Game!!")
    player_to_flag_assignments = {}
    print(f"Game Started! Side: {'Left' if world.is_on_left(list(world.my_team_target)[0]) else 'Right'}")

def plan_next_actions(req):
    if not world.update(req):
        return

    global player_to_flag_assignments

    # Render the map
    # world.show(do_not_clear=False)

    my_players = world.list_players(mine=True, inPrison=False, hasFlag=None)
    opponents = world.list_players(mine=False, inPrison=False, hasFlag=None)
    enemy_flags = world.list_flags(mine=False, canPickup=True)
    my_targets = list(world.list_targets(mine=True))

    # 2. Logic: Assign flags to players without flags
    # We maintain the original logic of matching players to specific flag coordinates
    active_player_names = {p["name"] for p in my_players if not p["hasFlag"]}

    # Cleanup assignments for players captured or flags already taken
    player_to_flag_assignments = {
        name: pos for name, pos in player_to_flag_assignments.items()
        if name in active_player_names
    }

    if enemy_flags:
        for p in my_players:
            if not p["hasFlag"] and p["name"] not in player_to_flag_assignments:
                # Randomly assign one of the available enemy flags
                f = random.choice(enemy_flags)
                player_to_flag_assignments[p["name"]] = (f["posX"], f["posY"])

    # 3. Plan moves for each player
    player_moves = {}
    my_side_is_left = world.is_on_left(my_targets[0])

    for p in my_players:
        curr_pos = (p["posX"], p["posY"])

        # Determine Target: Either the assigned flag or the home target
        if p["hasFlag"]:
            dest = my_targets[0]
        elif p["name"] in player_to_flag_assignments:
            dest = player_to_flag_assignments[p["name"]]
        else:
            continue

        # Determine Obstacles: Avoid opponents if we are in enemy territory
        is_safe = world.is_on_left(curr_pos) == my_side_is_left
        blockers = [] if is_safe else [(o["posX"], o["posY"]) for o in opponents]

        # Calculate Path
        path = world.route_to(curr_pos, dest, extra_obstacles=blockers)

        if len(path) > 1:
            move = world.get_direction(curr_pos, path[1])
            player_moves[p["name"]] = move

    return player_moves

def game_over(req):
    print("Game Over!")
    world.show(force=True)


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
    except Exception as e:
        print(f"Server Stopped: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
