# Capture the Flag

Your job is to implement your own algorithm (see `backend/server.cpp`) to control
a team to compete in "Capture the Flag" Game.

## Game Rules
Capture the Flag is a popular outdoor game where two teams compete in an open field.
Each team has a territory and a set of flags located within the territory. Each team's
goal is to collect flags of the opponent team and bring them back to the target area.

A player can tag the opponent team's player within his territory.
When tagged, the opponent team's player will be put into the prison.
The player will stay in the prison for a period of time, unless he is saved by
his teammate earlier.
The initial positions of flags, the target area and the prison are within the team's
territory.
A player can only pick the flags of the opponent team. That is, he cannot pick up
his team's flag and put it in a different place.

Our game has two teams: "L" and "R" team. The field is a rectangle area,
where the left half is "L" team's territory and the other is "R" team's.
There are obstacles and walls within the map.

![Capture The Flag Map](./map_example.png)


## Play

The game consists of 2 parts:
- __frontend/__: starts the game web server, written in Javascript. It optionally connects to 2 backend servers to move the players. You should NOT change the code, but you may read the code to understand how it generates the map and communicates with the backend.
- __backend/__: is the backend server which sends back instructions to frontend to move players. This is where you implement your algorithms. Note that in real competition, your server controls one team and the other is your opponent team's implementation.

To play it manually, you can use `w a s d` keys to control L team and `↑ ← ↓ →` keys to control R team. The keys override backend server's decisions. Note that the pressed keys move all players in one direction while your code can move each player independently.

Press SPACE KEY to start, pause or continue the game.

1. Install dependency
  ```
  brew update;
  brew install boost nlohmann-json
  ```
2. Compile server.cpp
  ```
  cd backend/;
  g++ -std=c++17 server.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lpthread -o server
  ```
3. Run server on port 8081 (can run on other ports)
  ```
  ./server 8081
  ```
4. Update `assets/remote_config.json` to the local port. Update `ws_url` with your port.
  ```
  {
    "teams": [
      { "name": "L", "ws_url": "ws://localhost:8080" },
      { "name": "R", "ws_url": "ws://localhost:8081" }
    ]
  }
  ```
5. Start frontend website
  ```
  cd frontend/;
  python3 -m http.server 8000
  ```
6. In your browser, open "http://localhost:8000/index.html" to play.
   - Press (Cmd + Option + I on macOS) to open DevTools
   - Go to the Network tab
   - Check ✅ “Disable cache” (upper-left toolbar) to ensure all your updated remote_config.json is loaded properly.

## Your Job

In `backend/server.cpp`, your need to implement `startGame(req)` and `planNextActions(req, ws)` functions.
  - `startGame(req)` is called once when the game starts. `req` contains the game information, such as map (e.g., height, width, obstacle positions) and team (e.g., name, number of players and number of flags).
  - `planNextActions(req, ws)` is called periodically to update you all the player and flags' information. You should use `ws` to send back the actions taken for your team player. The current implementation returns random actions for every team player.
  - `gameOver(req)` is called once the game finishes and a winner is determined. You may clean up the state for the next `startGame`.


## Write up (Important!)

You must submit a markdown writeup consisting of the following:
1. The top 3-5 "strategic" decisions to compete against the opponents? Explain the intuition,
   the core idea and the technical details (such as the data structure & algorithms).
2. Some interesting and funny moments when you are testing your implementations, or competing
   with your friends. What changes did you make after the test?


## Sample Team (in Python)

We provide a "not-so-dummy" python backend as a competitive opponent for you to develop your own algorithm.
DO NOT translate the python algorithm into C++ as your own implementation.
To run the python example,
```
python3 pick_closest_flag.py 8081
```
You'll need to `pip3 install` dependency, such as `asyncio`.
