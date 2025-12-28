/*
* Asset from: https://kenney.nl/assets/pixel-platformer
*/
import ASSETS from '../assets.js';
import Player from '../gameObjects/Player.js';
import PlayerDirection from '../gameObjects/Player.js';
import Flag from '../gameObjects/Flag.js';

export class Game extends Phaser.Scene {
    constructor() {
        super('Game');
    }

    create() {
        // do nothing
        // we only do in create_after_preload()
    }

    create_after_preload() {
        // constant

        // this.NUM_PLAYERS = 3;
        // this.NUM_FLAGS = 9;
        // this.useRandomFlags = false;

        this.NUM_OBSTACLES_1 = 8;
        this.NUM_OBSTACLES_2 = 4;
        this.stageSent = false;

        this.initVariables();
        this.initGameUi();
        this.initAnimations();
        this.initInput();
        this.initMap();
        this.initBoundary();
        this.initTeams();
        this.initPhysics();
    }

    startOrPauseOrContinue() {
        if (!this.gameStarted) {
            this.startGame();
        } else {
            this.gamePaused = !this.gamePaused;
        }
    }

    update(time, delta) {
        if (!this.gameStarted || this.gamePaused) return;
        // move players
        let players_ready = 0, total_players = 0;

        let tick = (/** @type {string} */ can_go_next_tile) => {
            this.lteamPlayers.getChildren().forEach(player => {
                ++total_players;
                player.can_go_next_tile |= can_go_next_tile;
                player.update(time, delta);
                if (player.x == player.target.x && player.y == player.target.y)
                    ++players_ready;
            });
            this.rteamPlayers.getChildren().forEach(player => {
                ++total_players;
                player.can_go_next_tile |= can_go_next_tile;
                player.update(time, delta);
                if (player.x == player.target.x && player.y == player.target.y)
                    ++players_ready;
            });
        };
        tick(false);

        if (players_ready !== total_players) {
            this.stageSent = false;
            return;
        }

        tick(true);

        if (this.stageSent && time - this.lastSendTime < 600)
            return;
        setTimeout(() => {
            this.stageSent = true;

            // notify server backend
            let lteamPlayerStatus = this.lteamPlayers.getChildren().map(player => player.getStatus());
            let lteamFlagStatus = this.lteamFlags.getChildren().map(flag => flag.getStatus());
            let rteamPlayerStatus = this.rteamPlayers.getChildren().map(player => player.getStatus());
            let rteamFlagStatus = this.rteamFlags.getChildren().map(flag => flag.getStatus());

            // each team gets its own perspective
            if (this.lteamSocket && this.lteamSocket.readyState == WebSocket.OPEN) {
                const lteamPlayerStatus = this.lteamPlayers.getChildren().map(player => player.getStatus());
                const payload = {
                    action: "status",

                    time: time,
                    myteamPlayer: lteamPlayerStatus,
                    myteamFlag: lteamFlagStatus,
                    myteamScore: this.lteamState.score,

                    opponentPlayer: rteamPlayerStatus,
                    opponentFlag: rteamFlagStatus,
                    opponentScore: this.rteamState.score,
                }
                this.lteamSocket.send(JSON.stringify(payload));
            }

            if (this.rteamSocket && this.rteamSocket.readyState == WebSocket.OPEN) {
                const payload = {
                    action: "status",

                    time: time,
                    myteamPlayer: rteamPlayerStatus,
                    myteamFlag: rteamFlagStatus,
                    myteamScore: this.rteamState.score,

                    opponentPlayer: lteamPlayerStatus,
                    opponentFlag: lteamFlagStatus,
                    opponentScore: this.lteamState.score,
                }
                this.rteamSocket.send(JSON.stringify(payload));
            }
            this.lastSendTime = time;
        }, 50);
    }

    async preload() {
        // load team info from JSON file
        const resp = await fetch("game_config.json");
        const data = await resp.json();
        this.team_config = data.teams;
        this.team_servers = data.servers;
        this.initSockets();
        this.NUM_PLAYERS = data.setup.numPlayers;
        this.NUM_FLAGS = data.setup.numFlags;
        this.useRandomFlags = data.setup.useRandomFlags;
        this.create_after_preload();
    }

    updatePlayerInfo(teamName, data) {
        try {
            const actions = JSON.parse(data);
            Object.keys(actions.players).forEach(p => {
                let d = actions.players[p];
                console.assert(p.startsWith(teamName), `Invalid operation to control player ${p} for team ${teamName}`);
                console.assert(d == "up" || d == "down" || d == "left" || d == "right",
                    `Invalid operation to move player to direction ${d}`);
            });
            let teamPlayers = null;
            if (teamName === "L") {
                teamPlayers = this.lteamPlayers.getChildren();
            } else if (teamName === "R") {
                teamPlayers = this.rteamPlayers.getChildren();
            }
            // For each team player, we will set its direction.
            teamPlayers.forEach(player => {
                let remoteControl = actions.players[player.name];
                player.setRemoteControl(remoteControl);
            })
        } catch (e) {
            console.error("Invalid JSON from backend:", e, data);
        }
    }

    initSockets() {
        this.lteamSocket = null;
        this.rteamSocket = null;
        let lTeamWho = "-";
        let rTeamWho = "-"
        // connect to LTeam and RTeam backends
        for (let i = 0; i < this.team_config.length; ++i) {
            const team = this.team_config[i];
            if (team.name != "L" && team.name != "R") {
                console.log(`Unknown team ${team.name} found in remote_config.json. Skip.`);
                continue;
            }

            if (team["ws_url"] == null && (team["who"] == null || this.team_servers[team.who] == null)) {
                console.log(`Unknown server ${team["who"]} for ${team.name} found in remote_config.json. Skip.`);
                continue;
            }
            if (team["who"] != null) {
                if (team.name == "L") {
                    lTeamWho = team.who;
                } else {
                    rTeamWho = team.who;
                }
            }

            let ws = team["ws_url"] == null ? new WebSocket(this.team_servers[team.who]) : new WebSocket(team.ws_url);
            ws.onopen = () => console.log(`${team.name} connected`);
            ws.onmessage = (msg) => this.updatePlayerInfo(team.name, msg.data);
            ws.onerror = (err) => console.error("WebSocket error", err);
            if (team.name === "L") {
                this.lteamSocket = ws;
            }
            else if (team.name === "R") {
                this.rteamSocket = ws;
            }
        };
        this.lastSendTime = 0;

        this.lTeamWhoText = this.add.text(30, 60, `${lTeamWho}`, {
            fontFamily: 'Arial Black', fontSize: 36, color: '#ffffff',
            stroke: '#000000', strokeThickness: 8, align: 'left'
        })
            .setDepth(100);
        this.rTeamWhoText = this.add.text(this.scale.width - 450, 60, `${rTeamWho}`, {
            fontFamily: 'Arial Black', fontSize: 36, color: '#ffffff',
            stroke: '#000000', strokeThickness: 8, align: 'right'
        })
            .setDepth(100);
    }

    initVariables() {
        this.gameStarted = false;
        this.gamePaused = false;
        this.centerX = this.scale.width * 0.5;
        this.centerY = this.scale.height * 0.5;

        this.tileSize = 32; // width and height of a tile in pixels
        this.halfTileSize = this.tileSize * 0.5; // width and height of a tile in pixels

        this.mapHeight = (this.scale.height / this.tileSize) - 5 * 2; // height of the tile map (in tiles)
        this.mapWidth = (this.scale.width / this.tileSize) - 5 * 2; // width of the tile map (in tiles)
        this.mapX = this.centerX - (this.mapWidth * this.tileSize * 0.5); // x position of the top-left corner of the tile map
        this.mapY = this.centerY - (this.mapHeight * this.tileSize * 0.5); // y position of the top-left corner of the tile map

        // used to generate random background image
        this.backgroundTiles = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 44];
        this.targetTiles = [13, 14, 15, 25, 26, 27, 37, 38, 39];
        this.prisonTiles = [97, 98, 99, 109, 110, 111, 121, 122, 123];
        this.wallTiles = [45, 46, 47, 57, 59, 69, 70, 71];
        this.tree1Tiles = [6, 18, 30, 29, 28];
        this.tree2Tiles = [[4, 16], [5, 17]];

        // generate walls
        this.walls = [
            { x: 0, y: 0, tileId: 45 }, { x: this.mapWidth - 1, y: 0, tileId: 47 }, { x: 0, y: this.mapHeight - 1, tileId: 69 }, { x: this.mapWidth - 1, y: this.mapHeight - 1, tileId: 71 }
        ].concat(
            Array.from({ length: this.mapWidth - 2 }, (_, i) => ({ x: i + 1, y: 0, tileId: 46 }))
        ).concat(
            Array.from({ length: this.mapWidth - 2 }, (_, i) => ({ x: i + 1, y: this.mapHeight - 1, tileId: 46 }))
        ).concat(
            Array.from({ length: this.mapHeight - 2 }, (_, i) => ({ x: 0, y: i + 1, tileId: 57 }))
        ).concat(
            Array.from({ length: this.mapHeight - 2 }, (_, i) => ({ x: this.mapWidth - 1, y: i + 1, tileId: 59 }))
        );

        function notContains(xyArrays, x, y) {
            const ret = xyArrays.find(obj => (obj.x === x && obj.y === y));
            return ret == null;
        }

        // generate obstacles
        this.obstacles1 = [];
        for (let i = 0; i < this.NUM_OBSTACLES_1; ++i) {
            while (true) {
                const x = Phaser.Math.RND.integerInRange(4, this.mapWidth - 5);
                const y = Phaser.Math.RND.integerInRange(1, this.mapHeight - 2);
                if (notContains(this.obstacles1, x, y)) {
                    this.obstacles1.push({ x: x, y: y });
                    break;
                }
            }
        }
        this.obstacles2 = [];
        for (let i = 0; i < this.NUM_OBSTACLES_2; ++i) {
            while (true) {
                const x = Phaser.Math.RND.integerInRange(4, this.mapWidth - 5);
                const y = Phaser.Math.RND.integerInRange(1, this.mapHeight - 3);
                if (notContains(this.obstacles1, x, y)
                    && notContains(this.obstacles1, x, y + 1)
                    && notContains(this.obstacles2, x, y - 1)
                    && notContains(this.obstacles2, x, y)) {
                    this.obstacles2.push({ x: x, y: y });
                    break;
                }
            }
        }

        // Randomly generate flags
        let lFlags = [];
        for (let i = 0; i < this.NUM_FLAGS; ++i) {
            while (true) {
                const x = Phaser.Math.RND.integerInRange(2, this.mapWidth / 2 - 1);
                const y = Phaser.Math.RND.integerInRange(1, this.mapHeight - 3);
                if (notContains(this.obstacles1, x, y)
                    && notContains(this.obstacles2, x, y - 1)
                    && notContains(this.obstacles2, x, y)
                    && notContains(lFlags, x, y)) {
                    lFlags.push({ x: x, y: y });
                    break;
                }
            }
        }
        let rFlags = [];
        for (let i = 0; i < this.NUM_FLAGS; ++i) {
            while (true) {
                const x = Phaser.Math.RND.integerInRange(this.mapWidth / 2, this.mapWidth - 2);
                const y = Phaser.Math.RND.integerInRange(1, this.mapHeight - 3);
                if (notContains(this.obstacles1, x, y)
                    && notContains(this.obstacles2, x, y - 1)
                    && notContains(this.obstacles2, x, y)
                    && notContains(rFlags, x, y)) {

                    rFlags.push({ x: x, y: y });
                    break;
                }
            }
        }

        // generate flag and player position for LTeam and RTeam
        // left team
        this.lteamState = {
            score: 0,
            player_sprite_choice: 1,
            flags: this.useRandomFlags ? lFlags : Array.from({ length: this.NUM_FLAGS }, (_, i) => ({ x: 1, y: i + 1 })),
            players: this.useRandomFlags ? Array.from({ length: this.NUM_PLAYERS }, (_, i) => ({ x: 1, y: i + 1, name: "L" + i })) : Array.from({ length: this.NUM_PLAYERS }, (_, i) => ({ x: 2, y: i + 1, name: "L" + i })),
            target: this.create3x3grid(2, this.mapHeight / 2),
            prison: this.create3x3grid(2, this.mapHeight - 3),
        };

        // right team
        this.rteamState = {
            score: 0,
            player_sprite_choice: 4,
            // flags: Array.from({ length: this.NUM_FLAGS }, (_, i) => ({ x: this.mapWidth - 2, y: i + 1})),
            flags: this.useRandomFlags ? rFlags : Array.from({ length: this.NUM_FLAGS }, (_, i) => ({ x: this.mapWidth - 2, y: i + 1 })),
            players: this.useRandomFlags ? Array.from({ length: this.NUM_PLAYERS }, (_, i) => ({ x: this.mapWidth - 2, y: i + 1, name: "R" + i })) : Array.from({ length: this.NUM_PLAYERS }, (_, i) => ({ x: this.mapWidth - 3, y: i + 1, name: "R" + i })),
            target: this.create3x3grid(this.mapWidth - 3, this.mapHeight / 2),
            prison: this.create3x3grid(this.mapWidth - 3, this.mapHeight - 3),
        };

        this.map; // rference to tile map
        this.groundLayer; // used to create background layer of tile map
        this.levelLayer; // reference to level layer of tile map
    }

    initGameUi() {
        // Create tutorial text
        this.tutorialText = this.add.text(this.centerX, this.centerY, 'Arrow keys to move!\nPress Spacebar to Start', {
            fontFamily: 'Arial Black', fontSize: 48, color: '#ffffff',
            stroke: '#000000', strokeThickness: 8,
            align: 'center'
        })
            .setOrigin(0.5)
            .setDepth(100);

        // Create score text
        this.lScoreText = this.add.text(30, 20, 'LTeam #Flags: 0', {
            fontFamily: 'Arial Black', fontSize: 36, color: '#ffffff',
            stroke: '#000000', strokeThickness: 8, align: 'left'
        })
            .setDepth(100);
        this.rScoreText = this.add.text(this.scale.width - 450, 20, 'RTeam #Flags: 0', {
            fontFamily: 'Arial Black', fontSize: 36, color: '#ffffff',
            stroke: '#000000', strokeThickness: 8, align: 'right'
        })
            .setDepth(100);

        // Create game over text
        this.gameOverText = this.add.text(this.scale.width * 0.5, this.scale.height * 0.5, 'Game Over', {
            fontFamily: 'Arial Black', fontSize: 64, color: '#ffffff',
            stroke: '#000000', strokeThickness: 8,
            align: 'center'
        })
            .setOrigin(0.5)
            .setDepth(100)
            .setVisible(false);
    }

    initAnimations() {
        const flag_choices = ["characters", "characters_L_flag", "characters_R_flag"];
        const dir_choices = ["left", "down", "up", "right"]

        for (let k = 0; k < 3; ++k) {
            for (let i = 1; i <= 6; ++i) {
                for (let j = 0; j < 4; ++j) {
                    const key = "player" + i + "-" + flag_choices[k] + "-" + dir_choices[j];
                    const config = { frames: [(i - 1) * 12 + j, (i - 1) * 12 + j + 4, (i - 1) * 12 + j + 8] };
                    this.anims.create({
                        key: key,
                        frames: this.anims.generateFrameNumbers(flag_choices[k], config),
                        frameRate: 10,
                        repeat: 0
                    });
                }
            }
        }
    }

    initPhysics() {
        this.physics.add.overlap(this.lteamPlayers, this.rteamPlayers, this.hitPlayer, null, this);

        this.physics.add.overlap(this.lteamPlayers, this.rteamFlags, this.collectFlag, null, this);
        this.physics.add.overlap(this.rteamPlayers, this.lteamFlags, this.collectFlag, null, this);

        this.physics.add.overlap(this.lteamPlayers, this.lteamTargetZone, this.dropFlag, null, this);
        this.physics.add.overlap(this.rteamPlayers, this.rteamTargetZone, this.dropFlag, null, this);

        this.physics.add.overlap(this.lteamPlayers, this.lteamPrisonZone, this.freePlayer, null, this);
        this.physics.add.overlap(this.rteamPlayers, this.rteamPrisonZone, this.freePlayer, null, this);
    }

    initTeams() {
        // init L Team
        this.lteamFlags = this.add.group();
        this.lteamPlayers = this.add.group();

        this.lteamState.flags.forEach(flag => {
            const flagObj = new Flag(this, flag.x, flag.y, "L", true);
            this.lteamFlags.add(flagObj);
        });
        this.lteamState.players.forEach(player => {
            const playerObj = new Player(this, player.name, player.x, player.y, "L", this.lteamState.player_sprite_choice, true);
            this.lteamPlayers.add(playerObj);
        });
        // this.lteamTargetZone = this.add.zone(this.lteamState.target[0].x * this.tileSize, this.lteamState.target[0].y * this.tileSize, 3 * this.tileSize, 3 * this.tileSize);
        this.lteamTargetZone = this.add.zone(
            this.mapX + (this.lteamState.target[0].x * this.tileSize + 1.5 * this.tileSize),
            this.mapY + (this.lteamState.target[0].y * this.tileSize + 1.5 * this.tileSize),
            3 * this.tileSize, 3 * this.tileSize);
        this.physics.add.existing(this.lteamTargetZone);
        this.lteamTargetZone.body.setAllowGravity(false);
        this.lteamTargetZone.body.setImmovable(true);

        // this.lteamPrisonZone = this.add.zone(this.lteamState.prison[0].x * this.tileSize, this.lteamState.prison[0].y * this.tileSize, 3 * this.tileSize, 3 * this.tileSize);
        this.lteamPrisonZone = this.add.zone(
            this.mapX + (this.lteamState.prison[0].x * this.tileSize + 1.5 * this.tileSize),
            this.mapY + (this.lteamState.prison[0].y * this.tileSize + 1.5 * this.tileSize),
            3 * this.tileSize, 3 * this.tileSize);
        this.physics.add.existing(this.lteamPrisonZone);
        this.lteamPrisonZone.body.setAllowGravity(false);
        this.lteamPrisonZone.body.setImmovable(true);

        // init R Team
        this.rteamFlags = this.add.group();
        this.rteamPlayers = this.add.group();

        this.rteamState.flags.forEach(flag => {
            const flagObj = new Flag(this, flag.x, flag.y, "R", true);
            this.rteamFlags.add(flagObj);
        });
        this.rteamState.players.forEach(player => {
            const playerObj = new Player(this, player.name, player.x, player.y, "R", this.rteamState.player_sprite_choice, false);
            this.rteamPlayers.add(playerObj);
        });

        this.rteamTargetZone = this.add.zone(
            this.mapX + (this.rteamState.target[0].x * this.tileSize + 1.5 * this.tileSize),
            this.mapY + (this.rteamState.target[0].y * this.tileSize + 1.5 * this.tileSize),
            3 * this.tileSize, 3 * this.tileSize);
        this.physics.add.existing(this.rteamTargetZone);
        this.rteamTargetZone.body.setAllowGravity(false);
        this.rteamTargetZone.body.setImmovable(true);

        this.rteamPrisonZone = this.add.zone(
            this.mapX + (this.rteamState.prison[0].x * this.tileSize + 1.5 * this.tileSize),
            this.mapY + (this.rteamState.prison[0].y * this.tileSize + 1.5 * this.tileSize),
            3 * this.tileSize, 3 * this.tileSize);
        this.physics.add.existing(this.rteamPrisonZone);
        this.rteamPrisonZone.body.setAllowGravity(false);
        this.rteamPrisonZone.body.setImmovable(true);
    }

    initInput() {
        this.cursors = this.input.keyboard.createCursorKeys();
        this.awsd_keys = this.input.keyboard.addKeys({
            up: Phaser.Input.Keyboard.KeyCodes.W,
            left: Phaser.Input.Keyboard.KeyCodes.A,
            down: Phaser.Input.Keyboard.KeyCodes.S,
            right: Phaser.Input.Keyboard.KeyCodes.D
        });

        // check for spacebar press only once
        // this.cursors.space.once('down', (key, event) => {
        //     this.startGame();
        // });

        this.cursors.space.on('down', (key, event) => {
            this.startOrPauseOrContinue();
        });
    }

    // create tile map data
    initMap() {
        this.map = this.make.tilemap({ key: ASSETS.tilemapTiledJSON.map.key });
        const tileset = this.map.addTilesetImage(ASSETS.spritesheet.tiles.key);

        // create background layer, randomly pick the tiles
        this.groundLayer = this.map.createBlankLayer('ground', tileset, this.mapX, this.mapY);
        for (let y = 0; y < this.mapHeight; y++) {
            for (let x = 0; x < this.mapWidth; x++) {
                // randomly choose a tile id from this.tiles
                // weightedPick favours items earlier in the array
                const tileIndex = Phaser.Math.RND.pick(this.backgroundTiles);
                this.groundLayer.putTileAt(tileIndex, x, y);
            }
        }

        // create level layer to show game level elements
        this.levelLayer = this.map.createBlankLayer('level', tileset, this.mapX, this.mapY);
        this.levelLayer.fill(0, 0, 0, this.mapWidth, this.mapHeight);
        // show prison tiles
        for (let i = 0; i < this.lteamState.prison.length; ++i) {
            const prison = this.lteamState.prison[i];
            const tile = this.levelLayer.getTileAt(prison.x, prison.y);
            tile.index = this.prisonTiles[i];
        }
        for (let i = 0; i < this.rteamState.prison.length; ++i) {
            const prison = this.rteamState.prison[i];
            const tile = this.levelLayer.getTileAt(prison.x, prison.y);
            tile.index = this.prisonTiles[i];
        }
        for (let i = 0; i < this.lteamState.target.length; ++i) {
            const target = this.lteamState.target[i];
            const tile = this.levelLayer.getTileAt(target.x, target.y);
            tile.index = this.targetTiles[i];
        }
        for (let i = 0; i < this.rteamState.target.length; ++i) {
            const target = this.rteamState.target[i];
            const tile = this.levelLayer.getTileAt(target.x, target.y);
            tile.index = this.targetTiles[i];
        }

        // create wall
        for (let i = 0; i < this.walls.length; ++i) {
            const wall = this.walls[i];
            const tile = this.levelLayer.getTileAt(wall.x, wall.y);
            tile.index = wall.tileId;
            const collisionId = (wall.x - 1) * this.mapWidth + wall.y;
            this.map.setCollision(collisionId);
        }

        // create obstacles
        for (let i = 0; i < this.obstacles1.length; ++i) {
            const obs = this.obstacles1[i];
            const tile = this.levelLayer.getTileAt(obs.x, obs.y);
            tile.index = Phaser.Math.RND.pick(this.tree1Tiles);
            const collisionId = (obs.x - 1) * this.mapWidth + obs.y;
            this.map.setCollision(collisionId);
        }
        for (let i = 0; i < this.obstacles2.length; ++i) {
            const obs = this.obstacles2[i];
            const treeTile = Phaser.Math.RND.pick(this.tree2Tiles);
            const tile1 = this.levelLayer.getTileAt(obs.x, obs.y);
            tile1.index = treeTile[0];
            const tile2 = this.levelLayer.getTileAt(obs.x, obs.y + 1);
            tile2.index = treeTile[1];
            const collisionId = (obs.x - 1) * this.mapWidth + obs.y;
            this.map.setCollision(collisionId, collisionId + 1);
        }
    }

    // create a thin black line in the middle
    initBoundary() {
        const startY = this.centerY - this.mapHeight * this.tileSize / 2;
        const endY = this.centerY + this.mapHeight * this.tileSize / 2;
        this.add.line(0, 0, this.centerX, startY, this.centerX, endY, 0x000000)
            .setOrigin(0, 0)
            .setLineWidth(1);
    }

    startGame() {
        this.gameStarted = true;
        this.tutorialText.setVisible(false);
        const mapPayload = {
            "width": this.mapWidth,
            "height": this.mapHeight,
            "walls": this.walls.map(w => { return { x: w.x, y: w.y } }),
            "obstacles": this.obstacles1.concat(this.obstacles2).concat(
                this.obstacles2.map(w => { return { "x": w.x, "y": w.y + 1 } })
            ),
        };

        if (this.lteamSocket && this.lteamSocket.readyState == WebSocket.OPEN) {
            const payload = {
                "action": "init",
                "map": mapPayload,
                "numPlayers": this.NUM_PLAYERS,
                "numFlags": this.NUM_FLAGS,
                "myteamName": "L",
                // this is where you will be sent to, if you were caught by opponent
                "myteamPrison": this.lteamState.prison,
                // this is where you will drop the flags
                "myteamTarget": this.lteamState.target,
                "opponentPrison": this.rteamState.prison,
                "opponentTarget": this.rteamState.target,
            }
            this.lteamSocket.send(JSON.stringify(payload));
        }

        if (this.rteamSocket && this.rteamSocket.readyState == WebSocket.OPEN) {
            const payload = {
                "action": "init",
                "map": mapPayload,
                "numPlayers": this.NUM_PLAYERS,
                "numFlags": this.NUM_FLAGS,
                "myteamName": "R",
                // this is where you will send the opponent to prison
                "myteamPrison": this.rteamState.prison,
                // this is where you will drop the flags
                "myteamTarget": this.rteamState.target,
                "opponentPrison": this.lteamState.prison,
                "opponentTarget": this.lteamState.target,
            }
            this.rteamSocket.send(JSON.stringify(payload));
        }
    }

    hitPlayer(player1, player2) {
        // does not matter if player1 is not in the same team as player2
        if (player1.team === player2.team) {
            return;
        }
        if (player1.inPrison || player2.inPrison) {
            return;
        }

        // When collision happens around the center, we use the middle X
        let playerCenterX = (player1.x + player2.x) / 2;

        // in L team's side
        if (playerCenterX < this.centerX) {
            // find the prison tile to send the R team player
            // If the flag is held by the R team player, drop the flag at where it was caught.
            const spot = this.findAvailablePrisonTile(this.rteamPlayers.getChildren(), this.rteamState.prison);
            const caughtPlayer = player1.team === "R" ? player1 : player2;
            if (caughtPlayer.hasFlag) {
                const tile = this.getTileAt(caughtPlayer.x, caughtPlayer.y);
                const flag = new Flag(this, tile.x, tile.y, "L", true);
                this.lteamFlags.add(flag);
                caughtPlayer.hasFlag = false;
            }
            caughtPlayer.toPrison(spot.x, spot.y);
        } else {
            const spot = this.findAvailablePrisonTile(this.lteamPlayers.getChildren(), this.lteamState.prison);
            const caughtPlayer = player1.team === "L" ? player1 : player2;
            if (caughtPlayer.hasFlag) {
                const tile = this.getTileAt(caughtPlayer.x, caughtPlayer.y);
                const flag = new Flag(this, tile.x, tile.y, "R", true);
                this.rteamFlags.add(flag);
                caughtPlayer.hasFlag = false;
            }
            caughtPlayer.toPrison(spot.x, spot.y);
        }
    }

    findAvailablePrisonTile(players, prisons) {
        for (let i = 0; i < prisons.length; ++i) {
            let isAvailable = true;
            for (let j = 0; j < players.length; ++j) {
                if (!players[j].inPrison) {
                    continue;
                }
                const tile = this.getTileAt(players[j].x, players[j].y);
                if (tile.x == prisons[i].x && tile.y == prisons[i].y) {
                    isAvailable = false;
                    break;
                }
            }
            if (isAvailable) {
                return { x: prisons[i].x, y: prisons[i].y };
            }
        }
    }

    dropFlag(player) {
        if (!player.hasFlag) {
            return;
        }
        player.dropFlag();
        if (player.team == "L") {
            const spot = this.findAvailableFlagTile(this.rteamFlags.getChildren(), this.lteamState.target);
            const flag = new Flag(this, spot.x, spot.y, "R", false);
            this.rteamFlags.add(flag);
            this.updateTeamScore("L");
        } else {
            const spot = this.findAvailableFlagTile(this.lteamFlags.getChildren(), this.rteamState.target);
            const flag = new Flag(this, spot.x, spot.y, "L", false);
            this.lteamFlags.add(flag);
            this.updateTeamScore("R");
        }
    }

    findAvailableFlagTile(flags, targets) {
        for (let i = 0; i < targets.length; ++i) {
            let isAvailable = true;
            for (let j = 0; j < flags.length; ++j) {
                if (flags[j].canPickup) {
                    continue;
                }
                const tile = this.getTileAt(flags[j].x, flags[j].y);
                if (tile.x == targets[i].x && tile.y == targets[i].y) {
                    isAvailable = false;
                    break;
                }
            }
            if (isAvailable) {
                return { x: targets[i].x, y: targets[i].y };
            }
        }
    }

    collectFlag(player, flag) {
        // cannot collect flag from my team
        if (player.team == flag.team) {
            return;
        }
        if (player.inPrison) {
            return;
        }
        // a player cannot collect >1 flag
        if (player.hasFlag) {
            return;
        }
        // cannot collect flag that cannot be collected
        if (!flag.canPickup) {
            return;
        }
        flag.collect();
        player.collectFlag();
    }

    removeFlagItem(flag) {
        if (flag.team == "L") {
            this.lteamFlags.remove(flag, true, true);
        } else if (flag.team == "R") {
            this.rteamFlags.remove(flag, true, true);
        }
    }

    freePlayer(player) {
        // player sent to prison cannot free others
        if (player.inPrison) {
            return;
        }
        if (player.team == "L") {
            this.lteamPlayers.getChildren().forEach(player => {
                if (player.inPrison) { player.inPrison = false; }
            })
        } else {
            this.rteamPlayers.getChildren().forEach(player => {
                if (player.inPrison) { player.inPrison = false; }
            })
        }
    }

    updateTeamScore(team) {
        if (team == "L") {
            ++this.lteamState.score;
            this.lScoreText.setText(`LTeam #Flags: ${this.lteamState.score}`);
            if (this.lteamState.score == this.NUM_FLAGS) {
                this.GameOver(team);
            }
        } else if (team == "R") {
            ++this.rteamState.score;
            this.rScoreText.setText(`RTeam #Flags: ${this.rteamState.score}`);
            if (this.rteamState.score == this.NUM_FLAGS) {
                this.GameOver(team);
            }
        }
    }

    getMapOffset() {
        return {
            x: this.mapX + this.halfTileSize,
            y: this.mapY + this.halfTileSize,
            width: this.mapWidth,
            height: this.mapHeight,
            tileSize: this.tileSize
        }
    }

    getTileAt(x, y) {
        const tile = this.levelLayer.getTileAtWorldXY(x, y, true);
        return tile;
    }

    isWall(x, y) {
        const tile = this.levelLayer.getTileAtWorldXY(x, y, true);
        return this.wallTiles.indexOf(tile.index) >= 0 ||
            this.tree1Tiles.indexOf(tile.index) >= 0 ||
            this.tree2Tiles[0].indexOf(tile.index) >= 0 ||
            this.tree2Tiles[1].indexOf(tile.index) >= 0
            ;
    }

    // return a 3x3 grid from x, y
    create3x3grid(x, y) {
        return [
            { x: x - 1, y: y - 1 }, { x: x, y: y - 1 }, { x: x + 1, y: y - 1 },
            { x: x - 1, y: y }, { x: x, y: y }, { x: x + 1, y: y },
            { x: x - 1, y: y + 1 }, { x: x, y: y + 1 }, { x: x + 1, y: y + 1 },
        ]
    }

    GameOver(team) {
        this.gameStarted = false;
        this.gameOverText.setText(team + "Team Won!")
        this.gameOverText.setVisible(true);

        if (this.lteamSocket && this.lteamSocket.readyState == WebSocket.OPEN) {
            const payload = {
                action: "finished",
                myteamScore: this.lteamState.score,
                opponentScore: this.rteamState.score,
            }
            this.lteamSocket.send(JSON.stringify(payload));
        }

        if (this.rteamSocket && this.rteamSocket.readyState == WebSocket.OPEN) {
            const payload = {
                action: "finished",
                myteamScore: this.rteamState.score,
                opponentScore: this.lteamState.score,
            }
            this.rteamSocket.send(JSON.stringify(payload));
        }
    }
}
