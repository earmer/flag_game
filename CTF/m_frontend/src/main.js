import { Boot } from './scenes/Boot.js';
import { Preloader } from './scenes/Preloader.js';
import { Game } from './scenes/Game.js';
import { GameOver } from './scenes/GameOver.js';

const GAME_COUNT = 9;
const GRID_COLUMNS = 3;
// REMEMBER to keep consistent with tilemap.json
const GAME_WIDTH = (20 + 10) * 32;
const GAME_HEIGHT = (20 + 10) * 32;

const baseConfig = {
    type: Phaser.AUTO,
    width: GAME_WIDTH,
    height: GAME_HEIGHT,
    backgroundColor: '#2d3436',
    scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH,
    },
    physics: {
        default: 'arcade',
        arcade: {
            debug: false,
            gravity: { y: 0 }
        }
    },
    scene: [
        Boot,
        Preloader,
        Game,
        GameOver
    ]
};

async function boot() {
    const resp = await fetch("game_config.json");
    const gameConfigData = await resp.json();
    const grid = document.getElementById('games-grid');
    grid.style.setProperty('--grid-cols', GRID_COLUMNS);

    for (let i = 0; i < GAME_COUNT; i++) {
        const container = document.createElement('div');
        container.className = 'game-cell';
        container.id = `game-container-${i}`;
        grid.appendChild(container);

        const config = {
            ...baseConfig,
            parent: container.id,
            gameConfigData,
            enableKeyboard: false,
            autoStart: true,
        };
        new Phaser.Game(config);
    }
}

boot();
