import ASSETS from '../assets.js';

const PlayerDirection = Object.freeze({
  UP: "up",
  DOWN: "down",
  LEFT: "left",
  RIGHT: "right",
});

export default class Player extends Phaser.Physics.Arcade.Sprite {
    moveSpeed = 300; // time in milliseconds to move from one tile to another
    frameDuration = 0;
    accumulator = 0;
    target = { x: 0, y: 0 };

    constructor(scene, name, x, y, team, spriteChoice = 1, useAWSD = true) {
        super(scene, x, y, ASSETS.spritesheet.characters.key, (spriteChoice - 1) * 12 + 1);

        scene.add.existing(this);
        scene.physics.add.existing(this);

        this.name = name;
        this.team = team;
        this.inPrison = false;
        this.inPrisonTimeLeft = 0;
        this.inPrisonDuration = 20000; // time in milliseconds to stay in prison unless a teammate saves the player.
        this.hasFlag = false;
        this.spriteChoice = spriteChoice;

        this.mapOffset = scene.getMapOffset();
        this.target.x = this.mapOffset.x + (x * this.mapOffset.tileSize);
        this.target.y = this.mapOffset.y + (y * this.mapOffset.tileSize);
        this.setPosition(this.target.x, this.target.y);
        this.setCollideWorldBounds(true);
        this.setDepth(100);
        this.scene = scene;
        this.frameDuration = this.moveSpeed / this.mapOffset.tileSize;

        this.remoteControl = null;
        // key control
        if (useAWSD) {
            this.keys = this.scene.awsd_keys;
        } else {
            this.keys = this.scene.cursors;
        }

        this.can_go_next_tile = false; // will go next tile only if this is true
    }

    collectFlag() {
        this.hasFlag = true;
    }

    dropFlag() {
        this.hasFlag = false;
    }

    setRemoteControl(remoteControl) {
      this.remoteControl = remoteControl;
    }

    update(time, delta) {
        this.accumulator += delta;

        while (this.accumulator > this.frameDuration) {
            this.accumulator -= this.frameDuration;
            if (this.inPrison) {
                this.inPrisonTimeLeft -= this.frameDuration;
                if (this.inPrisonTimeLeft <= 0) {
                    this.inPrison = false;
                    this.inPrisonTimeLeft = 0;
                }
            }
            if (!this.inPrison) {
                this.checkInput();
                this.move();
            } else {
                this.showStaticImage();
            }
        }
    }

    checkInput() {
        // check if player is at target position
        if (this.can_go_next_tile && this.target.x === this.x && this.target.y === this.y) {
            this.can_go_next_tile = false;
            const moveDirection = { x: 0, y: 0 }; // default move direction

            // Keys take priority over heuristics
            if (this.keys.left.isDown) moveDirection.x--;
            else if (this.keys.right.isDown) moveDirection.x++;
            else if (this.keys.up.isDown) moveDirection.y--;
            else if (this.keys.down.isDown) moveDirection.y++;
            else if (this.remoteControl == PlayerDirection.LEFT) moveDirection.x--;
            else if (this.remoteControl == PlayerDirection.RIGHT) moveDirection.x++;
            else if (this.remoteControl == PlayerDirection.UP) moveDirection.y--;
            else if (this.remoteControl == PlayerDirection.DOWN) moveDirection.y++;

            // set next tile coordinates to move towards
            const nextPosition = {
                x: this.x + (moveDirection.x * this.mapOffset.tileSize),
                y: this.y + (moveDirection.y * this.mapOffset.tileSize)
            };

            // check if next tile to move towards is walkable
            if (!this.scene.isWall(nextPosition.x, nextPosition.y)) {
                // set target position to move towards
                this.target.x = nextPosition.x;
                this.target.y = nextPosition.y;
            }
        }
    }

    // move player towards target position
    move() {
        let animation_key = "player" + this.spriteChoice + (this.hasFlag ? "-characters_"+this.team+"_flag-": "-characters-");

        if (this.x < this.target.x) {
            this.x ++;
            this.anims.play(animation_key + "right", true);
        }
        else if (this.x > this.target.x) {
            this.x --;
            this.anims.play(animation_key + "left", true);
        }
        if (this.y < this.target.y) {
            this.y ++;
            this.anims.play(animation_key + "down", true);
        }
        else if (this.y > this.target.y) {
            this.y --;
            this.anims.play(animation_key + "up", true);
        }
    }

    showStaticImage() {
        let animation_key = "player" + this.spriteChoice + "-characters-down";
        this.anims.play(animation_key, true);
    }

    toPrison(prisonX, prisonY) {
        this.target.x = this.mapOffset.x + (prisonX * this.mapOffset.tileSize);
        this.target.y = this.mapOffset.y + (prisonY * this.mapOffset.tileSize);
        this.setPosition(this.target.x, this.target.y);
        this.inPrison = true;
        this.inPrisonTimeLeft = this.inPrisonDuration;
    }

    hit() {
        this.destroy();
    }

    // return player status to send to remote backend
    getStatus() {
        return {
            "name": this.name,
            "team": this.team,
            "hasFlag": this.hasFlag,
            "posX": (this.target.x - this.mapOffset.x) / this.mapOffset.tileSize,
            "posY": (this.target.y - this.mapOffset.y) / this.mapOffset.tileSize,
            "inPrison": this.inPrison,
            "inPrisonTimeLeft": this.inPrisonTimeLeft,
            "inPrisonDuration": this.inPrisonDuration,
        }
    }
}
