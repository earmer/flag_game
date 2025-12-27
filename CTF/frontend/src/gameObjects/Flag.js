import ASSETS from '../assets.js';

export default class Flag extends Phaser.Physics.Arcade.Sprite
{
    constructor(scene, x, y, team, canPickup)
    {
        if (team == "L") {
          super(scene, x, y, ASSETS.spritesheet.L_flag.key);
        } else {
          super(scene, x, y, ASSETS.spritesheet.R_flag.key);
        }

        scene.add.existing(this);
        scene.physics.add.existing(this);

        this.team = team;
        this.mapOffset = scene.getMapOffset();
        this.posX = x;
        this.posY = y;
        this.setPosition(this.mapOffset.x + (x * this.mapOffset.tileSize), this.mapOffset.y + (y * this.mapOffset.tileSize));
        this.setDepth(90);
        this.scene = scene;
        this.canPickup = canPickup;
    }

    collect() {
        if (!this.canPickup) {
            return false;
        }
        this.scene.removeFlagItem(this);
        return true;
    }

    // return player status to send to remote backend
    getStatus() {
        return {
          "canPickup": this.canPickup,
          "posX": this.posX,
          "posY": this.posY,
        }
    }
}
