import numpy as np
from collections import namedtuple

Level = namedtuple("Level", ["h", "w", "n_shapes", "board", "list_monsters"])
base_hp = 0

class GameObject:
    immovable_shape = -1
    # Tile
    color1 = 1
    color2 = 2
    color3 = 3
    color4 = 4
    color5 = 5
    tiles = np.arange(color1, color5 + 1, 1)
    # Power up
    power_missile_h = 6  # horizontal missile
    power_missile_v = 7  # vertical missile
    power_bomb = 8
    power_plane = 9
    power_disco = 10
    powers = np.arange(power_missile_h, power_disco + 1, 1)
    # Blocker
    blocker_box = 11
    blocker_thorny = 12
    blocker_bomb = 13
    blockers = np.arange(blocker_box, blocker_bomb + 1, 1)
    # Monster
    monster_dame = 14
    monster_box_box = 15
    monster_box_bomb = 16
    monster_box_thorny = 17
    monster_box_both = 18
    monsters = np.arange(monster_dame, monster_box_both + 1, 1)


def mask_immov_mask(line, immovable_shape, can_move_blocker=False):
    immov_mask = line == immovable_shape
    for _immov_obj in GameObject.monsters:
        immov_mask |= line == _immov_obj
    if not can_move_blocker:
        for _immov_obj in GameObject.blockers:
            immov_mask |= line == _immov_obj

    return immov_mask

def need_to_match(shape):
    return shape in np.concatenate([GameObject.tiles, GameObject.powers])
