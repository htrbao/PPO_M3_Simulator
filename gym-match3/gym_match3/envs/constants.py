from dataclasses import dataclass

import numpy as np
from collections import namedtuple

Level = namedtuple("Level", ["h", "w", "n_shapes", "board", "list_monsters"])
base_hp = 5

@dataclass(frozen=True)
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

    # Set of all type of shapes for faster check action
    set_tiles_shape = set(tiles)
    set_powers_shape = set(powers)
    set_blockers_shape = set(blockers)
    set_monsters_shape = set(monsters)
    set_unmovable_shape = set_monsters_shape | set_blockers_shape
    set_unmovable_shape.add(immovable_shape)
    set_movable_shape = set_tiles_shape | set_powers_shape


@dataclass(frozen=True)
class GameAction:
    swap_normal = 0
    swap_4_v = 1
    swap_4_h = 2
    swap_L = 3
    swap_T = 4
    swap_2x2 = 5
    swap_5 = 6
    swap_power_missile_h = 7
    swap_power_missile_v = 8
    swap_power_bomb = 9
    swap_power_plane = 10
    swap_power_disco = 11
    swap_merge_missile_missile = 12
    swap_merge_missile_bomb = 13
    swap_merge_missile_plane = 14
    swap_merge_missile_disco = 15
    swap_merge_bomb_bomb = 16
    swap_merge_bomb_plane = 17
    swap_merge_bomb_disco = 18
    swap_merge_plane_plane = 19
    swap_merge_plane_disco = 20
    swap_merge_disco_disco = 21


def mask_immov_mask(line, immovable_shape, can_move_blocker=False):
    immov_mask = line == immovable_shape
    for _immov_obj in GameObject.monsters:
        immov_mask |= line == _immov_obj
    if not can_move_blocker:
        for _immov_obj in GameObject.blockers:
            immov_mask |= line == _immov_obj

    return immov_mask


def need_to_match(shape):
    return shape in GameObject.set_movable_shape
