import json
import os
import pandas as pd

import test.SkillDefine as SkillDefine

from gym_match3.envs.game import DameMonster, BoxMonster
from gym_match3.envs.constants import GameObject

def read_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

config_dir = "test/config"
level_file = os.path.join(config_dir, "Level.json")
monster_file = os.path.join(config_dir, "MonsterV2.json")
level_match = os.path.join(config_dir, "LevelMatch.xlsx")
real2vir_file = os.path.join(config_dir, "RealToVir.json")
monster_atk_file = os.path.join(config_dir, "MonsterTimeAtk.xlsx")


level_data = read_file(level_file)
monster_data = read_file(monster_file)

DEFAULT_TIME = 180

REAL2VIR = read_file(real2vir_file)
MONSTER_ATK = pd.read_excel(monster_atk_file)
LEVEL_MATCH = pd.read_excel(level_match)


REALMS_CONFIG, LEVELS_CONFIG, PHASES_CONFIG = level_data['pve'].values()
MONSTERS_CONFIG, MONSTER_PASSIVE_CONFIG, _ = monster_data.values()

AVAILABLE_SKILL = [SkillDefine.ACTION_GUARD, SkillDefine.ACTION_THROW_BLOCKER, SkillDefine.ACTION_ADD_SHIELD]


MONSTER_CREATE = {
    SkillDefine.ACTION_GUARD: {
        "class": DameMonster,
        "object": GameObject.monster_dame,
        },
    SkillDefine.ACTION_ADD_SHIELD: {
        "class": DameMonster,
        "object": GameObject.monster_dame      
        },
    SkillDefine.ACTION_THROW_BLOCKER: {
        "class": BoxMonster,
        "object": GameObject.monster_box_box
        },
    "default": {
        "class": DameMonster,
        "object": GameObject.monster_dame
        }
}

STATIC_MONSTERS = {
    SkillDefine.ACTION_GUARD: 0,
    SkillDefine.ACTION_ADD_SHIELD: 0,
    SkillDefine.ACTION_THROW_BLOCKER: 0,
    "default": 0
}

MONSTER_ID = '+'
TILE_ID = 'o'
NONTILE = 'x'

MONSTER_DIRECTION = [
                    [-1, 0],
                    [1, 1]
                ]

