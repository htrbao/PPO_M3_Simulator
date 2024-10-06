import math
import numpy as np
import test.SkillDefine as SkillDefine

from test.global_config import (
    LEVEL_MATCH, 
    MONSTER_ATK, 
    DEFAULT_TIME, 
    REAL2VIR,
    MONSTER_CREATE,
    MONSTER_DIRECTION,
    MONSTER_ID,
    MONSTER_PASSIVE_CONFIG,
    STATIC_MONSTERS,
    LEVELS_CONFIG,
    MONSTERS_CONFIG,
    PHASES_CONFIG,
    AVAILABLE_SKILL,
    NONTILE
)

from gym_match3.envs.game import Point
from gym_match3.envs.constants import Level, GameObject

np.random.seed(13)

def get_monster_max_hp(realm_id, node_id):
    try:
        level_info = LEVEL_MATCH.loc[LEVEL_MATCH.RealmID.eq(realm_id) & LEVEL_MATCH.Node.eq(node_id)].reset_index()
        max_hp = int(math.ceil(level_info.at[0, "SumMatch"]))
    except:
        max_hp = int(LEVEL_MATCH["SumMatch"].mean())
    return max_hp


def get_player_hp(realm_id, monsters):
    player_hp = 0
    time_atk = 0
    for monster in monsters:
        monster_creep = "_".join(monster["name"].split("_")[2:])
        try:
            monster_time = MONSTER_ATK.loc[MONSTER_ATK.CreepID.eq(monster_creep)].reset_index().at[0, "Time"]
        except:
            # print(monster_creep, "set default 180")
            monster_time = DEFAULT_TIME
        if monster_time > 0:
            time_atk += 1.0/float(monster_time)
    if time_atk > 0:
        time_atk = 1 / time_atk
    else:
        time_atk = DEFAULT_TIME
    player_hp = int(math.ceil(time_atk * REAL2VIR[str(realm_id)]["match/time"]))
    return player_hp


def order_monster_ids(monster_ids_start, monsters, height, width):
    monster_ids_choose = [False] * monster_ids_start.shape[0]
    monster_orders = []
    
    monsters = sorted(monsters, key=lambda x: (x['kwargs']['height'] * x['kwargs']['width']), reverse=True)

    for monster in monsters:
        kwargs = monster['kwargs']
        monster_height = kwargs["height"] - 1
        monster_width = kwargs["width"] - 1
        for id, point in enumerate(monster_ids_start):
            if monster_ids_choose[id]:
                continue
            
            monster_start = point + [MONSTER_DIRECTION[0][0] * monster_height, MONSTER_DIRECTION[0][1] * monster_width]
            monster_end = monster_start + [MONSTER_DIRECTION[1][0] * monster_height, MONSTER_DIRECTION[1][1] * monster_width]
            if not(monster_end[0] < 0 or monster_end[0] >= height or monster_end[1] < 0 or monster_end[1] >= width):
                monster_ids_choose[id] = True
                monster_orders.append(point)

    return monsters, np.asarray(monster_orders)
                

def process_map(map_str, monsters, monster_max_hp, num_tiles):
    height = len(map_str)
    width = len(map_str[0])
    monster_list = []
    # print(map_str)
    processed_map = np.array([list(row) for row in map_str])
    
    monster_ids_start = np.transpose(np.nonzero(processed_map == MONSTER_ID))
    processed_map = np.where(processed_map == NONTILE, -1, 0)
    
    monsters, monster_ids_start = order_monster_ids(monster_ids_start, monsters, height, width)
    if monster_ids_start.shape[0]!= len(monsters):
        raise Exception("Cannot find suitable starting points for monsters")

    total_monster_hp = sum([monster_info['monster_hp'] for monster_info in monsters])

    for monster_id, monster_info in zip(monster_ids_start, monsters):
        monster_hp = monster_info['monster_hp']
        monster_create = monster_info['monster_create']
        kwargs = monster_info['kwargs']
        monster_height = kwargs["height"] - 1
        monster_width = kwargs["width"] - 1
        monster_id_start = monster_id + [MONSTER_DIRECTION[0][0] * monster_height, MONSTER_DIRECTION[0][1] * monster_width]
        monster_id_end = monster_id_start + [MONSTER_DIRECTION[1][0] * monster_height, MONSTER_DIRECTION[1][1] * monster_width]

        processed_map[monster_id_start[0]: monster_id_end[0]+1, monster_id_start[1]:monster_id_end[1]+1] = monster_create['object']
        kwargs["position"] = Point(
            monster_id_start[0], 
            monster_id_start[1])
        kwargs["hp"] = int(monster_max_hp * monster_hp / total_monster_hp)
        monster_list.append(
            monster_create["class"](**kwargs)
        )

    return Level(
        height,
        width,
        num_tiles,
        processed_map.tolist(),
        monster_list
        )
            

def get_skill_monster(stateInfos):
    monster_type = "default"
    for skill_ids in stateInfos["1"].values():
        for skill_id in skill_ids:
            skill_id = str(skill_id)
            *_, commands = MONSTER_PASSIVE_CONFIG[skill_id].values()
            # print(MONSTER_PASSIVE_CONFIG[skill_id])
            for command in commands:
                if command.get("action", None) is not None and command["action"].get("type", -1) in AVAILABLE_SKILL: 
                    return command["action"]["type"], command["action"]
    return monster_type, None


def create_kwargs_class(width, height, monster_type, action):
    kwargs = {
            "width": width,
            "height": height,
            }

        
    if monster_type == SkillDefine.ACTION_THROW_BLOCKER:
        kwargs["box_mons_type"] = GameObject.monster_box_box
    else:
        kwargs["dame"] = 0
        if monster_type == SkillDefine.ACTION_GUARD:
            direction = action.get("direction", 15)
            kwargs["request_masked"] = [not (direction & SkillDefine.DIRECTION_LEFT), 
                                        not (direction & SkillDefine.DIRECTION_RIGHT), 
                                        not (direction & SkillDefine.DIRECTION_UP), 
                                        not (direction & SkillDefine.DIRECTION_DOWN), 1]
        elif monster_type == SkillDefine.ACTION_ADD_SHIELD:
            ratio = action.get("ratio", 0.5)
            kwargs["have_paper_box"] = True
            kwargs["relax_interval"] = max(min(2 + min(int(math.ceil(2 / ratio)), 10) + np.random.randint(-1, 1), 15), 2)
            kwargs["setup_interval"] = max(min(1 + min(int(math.ceil(1 / ratio)), 5) + np.random.randint(-1, 1), 8), 1)
        
    return kwargs


def create_monster(monster_infos, realm_level):
    monsters = []

    for monster_info in monster_infos:
        try:            
            monster_id, level, _ = monster_info.values()
            monster_id = str(monster_id)

            name, _, _, width, height, stats, stateInfos = MONSTERS_CONFIG[monster_id].values()
            monster_hp = int(stats[0]['maxHp'])
            monster_type, action = get_skill_monster(stateInfos)
            monster_create = MONSTER_CREATE[monster_type]

            
            kwargs = create_kwargs_class(width, height, monster_type, action)
            
            monsters.append({
                "name": name,
                "monster_create": monster_create,
                "kwargs": kwargs,
                "monster_hp": monster_hp,
            })
            STATIC_MONSTERS[monster_type] += 1
        except Exception as e:
            # print(f"\t\t\tWARNING: realm level {realm_level} connot load monster")
            continue
    return monsters


def get_map_infos(realm_info: dict, hit_rate: float = 0.5):
    try:
        realm_id = realm_info["realmId"]
        realm_node = realm_info["nodeId"]
        realm_level = str(realm_info["levels"][0])
        realm_level_phase = LEVELS_CONFIG[realm_level]["phases"][0]
        realm_phase_id, realm_map_id, realm_hp_multiplier = realm_level_phase.values()
        realm_phase_id = str(realm_phase_id)
        realm_map_id = str(realm_map_id)
        realm_level_phase_map, realm_level_phase_monster, realm_level_phase_num_tiles = PHASES_CONFIG[realm_phase_id]["maps"][realm_map_id].values()

        monster_max_hp = get_monster_max_hp(realm_id=realm_id, node_id=realm_node)

        monsters = create_monster(realm_level_phase_monster, realm_level)
        max_step = get_player_hp(realm_id, monsters)

        if len(monsters) == 0:
            raise Exception(f"Realm id {realm_id}, Node id {realm_node}, Level {realm_level} do not have monster")
        if sum([row.count('+') for row in realm_level_phase_map]) != len(monsters):
            raise Exception(f"Realm id {realm_id}, Node id {realm_node}, Level {realm_level} monster do not match map")
        
        processed_map = process_map(realm_level_phase_map, monsters, monster_max_hp, realm_level_phase_num_tiles)
        
        return {
            "realm_id": realm_id,
            "node_id": realm_node,
            "level_id": realm_level,
            "level": processed_map,
            "max_step": max_step,
            "monsters": monsters
        }
    except Exception as e:
        print("\t\tWARNING: ",e)
        return None