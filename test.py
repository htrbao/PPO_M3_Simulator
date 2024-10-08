import numpy as np
import torch
import argparse
import json
import time
import os
import math
import sys
from copy import deepcopy
from gym_match3.envs.match3_env import Match3Env
from gym_match3.envs.levels import Match3Levels
from test.level_generate import get_real_levels

import torch.multiprocessing as mp

from training.common.policies import (
    ActorCriticPolicy,
    ActorCriticCnnPolicy
)
from training.common.utils import obs_as_tensor

def eval_single_loop(envs, model, device, max_step, total_monster_hp):
    __num_damage = 0
    # xai nhung ko hit
    __num_pu_move = 0
    __num_match_move = 0
    # xai nhung hit
    __num_hit = 0
    __num_pu_hit = 0
    __num_match_hit = 0
    __pu_on_box = 0
    __dmg_match = 0
    __dmg_pu = 0
    __num_win_games = 0
    __remain_mons_hp = 0
    current_step = 0
    obs, infos = envs.reset()
    while(current_step < max_step):
        with torch.no_grad():
            action_space = infos['action_space']
            obs_tensor = obs_as_tensor(obs, device)
            action_space = obs_as_tensor(action_space, device)
            actions, values, log_probs = model(obs_tensor, action_space)
        actions = actions.cpu().numpy()[0]
        
        obs, reward, done, infos = envs.step(actions)
        
        total_dmg = reward["rate_match_damage_on_monster"] + reward["rate_power_damage_on_monster"]
        
        __num_damage += total_dmg
        __dmg_match += reward["rate_match_damage_on_monster"]
        __dmg_pu += reward["rate_power_damage_on_monster"]
        __pu_on_box += reward["pu_on_box"]
        if total_dmg > 0:
            __num_hit += 1
            if reward["rate_match_damage_on_monster"] > 0:
                __num_match_hit += 1
            if reward["rate_power_damage_on_monster"] > 0:
                __num_pu_hit += 1
        else:
            if reward["match_score"] > 0:
                __num_match_move += 1
            if reward["pu_score"] > 0:
                __num_pu_move += 1
                
        current_step += 1
        if "game" in reward.keys():
            if reward["game"] > 0:
                __num_win_games += 1
            elif reward["game"] < 0: 
                __remain_mons_hp += reward["hp_mons"] / total_monster_hp
            break
        if done:
            break
    return {
        "__num_damage": __num_damage, 
        "__num_hit": __num_hit, 
        "__dmg_match": __dmg_match, 
        "__dmg_pu": __dmg_pu, 
        "__num_win_games": __num_win_games, 
        "__remain_mons_hp": __remain_mons_hp,
        "current_step": current_step,
        "__pu_on_box": __pu_on_box,
        "__num_pu_move": __num_pu_move,
        "__num_match_move": __num_match_move,
        "__num_pu_hit": __num_pu_hit,
        "__num_match_hit": __num_match_hit,
        }

def eval(model, obs_order, env, device, store_dir, num_eval=5):
    max_step = env['max_step']
    level = deepcopy(env['level'])
    envs = Match3Env(
        max_step,
        obs_order=obs_order,
        random_state=13,
        levels=Match3Levels([level]),
    )
    for i in range(len(env['monsters'])):
        env['monsters'][i].pop('monster_create')
        env['monsters'][i]['kwargs']['position'] = str(env['monsters'][i]['kwargs']['position'])
    __num_win_games = 0
    __num_damage = 0
    __num_hit = 0
    __dmg_match = 0
    __dmg_pu = 0
    __total_step = 0
    __remain_mons_hp = 0
    __pu_on_box = 0
    __num_pu_move = 0
    __num_match_move = 0
    __num_pu_hit = 0
    __num_match_hit = 0
    total_monster_hp =  sum([m['kwargs']['hp'] for m in env['monsters']])
    start_time = time.time()



    for i in range(num_eval):
        stat = eval_single_loop(envs, model, device, max_step, total_monster_hp)
        __num_damage += stat['__num_damage']
        __num_hit += stat['__num_hit']
        __dmg_match += stat['__dmg_match']
        __dmg_pu += stat['__dmg_pu']
        __num_win_games += stat['__num_win_games']
        __remain_mons_hp += stat['__remain_mons_hp']
        __total_step += stat['current_step']
        __pu_on_box += stat['__pu_on_box']
        __num_pu_move += stat['__num_pu_move']
        __num_match_move += stat['__num_match_move']
        __num_pu_hit += stat['__num_pu_hit']
        __num_match_hit += stat['__num_match_hit']
    result = {
        "realm_id": env['realm_id'],
        "node_id": env['node_id'],
        "level_id": env['level_id'],
        "num_mons": env['monsters'],
        "max_step": max_step,
        "num_games": num_eval,
        "num_win_games": __num_win_games,
        "num_hit": __num_hit,
        "total_damage": __num_damage / num_eval,
        "total_damage_pu": __dmg_pu / num_eval,
        "total_damage_match": __dmg_match / num_eval,
        "total_step": __total_step,
        "avg_damage_per_hit": __num_damage / __num_hit,
        "hit_rate": __num_hit / __total_step,
        "win_rate": __num_win_games / num_eval,
        "remain_hp_monster": (__remain_mons_hp / (num_eval - __num_win_games)) if __num_win_games != num_eval else 0,
        "pu_on_box_rate": __pu_on_box / __total_step,
        "num_pu_move_rate": __num_pu_move  / __total_step,
        "num_match_move_rate": __num_match_move  / __total_step,
        "num_pu_hit_rate": __num_pu_hit  / __total_step,
        "num_match_hit_rate": __num_match_hit  / __total_step,
    }
    
    print("Evaluation time: {:.2f}s".format(time.time() - start_time), result)
    with open(os.path.join(store_dir, f"realm_{env['realm_id']:02d}__node_{env['node_id']:02d}.json"), "w") as f:
        json.dump(result, f, indent = 2)
    return result


def normal_eval(model, obs_order, device, num_eval, REAL_LEVELS, store_dir):
    results = []
    model = model.to(device)
    for level in REAL_LEVELS:
        results.append(eval(model, obs_order, level, device, store_dir, num_eval))
    return results
    
def wrapper_eval(queue, model, device, len_group, obs_order, num_eval, REAL_LEVELS, store_dir, num_processes):
    torch.set_num_threads(math.floor(os.cpu_count()/num_processes))
    model = model.to(device)
    results = []
    levels = deepcopy(REAL_LEVELS[len_group[0]: len_group[1]])
    np.random.shuffle(levels)
    for level in levels:
        result = eval(model, obs_order, level, device, store_dir, num_eval)
        queue.put(result)
    
    print(f"done {len_group}")

    
    
def write_csv(results, store_dir, sep=','):
    header = f"{sep}".join(results[0].keys())
    with open(os.path.join(store_dir, 'final_result.csv'), 'w') as f:
        f.write(header + "\n")
        for result in results:
            result["num_mons"] = len(result["num_mons"])
            line = f"{sep}".join(str(v) for v in result.values())
            f.write(line + "\n")

def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num-eval', type=int, default=2, metavar='N',
                        help='how many eval to use (default: 2)')
    parser.add_argument('--num-processes', type=int, default=0, metavar='N',
                        help='how many processes to use (default: 0)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
    parser.add_argument('--checkpoint', type=str)
    
    parser.add_argument('--model-type', type=str, default='cnn')
    
    parser.add_argument(
        "--obs-order",
        type=str,
        default=[],
        nargs="+",
        help="Which features you want to use?",
    )
    return parser.parse_args()

    
def main():
    args = get_arguments()
    start_time = time.time()
    use_cuda = args.cuda and torch.cuda.is_available()
    use_mps = args.mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    num_eval = args.num_eval
    num_processes = args.num_processes
    obs_order = args.obs_order
    store_dir = os.path.join("_saved_test", os.path.basename(args.checkpoint).split(".")[0])
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    manager = mp.Manager()
    
    
    model = ActorCriticCnnPolicy.load(args.checkpoint) if args.model_type == 'cnn' else ActorCriticPolicy.load(args.checkpoint)
    model = model.cpu()
    model.eval()
    model.share_memory()
    results = []
    REAL_LEVELS = get_real_levels(True)
    if num_processes > 0:
        queue = manager.Queue()  # Use manager.Queue() instead of manager.list()
        processes = []
        len_group = len(REAL_LEVELS) // num_processes
        for rank in range(num_processes):
            first_len_group = rank * len_group
            if rank == (num_processes - 1):
                last_len_group = len(REAL_LEVELS)
            else:
                last_len_group = (rank+1)*len_group
            p = mp.Process(target=wrapper_eval, args=(queue, model, device, (first_len_group, last_len_group), obs_order, num_eval, REAL_LEVELS, store_dir, num_processes))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Gather results from the queue
        results = []
        while not queue.empty():
            results.append(queue.get())  # Combine results from all processes
    else:
        results = normal_eval(model, obs_order, device, num_eval, REAL_LEVELS, store_dir)

    
    
    with open(os.path.join(store_dir, 'final_result.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    write_csv(results, store_dir)

    print("Total evaluation time: {:.2f}s".format(time.time() - start_time))
    
if __name__ == '__main__':
    main()
