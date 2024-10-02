import numpy as np
import torch
import argparse
import json
import time
import os

from gym_match3.envs.match3_env import Match3Env
from gym_match3.envs.levels import Match3Levels
from test.level_generate import get_real_levels

import torch.multiprocessing as mp

from training.common.policies import (
    ActorCriticPolicy,
    ActorCriticCnnPolicy
)
from training.common.utils import obs_as_tensor
from training.common.vec_env import SubprocVecEnv


def make_env(levels, max_step, obs_order):
    def _init():
        env = Match3Env(
            max_step,
            obs_order=obs_order,
            levels=Match3Levels([levels]),
            
        )
        return env

    return _init

def eval(model, obs_order, env, device, store_dir, num_eval=5):
    model = model.to(device)
    level = env['level']
    max_step = env['max_step']
    for i in range(len(env['monsters'])):
        env['monsters'][i].pop('monster_create')
        env['monsters'][i]['kwargs']['position'] = str(env['monsters'][i]['kwargs']['position'])
    __num_win_games = 0
    __num_damage = 0
    __num_hit = 0
    __total_step = 0
    __remain_mons_hp = 0
    total_monster_hp =  sum([m['kwargs']['hp'] for m in env['monsters']])
    start_time = time.time()

    envs = SubprocVecEnv(
            [
                make_env(level, max_step, obs_order)
                for _ in range(num_eval)
            ]
        )

    current_step = 0
    obs = envs.reset()
    action_space = np.stack([x["action_space"] for x in envs.reset_infos])
    
    check_dones = [0] * num_eval
    while(current_step < max_step):
        with torch.no_grad():
            obs_tensor = obs_as_tensor(obs, device)
            action_space = obs_as_tensor(action_space, device)

            actions, values, log_probs = model(obs_tensor, action_space)
            
        actions = actions.cpu().numpy()

        obs, rewards, dones, infos = envs.step(actions)
        for id, (done, reward) in enumerate(zip(dones, rewards)):
            if not check_dones[id]:
                total_dmg = reward["match_damage_on_monster"] + reward["power_damage_on_monster"]
                __num_damage += total_dmg
                __num_hit += 0 if total_dmg == 0 else 1
                __total_step += 1
                if "game" in reward.keys():
                    if reward["game"] > 0:
                        __num_win_games += 1
                    elif reward["game"] < 0: 
                        __remain_mons_hp += reward["hp_mons"] / total_monster_hp
            if done:
                check_dones[id] = 1
        if all(check_dones):
            break
        current_step += 1
        action_space = np.stack([x["action_space"] for x in infos])
    envs.close()
    
    result = {
        "realm_id": env['realm_id'],
        "node_id": env['node_id'],
        "level_id": env['level_id'],
        "num_mons": env['monsters'],
        "max_step": max_step,
        "num_games": num_eval,
        "num_win_games": __num_win_games,
        "num_hit": __num_hit,
        "total_damage": __num_damage,
        "total_step": __total_step,
        "avg_damage_per_hit": __num_damage / __num_hit,
        "hit_rate": __num_hit / __total_step,
        "win_rate": __num_win_games / num_eval,
        "remain_hp_monster": (__remain_mons_hp / (num_eval - __num_win_games)) if __num_win_games != num_eval else 0,
    }
    
    print("Evaluation time: {:.2f}s".format(time.time() - start_time), result)
    with open(os.path.join(store_dir, f"realm_{env['realm_id']:02d}__node_{env['node_id']:02d}.json"), "w") as f:
        json.dump(result, f, indent = 2)
    return result


def normal_eval(model, obs_order, device, num_eval, REAL_LEVELS, store_dir):
    results = []
    for level in REAL_LEVELS:
        results.append(eval(model, obs_order, level, device, store_dir, num_eval))
    return results
    
def wrapper_eval(queue, model, device, rank, len_group, obs_order, num_eval, REAL_LEVELS, store_dir):
    model = model.to(device)
    results = []
    for level in REAL_LEVELS[rank*len_group:(rank+1)*len_group]:
        results.append(eval(model, obs_order, level, device, store_dir, num_eval))
    queue.extend(results)
    
    
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
    
    
    model = ActorCriticCnnPolicy.load(args.checkpoint)
    # model = model.to(device)
    model.share_memory()
    results = []
    REAL_LEVELS = get_real_levels(True)
    if num_processes > 0:
        queue = manager.list()
        processes = []
        for rank in range(num_processes):
            len_group = len(REAL_LEVELS) // num_processes
            p = mp.Process(target=wrapper_eval, args=(queue, model, device, rank, len_group, obs_order, num_eval, REAL_LEVELS, store_dir))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
        
        results = []
        for item in queue:
            results.append(item)
    else:
        results = normal_eval(model, obs_order, device, num_eval, REAL_LEVELS, store_dir)
    
    
    with open(os.path.join(store_dir, 'final_result.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    write_csv(results, store_dir)

    print("Total evaluation time: {:.2f}s".format(time.time() - start_time))
    
if __name__ == '__main__':
    main()
