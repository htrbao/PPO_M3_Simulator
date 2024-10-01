import numpy as np
import torch
import argparse
import json
import time

from gym_match3.envs.match3_env import Match3Env
from gym_match3.envs.levels import Match3Levels
from test.level_generate import get_real_levels

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
def eval(model, obs_order, env, device, num_eval=5):
    level = env['level']
    max_step = env['max_step']

    __num_win_games = 0
    __num_damage = 0
    __num_hit = 0
    __total_step = 0
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
    

    while(current_step < max_step):
        obs_tensor = obs_as_tensor(obs, device)
        action_space = obs_as_tensor(action_space, device)

        actions, values, log_probs = model(obs_tensor, action_space)
        obs, rewards, dones, infos = envs.step(actions)
        __total_step += 1
        for done, reward in zip(dones, rewards):
            if "game" in reward.keys():
                print(dones, rewards)
                if done:
                    __num_win_games += 0 if reward["game"] < 0 else 1
                total_dmg = reward["match_damage_on_monster"] + reward["power_damage_on_monster"]
                __num_damage += total_dmg
                __num_hit += 0 if total_dmg == 0 else 1
        if all(dones):
            break
        action_space = np.stack([x["action_space"] for x in infos])
    print("Evaluation time: {:.2f}s".format(time.time() - start_time))
    envs.close()
    return {
        "realm_id": env['realm_id'],
        "node_id": env['node_id'],
        "level_id": env['level_id'],
        "max_step": max_step,
        "num_games": num_eval,
        "num_win_games": __num_win_games,
        "num_damage": __num_damage,
        "num_hit": __num_hit,
        "avg_step": __total_step,
    }
        

def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                        help='how many training processes to use (default: 2)')
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

    use_cuda = args.cuda and torch.cuda.is_available()
    use_mps = args.mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    num_processes = args.num_processes
    model = ActorCriticCnnPolicy.load(args.checkpoint)
    model = model.to(device)
    # model.share_memory()
    REAL_LEVELS = get_real_levels(True)[:5]
    results = []
    for level in REAL_LEVELS:
        results.append(eval(model, args.obs_order, level, device))

    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
if __name__ == '__main__':
    main()