import numpy as np
import torch
import argparse
import json
import time
import os
import math
import sys
from datetime import datetime
from copy import deepcopy
from gym_match3.envs.match3_env import Match3Env
from gym_match3.envs.levels import Match3Levels
from test.level_generate import get_real_levels
from test.utils import import_single_model
from test.global_config import DB_CONFIG
from test.database import ClickhouseDB
import torch.multiprocessing as mp

from training.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
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
    __create_pu = {"disco": 0, "missile_v": 0, "missile_h": 0, "plane": 0, "bomb": 0}
    current_step = 0
    obs, infos = envs.reset()
    while current_step < max_step:
        with torch.no_grad():
            action_space = infos["action_space"]
            obs_tensor = obs_as_tensor(obs, device)
            action_space = obs_as_tensor(action_space, device)
            actions, values, log_probs = model(obs_tensor, action_space, deterministic=True)
        actions = actions.cpu().numpy()[0]

        obs, reward, done, infos = envs.step(actions)

        total_dmg = (
            reward["rate_match_damage_on_monster"]
            + reward["rate_power_damage_on_monster"]
        )

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
            for k in __create_pu.keys():
                __create_pu[k] += reward["create_pu"].get(k, 0)

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
        "total_step": current_step,
        "total_hit": __num_hit,
        "total_dmg": __num_damage,
        "dmg_match": __dmg_match,
        "dmg_pu": __dmg_pu,
        "is_win": __num_win_games,
        "remain_mons_hp": __remain_mons_hp,
        "pu_on_box": __pu_on_box,
        "pu_missing": __num_pu_move,
        "match_missing": __num_match_move,
        "pu_hit": __num_pu_hit,
        "match_hit": __num_match_hit,
        "create_disco": __create_pu["disco"],
        "create_bomb": __create_pu["bomb"],
        "create_plane": __create_pu["plane"],
        "create_missile_h": __create_pu["missile_h"],
        "create_missile_v": __create_pu["missile_v"],
    }


def eval(model, obs_order, env, device, store_dir, num_eval=5):
    max_step = env["max_step"]
    level = deepcopy(env["level"])
    envs = Match3Env(
        max_step,
        obs_order=obs_order,
        random_state=13,
        levels=Match3Levels([level]),
    )
    for i in range(len(env["monsters"])):
        env["monsters"][i].pop("monster_create")
        env["monsters"][i]["kwargs"]["position"] = str(
            env["monsters"][i]["kwargs"]["position"]
        )
    results = []
    total_monster_hp = sum([m["kwargs"]["hp"] for m in env["monsters"]])
    start_time = time.time()

    for i in range(num_eval):
        stat = eval_single_loop(envs, model, device, max_step, total_monster_hp)
        stat["realm_id"] = env["realm_id"]
        stat["node_id"] = env["node_id"]
        stat["level_id"] = env["level_id"]
        stat["test_id"] = i
        stat["num_mons"] = env["monsters"]
        results.append(stat)

    print("Evaluation time: {:.2f}s".format(time.time() - start_time), f"realmd {env['realm_id']}, node {env['node_id']}")
    # with open(
    #     os.path.join(
    #         store_dir, f"realm_{env['realm_id']:02d}__node_{env['node_id']:02d}.json"
    #     ),
    #     "w",
    # ) as f:
    #     json.dump(results, f, indent=2)
    return results


def normal_eval(model, obs_order, device, num_eval, REAL_LEVELS, store_dir):
    results = []
    model = model.to(device)
    for level in REAL_LEVELS:
        results.extend(eval(model, obs_order, level, device, store_dir, num_eval))
    return results


def wrapper_eval(
    queue,
    model,
    device,
    len_group,
    obs_order,
    num_eval,
    REAL_LEVELS,
    store_dir,
    num_processes,
):
    torch.set_num_threads(math.floor(os.cpu_count() / num_processes))
    model = model.to(device)
    levels = deepcopy(REAL_LEVELS[len_group[0] : len_group[1]])
    np.random.shuffle(levels)
    for level in levels:
        results = eval(model, obs_order, level, device, store_dir, num_eval)
        for result in results:
            queue.put(result)

    print(f"done {len_group}")


def write_csv(results, store_dir, sep=","):
    header = f"{sep}".join(results[0].keys())
    with open(os.path.join(store_dir, "final_result.csv"), "w") as f:
        f.write(header + "\n")
        for result in results:
            result["num_mons"] = len(result["num_mons"])
            line = f"{sep}".join(str(v) for v in result.values())
            f.write(line + "\n")


def import_db(store_dir):
    db = ClickhouseDB(DB_CONFIG["host"], DB_CONFIG["port"], DB_CONFIG["user"], DB_CONFIG["password"], DB_CONFIG["database"])
    row = import_single_model(store_dir, datetime.today(), db)
    
    return row
def get_arguments():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=2,
        metavar="N",
        help="how many eval to use (default: 2)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=0,
        metavar="N",
        help="how many processes to use (default: 0)",
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument(
        "--mps", action="store_true", default=False, help="enables macOS GPU training"
    )
    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--model-type", type=str, default="cnn")

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

    # store directory for storing evaluation results
    store_dir = os.path.join(
        "_saved_test", os.path.basename(args.checkpoint).split(".")[0]
    )
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    # create free queue for storing results
    manager = mp.Manager()

    # load model
    model = (
        ActorCriticCnnPolicy.load(args.checkpoint)
        if args.model_type == "cnn"
        else ActorCriticPolicy.load(args.checkpoint)
    )
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
                last_len_group = (rank + 1) * len_group
            p = mp.Process(
                target=wrapper_eval,
                args=(
                    queue,
                    model,
                    device,
                    (first_len_group, last_len_group),
                    obs_order,
                    num_eval,
                    REAL_LEVELS,
                    store_dir,
                    num_processes,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Gather results from the queue
        results = []
        while not queue.empty():
            results.append(queue.get())  # Combine results from all processes
    else:
        results = normal_eval(
            model, obs_order, device, num_eval, REAL_LEVELS, store_dir
        )

    # save results as JSON file
    with open(os.path.join(store_dir, "final_result.json"), "w") as f:
        json.dump(results, f, indent=2)
    # save results as CSV file
    write_csv(results, store_dir)
    # import_db(store_dir)

    print("Total evaluation time: {:.2f}s".format(time.time() - start_time))


if __name__ == "__main__":
    main()
