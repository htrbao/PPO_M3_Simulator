import random
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from gym_match3.envs.match3_env import Match3Env
from gym_match3.envs.game import Point

env = Match3Env(90)

print(f"Total size of the game state{env.observation_space}")
print(f"Number of actions in this game{env.action_space}")

s_t = time.time()

_last_obs, infos = env.reset()
dones = False
action_space = infos["action_space"]

selected_action = 0

while not dones:
    # Identify the indices where the value is 1
    indices_with_one = [index for index, value in enumerate(action_space) if value == 1]

    # Randomly select one of those indices
    if indices_with_one:
        a = [int(x) for x in input("INPUT 4 COORD: ").split()]
        # env.render(selected_action)
        obs, reward, dones, infos = env.step(None, Point(a[0], a[1]), Point(a[2], a[3]))

        # print(obs.shape)
        print("Reward of this action:", reward)

        action_space = infos["action_space"]

    else:
        print("No indices with value 1 found.")
        dones = True

print("end game time", time.time() - s_t)