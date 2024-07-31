import random
import cv2
from gym_match3.envs.match3_env import Match3Env


env = Match3Env(90)
helper = env.helper
action_arr = helper._from_action_to_tile()

print(f"Total size of the game state{env.observation_space}")
print(f"Number of actions in this game{env.action_space}")

_last_obs, infos = env.reset()
dones = False
action_space = infos["action_space"]


while not dones:
    # Identify the indices where the value is 1
    indices_with_one = [index for index, value in enumerate(action_space) if value == 1]

    # Randomly select one of those indices
    if indices_with_one:

        selected_action= random.choice([index for index, value in enumerate(action_space) if value == 1])
        print("Selected index:", selected_action)

        
        env.render(selected_action)

        obs, reward, dones, infos = env.step(selected_action)
        # selected_action = int(input("Enter the index of the action you want to take: "))
        print("Reward of this action:", reward)
        action_space = infos["action_space"]
    else:
        print("No indices with value 1 found.")
        dones = True
    
cv2.waitKey(0)
