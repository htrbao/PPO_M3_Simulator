import random
import cv2
from gym_match3.envs.match3_env import Match3Env
from training.ppo import PPO
from training.m3_model.m3_cnn import ResNet, M3CnnLargerFeatureExtractor
from training.common.utils import obs_as_tensor
import torch
import argparse
import time

def test_args_parser():
    parser = argparse.ArgumentParser(
        "BEiT fine-tuning and evaluation script for image classification",
        add_help=False,
    )

    # Model Information
    parser.add_argument(
        "--pi",
        type=int,
        nargs="+",
        help="The linear layer size of the Policy Model",
    )
    
    parser.add_argument(
        "--vf",
        type=int,
        nargs="+",
        help="The linear layer size of the Value Function",
    )
    
    parser.add_argument(
        "--prefix_name",
        type=str,
        default="m3_with_cnn",
        help="prefix name of the model",
    )
    
    parser.add_argument(
        "--mid_channels",
        type=int,
        default=64,
        help="Number of intermediary channels in CNN model",
    )
    
    parser.add_argument(
        "--num_first_cnn_layer",
        type=int,
        default=4,
        help="Number of intermediary layers in CNN model",
    )
    parser.add_argument(
        "--resnet",
        type= bool,
        help="Use ResNet as the feature extractor",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model",
    )
    return parser.parse_args()
if __name__ == "__main__":
    args = test_args_parser()
    print("args:", args)
    env = Match3Env(90 ,test=True)
    PPO_test = PPO(
            policy="CnnPolicy",
            env=env,
            policy_kwargs={
                "net_arch": dict(pi=args.pi, vf=args.vf),
                "features_extractor_class": ResNet if args.resnet else  M3CnnLargerFeatureExtractor,
                "features_extractor_kwargs": {
                    "mid_channels": args.mid_channels,
                    "out_channels": 161,
                    "num_first_cnn_layer": args.num_first_cnn_layer,
                    "resnet_variant": "resnet50",  
                },
                "optimizer_class": torch.optim.Adam,
                "share_features_extractor": False,
            },
            device=args.device,
            prefix_name=args.prefix_name,
            _checkpoint=args.checkpoint,
            # actor_device_cpu=args.actor_device_cpu,
        )
    PPO_test.policy.set_training_mode(False)
    policy = PPO_test.policy
    device = PPO_test.device
    
    print(f"Total size of the game state{env.observation_space}")
    print(f"Number of actions in this game{env.action_space}")

    _last_obs, infos = env.reset()
    dones = False
    action_space = infos["action_space"]
    n_steps = 0
    while True:
        with torch.no_grad():
                obs_tensor = obs_as_tensor(_last_obs, device).clone().detach()
                action_space_tensor = obs_as_tensor(action_space, device).clone().detach()
                actions, values, log_probs = policy(obs_tensor, action_space_tensor)

        actions = actions.cpu().clone().detach().numpy()

        selected_action = actions[0]
        print("Selected index:", selected_action)
        env.render(selected_action)
        new_obs, rewards, dones, infos = env.step(selected_action)
        print("Action taken:", selected_action)
        print("Reward of this action:", rewards)
        # print("Game state after the action:", new_obs)
        action_space = infos["action_space"]
        _last_obs = new_obs

        n_steps += 1
        print("Number of steps taken:", n_steps)
        
        if "game" in rewards.keys():
            if rewards['game'] < 0:
                print("You have lost the game!")
            else:
                print("You have won the game!")
            time.sleep(100)
            n_steps = 0
        print('--------------------------------------------------------')
            