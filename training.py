import argparse
import time
import torch

from gym_match3.envs.match3_env import Match3Env
from gym_match3.envs.levels import Match3Levels, LEVELS
from training.ppo import PPO
from training.m3_model.m3_cnn import M3CnnFeatureExtractor, M3CnnLargerFeatureExtractor, ResNet
from training.common.utils import collect_rollouts

def get_args():
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

    # Rollout Data
    parser.add_argument(
        "--n_steps",
        type=int,
        default=32,
        metavar="n_steps",
        help="rollout data length (default: 32)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        metavar="LR",
        help="learning rate (default: 0.0003)",
    )

    parser.add_argument("--epochs", default=20, type=int)

    # Reward Config
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.90,
        metavar="gamma",
        help="Gamma in Reinforcement Learning",
    )

    # Continue training
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to current checkpoint to continue training",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Whether want to logging onto Wandb",
    )

    parser.add_argument(
        "--actor_device_cpu",
        default=True,
        type=bool,
        help="Whether want to use CPU for actor",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size for training",
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for training",
    )
        
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    env = Match3Env(90)

    PPO_trainer = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.lr,
        # n_steps=args.n_steps,
        n_steps=4000,
        gamma=args.gamma,
        ent_coef=0.00001,
        # policy_kwargs={
        #     "net_arch": dict(pi=args.pi, vf=args.vf),
        #     "features_extractor_class": M3CnnLargerFeatureExtractor,
        #     "features_extractor_kwargs": {
        #         "mid_channels": args.mid_channels,
        #         "out_channels": 161,
        #         "num_first_cnn_layer": args.num_first_cnn_layer,
        #     },
        #     "optimizer_class": torch.optim.Adam,
        #     "share_features_extractor": False,
        # },
        policy_kwargs={
            "net_arch": dict(pi=args.pi, vf=args.vf),
            "features_extractor_class": ResNet,
            "features_extractor_kwargs": {
                "mid_channels": args.mid_channels,
                "out_channels": 161,
                "num_first_cnn_layer": args.num_first_cnn_layer,
                "resnet_variant": "resnet50",  
            },
            "optimizer_class": torch.optim.Adam,
            "share_features_extractor": False,
        },
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        _checkpoint=args.checkpoint,
        _wandb=args.wandb,
        device="cpu",
        # "cuda"
        prefix_name=args.prefix_name,
        # actor_device_cpu=args.actor_device_cpu,
    )
    
    PPO_trainer.policy.share_memory()
    run_i = 0
    while run_i < 2:
        run_i += 1
        s_t = time.time()  
        # print(PPO_trainer.rollout_buffer.observations)
            
        _, num_completed_games, num_win_games = collect_rollouts(PPO_trainer)
        win_rate = num_win_games / num_completed_games * 100
        print(f"collect data: {time.time() - s_t}\nwin rate: {win_rate}")
        
        
        s_t = time.time()
        PPO_trainer.train(
            num_completed_games=num_completed_games, num_win_games=num_win_games
        )
        
        print("training time", time.time() - s_t)
