import os
import argparse
import time
import torch
import numpy as np

from gym_match3.envs.match3_env import Match3Env
from gym_match3.envs.levels import Match3Levels, LEVELS
from training.common.vec_env import SubprocVecEnv
from training.ppo import PPO
from training.m3_model.m3_cnn import (
    M3CnnFeatureExtractor,
    M3CnnLargerFeatureExtractor,
    M3CnnWiderFeatureExtractor,
    M3SelfAttentionFeatureExtractor,
    M3ExplainationFeatureExtractor,
    M3MlpFeatureExtractor,
    M3LocFeatureExtractor
)


def get_args():
    parser = argparse.ArgumentParser(
        "Match3 with PPO",
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
        "--obs-order",
        type=str,
        default=[],
        nargs="+",
        help="Which features you want to use?",
    )
    parser.add_argument(
        "--prefix_name",
        type=str,
        default="m3_with_cnn",
        help="prefix name of the model",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Number of kernel size in CNN model",
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
        "--num_self_attention_layers",
        type=int,
        default=8,
        help="Number of intermediary layers in CNN model",
    )

    # Rollout Data
    parser.add_argument(
        "--strategy",
        type=str,
        default="sequential",
        help="Strategy increasing the number of levels",
    )
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
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--ent_coef", default=0.01, type=float)

    # Reward Config
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.98,
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

    # Number of parallel environments
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="Number of parallel environments to run (default: 4)",
    )
    parser.add_argument(
        "--render",
        action='store_true',
        help="initiate renderer object",
    )

    return parser.parse_args()


def make_env(obs_order, level_group, render):
    def _init():
        env = Match3Env(
            90,
            obs_order=obs_order,
            level_group=level_group,
            is_render=render,
        )
        return env

    return _init

def make_env_loc(args, milestones=0, step=4, render=False):
    max_level = min(len(LEVELS), args.num_envs + step*milestones)
    
    r = max_level % args.num_envs
    d = max_level // args.num_envs
    num_keeps = args.num_envs - r

    envs = SubprocVecEnv(
        [
            make_env(args.obs_order, (i * d, (i + 1) * d), render)
            for i in range(num_keeps)
        ]
        
        + 
        
        [
            make_env(args.obs_order, (num_keeps * d + i * (d + 1), num_keeps * d + (i + 1) * (d + 1)), render)
            for i in range(r)
        ]
    )
    return envs


def main():
    args = get_args()
    max_level = len(LEVELS)
    envs = None
    milestone = 0
    if args.strategy == 'sequential':
        envs = SubprocVecEnv(
            [
                make_env(i, args.obs_order, max_level // args.num_envs, args.render)
                for i in range(args.num_envs)
            ]
        )
    elif args.strategy == 'milestone':
        envs = make_env_loc(args, milestones=milestone, render=args.render)
    else:
        raise ValueError(f'Invalid strategy: {args.strategy}')

    print(f"Agent will be train on {len(LEVELS)} levels")
    print("Observation Space:", envs.observation_space)
    print("Action Space", envs.action_space)
    PPO_trainer = PPO(
        policy="CnnPolicy",
        env=envs,
        learning_rate=args.lr,
        n_steps=args.n_steps // args.num_envs,
        gamma=args.gamma,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        policy_kwargs={
            "net_arch": dict(pi=args.pi, vf=args.vf),
            "features_extractor_class": M3CnnWiderFeatureExtractor,
            "features_extractor_kwargs": {
                "kernel_size": args.kernel_size,
                "start_channel": 32,
                "mid_channels": args.mid_channels,
                "out_channels": 256,
                "num_first_cnn_layer": args.num_first_cnn_layer,
                "num_self_attention_layers": args.num_self_attention_layers,
                "layers_dims": [4096, 2048, 2048, 2048],
                "max_channels": 128,
                "size": 9*10
            },
            "optimizer_class": torch.optim.AdamW,
            "share_features_extractor": False,
            "activation_fn": torch.nn.GELU
        },
        _checkpoint=args.checkpoint,
        _wandb=args.wandb,
        device="cuda",
        seed=13,
        prefix_name=args.prefix_name,
    )
    print("trainable parameters", sum(p.numel() for p in PPO_trainer.policy.parameters() if p.requires_grad))
    run_i = 0
    print(PPO_trainer.n_steps)
    while run_i < 700:
        run_i += 1
        s_t = time.time()
        res = (
            PPO_trainer.collect_rollouts(
                PPO_trainer.env, PPO_trainer.rollout_buffer, PPO_trainer.n_steps,
                num_levels = len(LEVELS)
            )
        )
        # extract stat
        num_win_games = res.get('num_win_games', None)
        num_completed_games = res.get('num_completed_games', None)
        num_damage = res.get('num_damage', None)
        num_hit = res.get('num_hit', None)
        win_list = res.get('win_list', None)
        hit_mask = res.get('hit_mask', None)

        if not os.path.isdir(f'./_saved_stat/hit_mask/{PPO_trainer._model_name}'):
            os.makedirs(f'./_saved_stat/hit_mask/{PPO_trainer._model_name}')
        with open(f'./_saved_stat/hit_mask/{PPO_trainer._model_name}/{run_i}.npy', 'wb') as f:
            np.save(f, hit_mask)

        win_rate = num_win_games / num_completed_games * 100
        print(f"collect data: {time.time() - s_t}\nwin rate: {win_rate}\nmilestone: {milestone}")
        print(f"{win_list}")
        s_t = time.time()
        PPO_trainer.train(
            num_completed_games=num_completed_games,
            num_win_games=num_win_games,
            num_damage=num_damage,
            num_hit=num_hit,
            win_list=win_list
        )
        print("training time", time.time() - s_t)
        if args.strategy == 'milestone' and win_rate > 80.0:
            milestone += 1
            envs.close()
            envs = make_env_loc(args, milestone, step=10, render=args.render)
            PPO_trainer.set_env(envs)
            PPO_trainer.set_random_seed(13)

if __name__ == "__main__":
    
    main()
