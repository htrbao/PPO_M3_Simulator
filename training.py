import argparse
import torch

from gym_match3.envs.match3_env import Match3Env
from gym_match3.envs.levels import Match3Levels, LEVELS
from training.ppo import PPO
from training.m3_model.m3_cnn import M3CnnFeatureExtractor
from configs import args

env = Match3Env(90)

print(env.observation_space)
print(env.action_space)

PPO_trainer = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=args.LR,
    vf_coef=args.VF_COEF,
    n_steps=args.N_STEPS,
    batch_size=args.BATCH_SIZE,
    gamma=args.GAMMA,
    ent_coef=args.ENTROPY_COEFF,
    policy_kwargs={
        "net_arch": dict(pi=args.PI, vf=args.VF),
        "features_extractor_class": M3CnnFeatureExtractor,
        "features_extractor_kwargs": {
            "mid_channels": args.MID_CHANNELS,
            "out_channels": 161,
            "num_first_cnn_layer": args.NUM_FIRST_CNN_LAYERS,
        },
        "optimizer_class": torch.optim.Adam,
        "share_features_extractor": args.SHARE_FEATURES_EXTRACTOR,
    },
    _checkpoint=args.CHECKPOINTS,
    _wandb=args.WANDB,
    device=args.DEVICE,
    prefix_name=args.PREFIX_NAME,
)

while True:
    import time

    s_t = time.time()
    PPO_trainer.collect_rollouts(
        PPO_trainer.env, PPO_trainer.rollout_buffer, PPO_trainer.n_steps
    )
    print("collect data", time.time() - s_t)
    s_t = time.time()
    PPO_trainer.train()
    print("training time", time.time() - s_t)
