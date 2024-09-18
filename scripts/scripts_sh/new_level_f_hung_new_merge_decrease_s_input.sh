CUDA_VISIBLE_DEVICES=1 python training.py \
    --prefix_name sequence_multienv_mlp_95gamma_new_reward_new_stats19 \
    --pi 512 512 512 512 256 128 \
    --vf 512 512 512 512 256 128 64 32 16 \
    --mid_channels 32 \
    --num_first_cnn_layer 10 \
    --n_steps 131072 \
    --batch_size 1024 \
    --lr 0.00003 \
    --gamma 0.95 \
    --num_envs 32 \
    --wandb

# REMEMBER TO CHANGE LOGIC OF HELPER
# "disco": (board == GameObject.power_disco) * 4.5,
# "bomb": (board == GameObject.power_bomb) * 2.5,
# "missile_h": (board == GameObject.power_missile_h) * 1.0,
# "missile_v": (board == GameObject.power_missile_v) * 1.5,
# "plane": (board == GameObject.power_plane) * 2,
# "buff": (board == GameObject.power_disco) \
#         | (board == GameObject.power_disco) \
#         | (board == GameObject.power_disco),
# "pu": (board == GameObject.power_disco)  \
#         + (board == GameObject.power_bomb)  \
#         + (board == GameObject.power_missile_h)  \
#         + (board == GameObject.power_missile_v) \
#         + (board == GameObject.power_plane) ,