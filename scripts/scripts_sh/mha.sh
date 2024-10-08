CUDA_VISIBLE_DEVICES=1 python training.py \
    --prefix_name milestone_mha_penalty_not_create_pu \
    --strategy milestone \
    --pi 256 128 128 128 128 128 128 128 128 128 \
    --vf 4096 256 256 256 256 256 256 256 256 256 \
    --num_heads 2 \
    --n_steps 131072 \
    --batch_size 1024 \
    --lr 0.000012 \
    --gamma 0.985 \
    --num_envs 64 \
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