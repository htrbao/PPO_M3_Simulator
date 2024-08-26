CUDA_VISIBLE_DEVICES=1 python training.py \
    --prefix_name 85gamma_new_levels_decrease_s_input_p256_180_180_161_v256_256_32 \
    --pi 256 180 180 161 \
    --vf 256 256 32 \
    --obs-order none_tile color_1 color_2 color_3 color_4 color_5 pu disco bomb missile_h missile_v plane blocker monster monster_match_dmg_mask monster_inside_dmg_mask self_dmg_mask legal_action \
    --mid_channels 32 \
    --num_first_cnn_layer 10 \
    --n_steps 32768 \
    --lr 0.00002 \
    --gamma 0.85 \
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