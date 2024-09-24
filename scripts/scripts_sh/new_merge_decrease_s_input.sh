CUDA_VISIBLE_DEVICES=1 python training.py \
    --prefix_name test \
    --pi 512 512 \
    --vf 512 256 128 64 32 16 \
    --mid_channels 32 \
    --num_first_cnn_layer 10 \
    --n_steps 32768 \
    --lr 0.00002 \
    --gamma 0.85 \
    --num_envs 8


    # --obs-order none_tile color_1 color_2 color_3 color_4 color_5 disco bomb missile_h missile_v plane blocker monster monster_match_dmg_mask monster_inside_dmg_mask self_dmg_mask legal_action \