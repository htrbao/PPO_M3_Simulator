##!/bin/bash
#git reset --hard HEAD
git pull
echo $BATCH_SIZE
echo $LEARNING_RATE
echo $GAMMA
echo $NUM_ENVS

python training.py \
  --prefix_name milestone_mha_penalty_not_both_harder_ta_advice \
  --strategy milestone \
  --pi 256 128 128 128 128 128 128 128 128 128 \
  --vf 4096 256 256 256 256 256 256 256 256 256 \
  --num_heads 2 \
  --n_steps 131072 \
  --obs-order none_tile color_1 color_2 color_3 color_4 color_5 pu disco bomb missile_h missile_v plane blocker monster monster_match_dmg_mask monster_match_hp monster_inside_dmg_mask monster_inside_hp self_dmg_mask match_normal match_2x2 match_4_v match_4_h match_L match_T match_5 legal_action heat_mask \
  --batch_size $BATCH_SIZE \
  --lr $LEARNING_RATE \
  --gamma $GAMMA \
  --n_updations 1 \
  --num_envs $NUM_ENVS >>"logging/{$MODEL_NAME}.log"
