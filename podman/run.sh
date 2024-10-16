##!/bin/bash
git pull
git checkout "$GIT_BRANCH"
git pull

cd /PPO_M3_Simulator/gym-match3
pip install -e .

cd /PPO_M3_Simulator
pip install -r requirements.txt

python training.py \
  --prefix_name milestone_mha_penalty_not_both_harder_ta_advice \
  --kernel_size "$KERNEL_SIZE" \
  --mid_channels "$MID_CHANNELS" \
  --num_first_cnn_layer "$NUM_FIRST_CNN_LAYER" \
  --num_self_attention_layers "$NUM_SELF_ATTENTION_LAYERS" \
  --strategy "$STRATEGY" \
  --pi "$PI" \
  --vf "$VF" \
  --epochs "$EPOCHS" \
  --ent_coef "$ENT_COEF" \
  --num_heads "$NUM_HEADS" \
  --n_steps "$N_STEPS" \
  --obs-order "$OBS_ORDER" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LEARNING_RATE" \
  --gamma "$GAMMA" \
  --n_updations "$N_UPDATIONS" \
  "${EXTRA1:-}" \
  "${EXTRA2:-}" \
  "${EXTRA3:-}" \
  "${EXTRA4:-}" \
  "${EXTRA5:-}" \
  --wandb \
  --num_envs "$NUM_ENVS" >>"logging/$MODEL_NAME.log"
