python training.py --prefix_name multienv_85gamma_full_input --pi 512 512 512 512 512 512 --vf 1024 512 256 512 256 128 64 32 16 8 4 2 --mid_channels 32 --num_first_cnn_layer 10 --n_steps 32768 --lr 0.00002 --gamma 0.85 --num_envs 8 --wandb

@REM # REMEMBER TO CHANGE LOGIC OF HELPER
@REM # "disco": (board == GameObject.power_disco) * 4.5,
@REM # "bomb": (board == GameObject.power_bomb) * 2.5,
@REM # "missile_h": (board == GameObject.power_missile_h) * 1.0,
@REM # "missile_v": (board == GameObject.power_missile_v) * 1.5,
@REM # "plane": (board == GameObject.power_plane) * 2,
@REM # "buff": (board == GameObject.power_disco) \
@REM #         | (board == GameObject.power_disco) \
@REM #         | (board == GameObject.power_disco),
@REM # "pu": (board == GameObject.power_disco)  \
@REM #         + (board == GameObject.power_bomb)  \
@REM #         + (board == GameObject.power_missile_h)  \
@REM #         + (board == GameObject.power_missile_v) \
@REM #         + (board == GameObject.power_plane) ,