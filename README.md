# M3 Simulator

Thanks for [kamildar/gym-match3](https://github.com/kamildar/gym-match3) publish the baseline of building gymnasium match-3 games

## Configurations
<p align="left">
 <a href=""><img src="https://img.shields.io/badge/python-3.9-aff.svg"></a>
</p>

## M3 Simulator Setup
1. You need to clone this repo into local first:
    ```bash
    git clone https://github.com/htrbao/PPO_M3_Simulator
    ```

2. Install the requirements
    - Using conda:
        ```bash
        conda create -n m3_simu python=3.9
        conda activate m3_simu
        conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
        cd gym-match3
        pip install -e .
        pip install -r requirements.txt

        ```

    - Using venv:
        ```bash
        python -m venv ./venv
        path/to/venv/Scripts/activate
        pip3 install torch --index-url https://download.pytorch.org/whl/cu118
        cd gym-match3
        pip install -e .
        pip install -r requirements.txt
        ```

## M3 Simulator Usage

1. Please refer to this [notebook](gym-match3\test_play.ipynb) to know how to use gym-match3 environment.

## M3 Simulator Levels Configs
1. You can also contribute your custom levels by append it into `LEVELS` list within [gym_match3.envs.levels](./gym-match3/gym_match3/envs/levels.py)

    For example:
    ```python
    Level(h=10, w=9, n_shape=6, board=[
        [-1, -1, -1, -1,  0, -1, -1, -1, -1],
        [-1, -1, -1,  0,  0,  0, -1, -1, -1],
        [-1, -1,  0,  0,  0,  0,  0, -1, -1],
        [-1,  0,  0,  0,  0,  0,  0,  0, -1],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
        [-1,  0,  0,  0,  0,  0,  0,  0, -1],
        [-1, -1,  0,  0,  0,  0,  0, -1, -1],
        [-1, -1, -1,  0,  0,  0, -1, -1, -1],
        [-1, -1, -1, -1,  0, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    ])


    ```

## M3 Training
- To understand board on `wandb`, you can follow [stackexchange](https://datascience.stackexchange.com/questions/115243/understanding-the-tensorboard-plots-on-a-stable-baseline3s-ppo).