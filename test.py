from gym_match3.envs.match3_env import Match3Env
from gym_match3.envs.levels import Match3Levels, LEVELS

env = Match3Env(
    90,
    obs_order=obs_order,
    level_group=(rank * num_per_group, (rank + 1) * num_per_group),
)