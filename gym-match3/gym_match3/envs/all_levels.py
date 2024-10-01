import os
import importlib

dict_levels= {
}

for dir in os.listdir('level'):
    try:
        module = importlib.import_module('level.'+dir.replace('.py',''))
        print(module)
        dict_levels[dir.replace('.py','')] = module.MY_LEVEL
    except Exception as e:
        print(f"Module {dir} {e} not found.")
# dict_keys(['5_color_tanky', 'multiple_mons', 'less_color_large_size', 'with_obstacle', 'no_dame_direction_yes_pu', 'no_matches', 'corners'])
print(dict_levels.keys())
ALL_LEVELS=[
    *dict_levels['less_color_large_size'],
    *dict_levels['5_color_tanky'],
    *dict_levels['corners'],
    *dict_levels['with_obstacle'],
    *dict_levels['multiple_mons_no_dame_direction']
    *dict_levels['multiple_mons'],
    *dict_levels['no_dame_direction_yes_pu'],
    *dict_levels['no_matches'],
]