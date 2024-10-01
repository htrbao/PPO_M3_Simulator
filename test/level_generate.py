from test.global_config import REALMS_CONFIG
from test.utils import get_map_infos

def get_real_levels():
    REAL_LEVELS = []
    count_loop = 0
    for realm, realm_infos in REALMS_CONFIG.items():
        for realm_info in realm_infos:
            level = get_map_infos(realm_info)
            if level is not None:
                REAL_LEVELS.append(level)
            count_loop += 1


    print(f"\t\tSUCCESS: load & process {len(REAL_LEVELS)}/{count_loop} levels")
    return REAL_LEVELS