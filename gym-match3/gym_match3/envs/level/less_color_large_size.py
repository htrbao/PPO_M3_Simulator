from gym_match3.envs.levels import Level, base_hp
from gym_match3.envs.constants import GameObject
from gym_match3.envs.game import Point, Board, DameMonster, BoxMonster


MY_LEVEL = [
    Level(10,9,3, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, GameObject.monster_dame, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0],
        [0, 0, GameObject.monster_dame, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0],
        [0, 0, GameObject.monster_dame, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0],
        [0, 0 ,0, 0, 0, 0, 0, 0, 0],
        [0, 0 ,0, 0, 0, 0, 0, 0, 0],
        [0, 0 ,0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position= Point(4,2),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=3,
                    height=3,
                    hp=25  + base_hp,
                    dame=2
        )
    ]),
    
    Level(10, 9, 4, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(1, 3),
                    relax_interval=2,
                    setup_interval=1,
                    width=3,
                    height=3,
                    hp=15 + base_hp,
                    dame=3,
                    )
    ]),

    Level(10, 9, 4, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(3, 4),
                    relax_interval=1,
                    setup_interval=1,
                    width=2,
                    height=4,
                    hp=12 + base_hp,
                    dame=2,
                    )
    ]),
        Level(10,9, 3, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0 ,0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(5,4),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=10 + base_hp,
                    dame=2,
                    )
    ]),
    Level(10,9, 3, [       
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0 ,0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(0, 6),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=15 + base_hp,
                    dame=2,
                    )
    ]),
    Level(10,9, 4, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0 ,0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(5,4),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=20 + base_hp,
                    dame=2,
                    )
    ]),
    
    Level(10, 9, 4, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, GameObject.monster_dame, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0],
        [0, 0, GameObject.monster_dame, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(8, 2),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=3,
                    height=3,
                    hp=12 + base_hp,
                    dame=2,
                    )
    ]),
    Level(10, 9, 4, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0 ,0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(2, 0),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=3,
                    height=3,
                    hp=15 + base_hp,
                    dame=2,
                    )    
    ]),
    
    Level(10, 9, 4, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],    
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
    ],[
        DameMonster(position=Point(8, 0),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=20 + base_hp,
                    dame=2,
        )
    ]),
        Level(10, 9, 4, [
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],    
    ],[
        DameMonster(position=Point(0, 0),
                    relax_interval = 2,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=20 + base_hp,
                    dame=2,
        )
    ]),
]

level_1 = MY_LEVEL