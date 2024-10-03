from gym_match3.envs.levels import Level, base_hp
from gym_match3.envs.constants import GameObject
from gym_match3.envs.game import Point, Board, DameMonster, BoxMonster
MY_LEVEL = [
    Level(10, 9, 5, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(6, 6),
                    relax_interval = 4,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=20+base_hp,
                    dame=2,
                    have_paper_box=True
                    )
    ]),

    Level(10, 9, 5, [
        [0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0],
        [0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(0, 0),
                    relax_interval = 3,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=20+base_hp,
                    dame=2,
                    have_paper_box=True
                    )
    ]),
        Level(10, 9, 5, [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(6, 6),
                    relax_interval = 4,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=20+base_hp,
                    dame=2,
                    have_paper_box=True
                    )
    ]),
    Level(10, 9, 5, [
        [-1, -1, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, -1, -1],
    ], [
        DameMonster(position=Point(4, 4),
                    relax_interval = 4,
                    setup_interval = 1,
                    width=2,
                    height=2,
                    hp=30+base_hp,
                    dame=2,
                    have_paper_box=True
                    )
    ]),
    Level(10, 9, 5, [
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, GameObject.monster_dame, 0, 0, 0, 0],
    [0, 0, 0, 0, GameObject.monster_dame, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(4, 4),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=30+base_hp,
                    dame=2,
                    have_paper_box=True
                    )
    ]),
    Level(10, 9, 5, [
    [0, 0, 0, 0, 0, 0, 0, -1, -1],
    [0, 0, 0, 0, 0, 0, -1, -1, 0],
    [0, 0, 0, 0, 0, -1, -1, 0, 0],
    [0, 0, 0, 0, -1, -1, 0, 0, 0],
    [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
    [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
    DameMonster(position=Point(4, 4),
                relax_interval=3,
                setup_interval=2,
                width=2,
                height=2,
                hp=20+base_hp,
                dame=3,
                have_paper_box=True
                ),
    ]),
    Level(10, 9, 5, [
    [-1, -1, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, -1, -1],
    [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
    [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], [
        DameMonster(position=Point(6, 4),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=20+base_hp,
                    dame=2,
                    have_paper_box=True
                    )
    ]),
    
    Level(10, 9, 5, [
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
        DameMonster(position=Point(0,0),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=20+base_hp,
                    dame=2,
                    have_paper_box=True
                    )
    ]),
    
    Level(10, 9, 5, [
        [-1, -1, -1 , -1, -1, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0], 
        [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],[
        DameMonster(position=Point(4,0),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=30+base_hp,
                    dame=2,
                    have_paper_box=True
                    )
    ]),
    Level(10, 9, 5, [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
    [0, 0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0],
], [
    DameMonster(position=Point(8, 4),
                relax_interval=2,
                setup_interval=1,
                width=2,
                height=2,
                hp=18+base_hp,
                dame=3,
                have_paper_box=True
                )
])

    
]
