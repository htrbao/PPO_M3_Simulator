from gym_match3.envs.levels import Level, base_hp
from gym_match3.envs.constants import GameObject
from gym_match3.envs.game import Point, Board, DameMonster, BoxMonster

MY_LEVEL =  [
    Level(10, 9, 5, [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, -1, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, -1, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, -1, 0, 0, -1],
            [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
            [GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, 0, 0, 0],
        ], [
            DameMonster(position=Point(8, 0), width=2, height=2, hp=15 + base_hp),
        ],
    ),
    Level(10, 9, 5, [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0,  0,  0,  0,  0,  0,  0,  0, -1],
    [0,  0,  0,  0,  0,  0,  0,  0, -1],
    [0,  0,  0,  0,  0,  0,  0,  0, -1],
    [0,  0,  0, GameObject.monster_dame, GameObject.monster_dame, 0,  0,  0, -1],
    [0,  0,  0, GameObject.monster_dame, GameObject.monster_dame, 0,  0,  0, -1],
    [0,  0,  0,  0,  0,  0,  0,  0, -1],
    [0,  0,  0,  0,  0,  0,  0,  0, -1],
    [0,  0,  0,  0,  0,  0,  0,  0, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    ], [
        DameMonster(position=Point(4, 3),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=30+base_hp,
                    dame=2
                    )
    ]), 
    Level(10, 9, 5, [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  0,  0, -1, -1, -1,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0, GameObject.monster_dame, GameObject.monster_dame, GameObject.monster_dame,  0,  0,  0],
    [ 0,  0,  0, GameObject.monster_dame, GameObject.monster_dame, GameObject.monster_dame,  0,  0,  0],
    [ 0,  0,  0, GameObject.monster_dame, GameObject.monster_dame, GameObject.monster_dame,  0,  0,  0],
    [ 0,  0,  0, -1, -1, -1,  0,  0,  0],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    ], [
        DameMonster(position=Point(5, 3),
                    relax_interval=2,
                    setup_interval=1,
                    width=3,
                    height=3,
                    hp=25+base_hp,
                    dame=2
                    )
    ]),
    Level(10, 9, 5, [
    [-1, -1,  0,  0,  0,  0,  0, -1, -1],
    [-1, -1,  0,  0,  0,  0,  0, -1, -1],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [GameObject.monster_dame, GameObject.monster_dame,  0,  0,  0,  0,  0,  0,  0],
    [GameObject.monster_dame, GameObject.monster_dame,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [-1, -1,  0,  0,  0,  0,  0, -1, -1],
    [-1, -1,  0,  0,  0,  0,  0, -1, -1]
    ]
    , [
        DameMonster(position=Point(4, 0),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=15+base_hp,
                    dame=2
                    )
    ]),
    Level(10, 9, 5, [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, -1, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0],
    [0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]
    , [
        DameMonster(position=Point(7, 3),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=30+base_hp,
                    dame=2
                    )
    ]),
    Level(10, 9, 5, [
    [-1, -1, 0, 0, 0, 0, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, 0, 0, 0, 0, 0, -1, -1],
    [-1, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, -1, -1],
    [-1, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, -1, -1],
    [-1, 0, 0, 0, 0, 0, 0, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1]
    ]
    , [
        DameMonster(position=Point(4, 3),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=30+base_hp,
                    dame=2
                    )
    ]),
    Level(10, 9, 5, [
    [0, 0, 0, 0, 0, 0, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, -1, -1, -1],
    [0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, -1, -1, -1],
    [0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0, 0, 0, 0]
    ], [
        DameMonster(position=Point(2, 2),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=30+base_hp,
                    dame=2
                    )
    ]),
    Level(10, 9, 5, [
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [GameObject.monster_dame, GameObject.monster_dame, 0, 0, -1, 0, 0, 0, 0],
    [GameObject.monster_dame, GameObject.monster_dame, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0, 0]
], [
        DameMonster(position=Point(5, 0),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=30+base_hp,
                    dame=2
                    )
    ]),
    Level(10, 9, 5, [
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [-1, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, -1],
    [-1, -1, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
], [
        DameMonster(position=Point(2, 2),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=30+base_hp,
                    dame=2
                    )
    ]),
        Level(10, 9, 5, [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, GameObject.monster_dame, GameObject.monster_dame],
    [0, 0, 0, 0, -1, 0, 0, GameObject.monster_dame, GameObject.monster_dame],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]
, [
        DameMonster(position=Point(4, 7),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=30+base_hp,
                    dame=2
                    )
    ]),
        
    Level(10, 9, 5, [
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, GameObject.monster_dame, GameObject.monster_dame, 0, 0, 0, -1],
    [0, 0, -1, GameObject.monster_dame, GameObject.monster_dame, -1, 0, 0, -1],
    [0, 0, -1, 0, 0, -1, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, -1]
], [
        DameMonster(position=Point(3, 3),
                    relax_interval=2,
                    setup_interval=1,
                    width=2,
                    height=2,
                    hp=30+base_hp,
                    dame=2
        )
    ]),
]


level_1 = MY_LEVEL