# About the meaning of the game state which format in match3_helper

Table of Features which extracted from the game state:

| Features | Description        |
| -------- | ------------------ |
| none_tile| The position without anything|
| color_1| Where `color_1` be placed |
| color_2| Where `color_2` be placed |
| color_3| Where `color_3` be placed |
| color_4| Where `color_4` be placed |
| color_5| Where `color_5` be placed |
| disco| Where `disco` be placed |
| bomb| Where `bomb` be placed |
| missile_h| Where `missile_h` be placed |
| missile_v| Where `missile_v` be placed |
| plane| Where `plane` be placed |
| pu| Where `pu` be placed |
| blocker| Where `blocker` be placed |
| monster| Where `monster` be placed |
| monster_match_dmg_mask| Where can use match to attack monsters |
| monster_inside_dmg_mask| Where can use Power Up to attack monsters |
| self_dmg_mask| the place where when you match next to, you will take damage |
| match_normal| Clarify possible `match 3` positions |
| match_2x2| Clarify possible `match_2x2` positions |
| match_4_v| Clarify possible `match_4_v` positions |
| match_4_h| Clarify possible `match_4_h` positions |
| match_L| Clarify possible `match_L` positions |
| match_T| Clarify possible `match_T` positions |
| match_5| Clarify possible `match_5` positions |
| legal_action| Where you can swap tiles to make the match |
| heat_mask| The mask that helps model paying more attention |