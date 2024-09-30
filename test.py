import numpy as np

def check_for_diagonal(i, _h, _v, point, board):
    out_lst = []
    prev_point = point
    for _ in range(1, i):
        if 0 <= prev_point[0] + _h <= 9:
            cur_point = (prev_point[0] + _h, prev_point[1] + 0)
            board[cur_point] = True
            prev_point = cur_point
        else:
            break
    
    prev_point = point
    for _ in range(1, i):
        if 0 <= prev_point[1] + _v <= 8:
            cur_point = (prev_point[0] + 0, prev_point[1] + _v)
            board[cur_point] = True
            prev_point = cur_point
        else:
            break
    
    if 0 <= point[0] + _h <= 9 and 0 <= point[1] + _v <= 8:
        print("abc")
        cur_point = (point[0] + _h, point[1] + _v)
        board[cur_point] = True
        check_for_diagonal(i - 1, _h, _v, cur_point, board)

    print(i, board)
        
    return out_lst

board = np.zeros((10, 9))
point = (8, 4)
is_ne_v = is_ne_h = is_po_v = is_po_h = True
for i in range(1, 5):
    prev_coeff = i - 1
    if point[0] + i < 10:
        board[point[0] + i, point[1]] = True
    if point[0] - i >= 0:
        board[point[0] - i, point[1]] = True
    if point[1] + i < 9:
        board[point[0], point[1] + i] = True
    if point[1] - i >= 0:
        board[point[0], point[1] - i] = True


print(board)

# check for diagonal
for i in range(-1, 2, 2):
    for j in range(-1, 2, 2):
        prev_point = point
        cur_point = (point[0] + i, point[1] + j)
        board[cur_point] = True

        print(board)

        prev_point = cur_point
        print(prev_point)


        if prev_point[0] + i < 10:
            board[prev_point[0] + i, prev_point[1] + 0] = True
        
        
        if prev_point[1] + j < 9:
            board[prev_point[0] + 0, prev_point[1] + j] = True
        cur_point = (prev_point[0] + i, prev_point[1] + j)
        print(prev_point)
        print(i, j)
        board[cur_point] = True

        print(board)

        check_for_diagonal(3, i, j, cur_point, board)