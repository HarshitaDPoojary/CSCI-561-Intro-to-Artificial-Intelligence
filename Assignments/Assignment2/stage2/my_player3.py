import os
import copy
import time
import random
import json
import numpy as np  

BOARD_SIZE = 5
KOMI = 2.5
INPUT = 'input.txt'
OUTPUT = 'output.txt'
ZOBRIST_TABLE = [[random.randint(1, 2**64 - 1) for _ in range(3)] for _ in range(BOARD_SIZE * BOARD_SIZE)]
TRANSPOSITION_TABLE = {}
HISTORY = {}
KILLER_MOVES = {}

def read_input(input_file):
    input_info = list()
    with open(input_file, 'r') as F:
        for line in F.readlines():
            input_info.append(line.strip())

    color = int(input_info[0])
    prev_board = [[int(val) for val in line] for line in input_info[1:BOARD_SIZE+1]]
    board = [[int(val) for val in line] for line in input_info[BOARD_SIZE+1: 2*BOARD_SIZE+1]]

    return color, board, prev_board

def write_output(output_file, move):
    with open(output_file, 'w') as F:
        if move == 'PASS':
            F.write(move)
        else:
           F.write(str(move[0])+','+str(move[1]))

def heuristic(board, np):
    player, opponent, heur_player, heur_opponent = 0, 0, 0, 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == color:
                player += 1
                heur_player += (player + cluster_liberty(board, i, j))
            elif board[i][j] == 3 - color:
                opponent += 1
                heur_opponent += (opponent + cluster_liberty(board, i, j))
    if np == 2:
        heur_player += KOMI
    else:
        heur_opponent += KOMI    

    if np == color:
        return heur_player - heur_opponent
    return heur_opponent - heur_player

X_CHANGES = [1, 0, -1, 0]
Y_CHANGES = [0, 1, 0, -1]


def calculate_euler_number(game_state, side):
    opponent_side = 3 - color
    new_game_state = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            new_game_state[i + 1][j + 1] = game_state[i][j]

    q1_side = 0
    q2_side = 0
    q3_side = 0
    q1_opponent_side = 0
    q2_opponent_side = 0
    q3_opponent_side = 0

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            new_game_sub_state = new_game_state[i: i + 2, j: j + 2]
            q1_side += count_q1(new_game_sub_state, side)
            q2_side += count_q2(new_game_sub_state, side)
            q3_side += count_q3(new_game_sub_state, side)
            q1_opponent_side += count_q1(new_game_sub_state, opponent_side)
            q2_opponent_side += count_q2(new_game_sub_state, opponent_side)
            q3_opponent_side += count_q3(new_game_sub_state, opponent_side)

    return (q1_side - q3_side + 2 * q2_side - (q1_opponent_side - q3_opponent_side + 2 * q2_opponent_side)) / 4

def count_q1(game_sub_state, side):
    if ((game_sub_state[0][0] == side and game_sub_state[0][1] != side
            and game_sub_state[1][0] != side and game_sub_state[1][1] != side)
            or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
                and game_sub_state[1][0] != side and game_sub_state[1][1] != side)
            or (game_sub_state[0][0] != side and game_sub_state[0][1] != side
                and game_sub_state[1][0] == side and game_sub_state[1][1] != side)
            or (game_sub_state[0][0] != side and game_sub_state[0][1] != side
                and game_sub_state[1][0] != side and game_sub_state[1][1] == side)):
        return 1
    else:
        return 0

def count_q2(game_sub_state, side):
    if ((game_sub_state[0][0] == side and game_sub_state[0][1] != side
            and game_sub_state[1][0] != side and game_sub_state[1][1] == side)
            or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
                and game_sub_state[1][0] == side and game_sub_state[1][1] != side)):
        return 1
    else:
        return 0

def count_q3(game_sub_state, side):
    if ((game_sub_state[0][0] == side and game_sub_state[0][1] == side
            and game_sub_state[1][0] == side and game_sub_state[1][1] != side)
            or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
                and game_sub_state[1][0] == side and game_sub_state[1][1] == side)
            or (game_sub_state[0][0] == side and game_sub_state[0][1] != side
                and game_sub_state[1][0] == side and game_sub_state[1][1] == side)
            or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
                and game_sub_state[1][0] == side and game_sub_state[1][1] == side)):
        return 1
    else:
        return 0


def evaluate_game_state(game_state, side):
    opponent_side = 3 - color
    side_count = 0
    side_liberty = set()
    opponent_count = 0
    opponent_liberty = set()
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if game_state[i][j] == side:
                side_count += 1
            elif game_state[i][j] == opponent_side:
                opponent_count += 1
            else:
                for index in range(len(X_CHANGES)):
                    new_i = i + X_CHANGES[index]
                    new_j = j + Y_CHANGES[index]
                    if 0 <= new_i < BOARD_SIZE and 0 <= new_j < BOARD_SIZE:
                        if game_state[new_i][new_j] == side:
                            side_liberty.add((i, j))
                        elif game_state[new_i][new_j] == opponent_side:
                            opponent_liberty.add((i, j))

    side_edge_count = 0
    opponent_side_edge_count = 0
    for j in range(BOARD_SIZE):
        if game_state[0][j] == side or game_state[BOARD_SIZE - 1][j] == side:
            side_edge_count += 1
        if game_state[0][j] == opponent_side or game_state[BOARD_SIZE - 1][j] == opponent_side:
            opponent_side_edge_count += 1

    for j in range(1, BOARD_SIZE - 1):
        if game_state[j][0] == side or game_state[j][BOARD_SIZE - 1] == side:
            side_edge_count += 1
        if game_state[j][0] == opponent_side or game_state[j][BOARD_SIZE - 1] == opponent_side:
            opponent_side_edge_count += 1

    center_unoccupied_count = 0
    for i in range(1, BOARD_SIZE - 1):
        for j in range(1, BOARD_SIZE - 1):
            if game_state[i][j] == 0:
                center_unoccupied_count += 1

    score = min(max((len(side_liberty) - len(opponent_liberty)), -8), 8) + (
            -4 *  calculate_euler_number(game_state, side)) + (
                    5 * (side_count - opponent_count)) - (9 * side_edge_count * (center_unoccupied_count / 9))
    if side == 2:
        score += KOMI
    return score



def find_dead_stones(board, color):
    dead_stones = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == color:
                if not cluster_liberty(board, i, j) and (i,j) not in dead_stones:
                    dead_stones.append((i, j))
    return dead_stones

def remove_stones(board, locs):
    for stone in locs:
        board[stone[0]][stone[1]] = 0
    return board

def remove_dead_stones(board, color):
    dead_stones = find_dead_stones(board, color)
    if not dead_stones:
        return board
    new_board = remove_stones(board, dead_stones)
    return new_board

def find_adjacent_stones(board, row, col):
    board = remove_dead_stones(board, (row, col))
    neighboring = [(row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1)]
    return ([point for point in neighboring if 0 <= point[0] < BOARD_SIZE and 0 <= point[1] < BOARD_SIZE])

def find_ally_neighbors(board, row, col):
    allies = list()
    for point in find_adjacent_stones(board, row, col):
        if board[point[0]][point[1]] == board[row][col]:
            allies.append(point)

    return allies

def find_ally_cluster(board, row, col):
    queue = [(row, col)]
    cluster = list()
    
    while queue:
        node = queue.pop(0)
        cluster.append(node)
        for neighbor in find_ally_neighbors(board, node[0], node[1]):
            if neighbor not in queue and neighbor not in cluster:
                queue.append(neighbor)
    return cluster

def cluster_liberty(board, row, col):
    count = 0
    for point in find_ally_cluster(board, row, col):
        for neighbor in find_adjacent_stones(board,  point[0], point[1]):
            if board[neighbor[0]][neighbor[1]] == 0:
                count += 1

    return count

def ko_(prev_board, board):
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != prev_board[i][j]:
                return False
    return True


def good_move(board, prev_board, player, row, col):
    if board[row][col] != 0:
        return False
    board_copy = copy.deepcopy(board)
    board_copy[row][col] = player
    dead_pieces = find_dead_stones(board_copy, 3 - player)
    board_copy = remove_dead_stones(board_copy, 3 - player)
    if cluster_liberty(board_copy, row, col) >= 1 and not (dead_pieces and ko_(prev_board, board_copy)):
        return True


def find_valid_moves(board, prev_board, player):
    valid_moves = list()
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if good_move(board, prev_board, player, i, j) == True:
                valid_moves.append((i,j))
    return valid_moves

def zobrist_hash(board):
    h = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != 0:
                h ^= ZOBRIST_TABLE[i * BOARD_SIZE + j][board[i][j]-1]
    return h

def save_transposition_table():
    with open('transposition_table.json', 'w') as f:
        json.dump(TRANSPOSITION_TABLE, f)

def rotate_90(board):
    return [list(reversed(col)) for col in zip(*board)]

def symmetry_lookup(board):
    rotations = [board]
    for _ in range(3):
        board = rotate_90(board)
        rotations.append(board)
    for r in rotations:
        h = zobrist_hash(r)
        if h in TRANSPOSITION_TABLE:
            return TRANSPOSITION_TABLE[h]
    return None


def update_history(move, depth):
    HISTORY[move] = HISTORY.get(move, 0) + (2 ** depth)

def sibling_promotion(move):
    row, col = move
    siblings = [(row+1, col), (row-1, col), (row, col+1), (row, col-1)]
    return siblings

def pvs(curr_state, prev_state, depth, alpha, beta, color):
    if depth == 0:
        return evaluate_game_state(curr_state, color)
    
    hash_value = zobrist_hash(curr_state)
    if hash_value in TRANSPOSITION_TABLE:
        return TRANSPOSITION_TABLE[hash_value]

    if depth >= 2: 
        valid_moves_etc = find_valid_moves(curr_state, prev_state, color)
        for move in valid_moves_etc:
            next_state = make_move(curr_state, move, color)
            hash_value = zobrist_hash(next_state)
            if hash_value in TRANSPOSITION_TABLE and TRANSPOSITION_TABLE[hash_value] >= beta:
                return TRANSPOSITION_TABLE[hash_value]


    sym_value = symmetry_lookup(curr_state)
    if sym_value is not None:
        return sym_value
    
    valid_moves = find_valid_moves(curr_state, prev_state, color)
    
    valid_moves.sort(key=lambda m: (HISTORY.get(m, 0), KILLER_MOVES.get(depth, {}).get(m, 0)), reverse=True)
    
    is_first_move = True
    for move in valid_moves:
        next_state = make_move(curr_state, move, color)
        if is_first_move:
            score = -pvs(next_state, curr_state, depth - 1, -beta, -alpha, 3 - color)
            is_first_move = False
        else:
            score = -pvs(next_state, curr_state, depth - 1, -alpha - 1, -alpha, 3 - color)
            if alpha < score and score < beta:
                score = -pvs(next_state, curr_state, depth - 1, -beta, -alpha, 3 - color)
        alpha = max(alpha, score)
        if alpha >= beta:
            KILLER_MOVES[depth] = move
            update_history(move, depth)
            break

    TRANSPOSITION_TABLE[hash_value] = alpha
    return alpha

def make_move(board, move, player):
    new_board = copy.deepcopy(board)
    new_board[move[0]][move[1]] = player
    new_board = remove_dead_stones(new_board, 3-player)
    return new_board




def search_best_move(curr_state, prev_state, depth, color):
    best_score = float('-inf')
    best_move_found = None

    valid_moves = find_valid_moves(curr_state, prev_state, color)

    
    while valid_moves:
        move = valid_moves.pop(0) 
        next_state = make_move(curr_state, move, color)
        score = -pvs(next_state, curr_state, depth-1, float('-inf'), float('inf'), 3 - color)
        
        if score > best_score:
            best_score = score
            best_move_found = move

            for sibling in sibling_promotion(best_move_found):
                if sibling in valid_moves:
                    valid_moves.remove(sibling)
                    valid_moves.insert(0, sibling)

    return best_move_found

import time

TIME_LIMIT = 9.0  
def best_move(curr_state, prev_state, color):

    best_move_found = None
    start_time = time.time()
    depth = 1

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > TIME_LIMIT:
            break
        
        current_best = search_best_move(curr_state, prev_state, depth, color)
        if current_best:
            best_move_found = current_best

        depth += 1

    return best_move_found

start = time.time()
color, cur_board, pre_board = read_input(INPUT)
action = best_move(cur_board, pre_board, color)
if not action:
    action = 'PASS'

write_output(OUTPUT, action)
end = time.time()
save_transposition_table()
print(f'total time of evaluation: {end-start}')