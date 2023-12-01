import copy

import numpy as np

STEP_COUNT_FILE = 'step_count.txt'
BOARD_SIZE = 5
UNOCCUPIED = 0
BLACK = 1
WHITE = 2
KOMI = 2.5
X_CHANGES = [1, 0, -1, 0]
Y_CHANGES = [0, 1, 0, -1]

def read_input(input_file_name='input.txt'):
    with open(input_file_name) as input_file:
        input_file_lines = [input_file_line.strip() for input_file_line in input_file.readlines()]
        player = int(input_file_lines[0])
        previous_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        current_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        for row in range(1, BOARD_SIZE + 1):
            for col in range(len(input_file_lines[row])):
                previous_board[row - 1][col] = input_file_lines[row][col]
        for row in range(BOARD_SIZE+1, 2*BOARD_SIZE + 1):
            for col in range(len(input_file_lines[row])):
                current_board[row - 6][col] = input_file_lines[row][col]
        return player, previous_board, current_board


def write_output(next_move, output_file_name='output.txt'):
    with open(output_file_name, 'w') as output_file:
        if next_move is None or next_move == (-1, -1):
            output_file.write('PASS')
        else:
            output_file.write(f'{next_move[0]},{next_move[1]}')


def calculate_step_count(previous_board, current_board):
    previous_board_init = True
    current_board_init = True
    for i in range(BOARD_SIZE - 1):
        for j in range(BOARD_SIZE - 1):
            if previous_board[i][j] != UNOCCUPIED:
                previous_board_init = False
                current_board_init = False
                break
            elif current_board[i][j] != UNOCCUPIED:
                current_board_init = False

    if previous_board_init and current_board_init:
        step_count = 0
    elif previous_board_init and not current_board_init:
        step_count = 1
    else:
        with open(STEP_COUNT_FILE) as step_count_file:
            step_count_str = step_count_file.readline()
            if(step_count_str is None):
                step_count = 3
            else:
                step_count = int(step_count_str)
            step_count += 2
    with open(STEP_COUNT_FILE, 'w') as step_count_file:
        step_count_file.write(f'{step_count}')

    return step_count




class MyPlayer:
    def __init__(self, player, previous_board, current_board):
        self.player = player
        self.opponent_player = self.get_opponent_player(self.player)
        self.previous_board = previous_board
        self.current_board = current_board

    def alphaBeta_search(self, search_depth, branching_factor, step_count):
        max_move, max_move_value = self.max_value(self.current_board, self.player, search_depth, 0, branching_factor,
                                                  -np.inf, np.inf, None, step_count, False)

        write_output(max_move)

    def max_value(self, game_state, player, search_depth, current_depth, branching_factor, alpha, beta, last_move,
                  step_count, is_second_pass):
        if search_depth == current_depth or step_count + current_depth == 24:
            return self.evaluate_game_state(game_state, player)
        if is_second_pass:
            return self.evaluate_game_state(game_state, player)
        is_second_pass = False
        max_move_value = -np.inf
        max_move = None
        legal_moves = self.find_legal_moves(game_state, player)
        legal_moves.append((-1, -1))
        if last_move == (-1, -1):
            is_second_pass = True
        for valid_move in legal_moves[:branching_factor]:
            opponent_player = self.get_opponent_player(player)
            if valid_move == (-1, -1):
                new_game_state = copy.deepcopy(game_state)
            else:
                new_game_state = self.move(game_state, player, valid_move)
            min_move_value = self.min_value(new_game_state, opponent_player, search_depth, current_depth + 1,
                                            branching_factor, alpha, beta, valid_move, step_count, is_second_pass)
            if max_move_value < min_move_value:
                max_move_value = min_move_value
                max_move = valid_move
            if max_move_value >= beta:
                if current_depth == 0:
                    return max_move, max_move_value
                else:
                    return max_move_value
            alpha = max(alpha, max_move_value)
        if current_depth == 0:
            return max_move, max_move_value
        else:
            return max_move_value

    def min_value(self, game_state, player, search_depth, current_depth, branching_factor, alpha, beta, last_move,
                  step_count, is_second_pass):
        if search_depth == current_depth:
            return self.evaluate_game_state(game_state, player)
        if step_count + current_depth == 24 or is_second_pass:
            return self.evaluate_game_state(game_state, self.player)
        is_second_pass = False
        min_move_value = np.inf
        legal_moves = self.find_legal_moves(game_state, player)
        legal_moves.append((-1, -1))
        if last_move == (-1, -1):
            is_second_pass = True
        for valid_move in legal_moves[:branching_factor]:
            opponent_player = self.get_opponent_player(player)
            if valid_move == (-1, -1):
                new_game_state = copy.deepcopy(game_state)
            else:
                new_game_state = self.move(game_state, player, valid_move)
            max_move_value = self.max_value(new_game_state, opponent_player, search_depth, current_depth + 1,
                                            branching_factor, alpha, beta, valid_move, step_count, is_second_pass)
            if max_move_value < min_move_value:
                min_move_value = max_move_value
            if min_move_value <= alpha:
                return min_move_value
            beta = min(beta, min_move_value)
        return min_move_value

    def evaluate_game_state(self, game_state, player):
        opponent_player = self.get_opponent_player(player)
        player_count = 0
        player_liberty = set()
        opponent_count = 0
        opponent_liberty = set()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if game_state[row][col] == player:
                    player_count += 1
                elif game_state[row][col] == opponent_player:
                    opponent_count += 1
                else:
                    for index in range(len(X_CHANGES)):
                        new_row = row + X_CHANGES[index]
                        new_col = col + Y_CHANGES[index]
                        if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                            if game_state[new_row][new_col] == player:
                                player_liberty.add((row, col))
                            elif game_state[new_row][new_col] == opponent_player:
                                opponent_liberty.add((row, col))

        player_edge_count = 0
        opponent_player_edge_count = 0
        for col in range(BOARD_SIZE):
            if game_state[0][col] == player or game_state[BOARD_SIZE - 1][col] == player:
                player_edge_count += 1
            if game_state[0][col] == opponent_player or game_state[BOARD_SIZE - 1][col] == opponent_player:
                opponent_player_edge_count += 1

        for col in range(1, BOARD_SIZE - 1):
            if game_state[col][0] == player or game_state[col][BOARD_SIZE - 1] == player:
                player_edge_count += 1
            if game_state[col][0] == opponent_player or game_state[col][BOARD_SIZE - 1] == opponent_player:
                opponent_player_edge_count += 1
        center_unoccupied_count = 0
        for row in range(1, BOARD_SIZE - 1):
            for col in range(1, BOARD_SIZE - 1):
                if game_state[row][col] == UNOCCUPIED:
                    center_unoccupied_count += 1

        score = min(max((len(player_liberty) - len(opponent_liberty)), -8), 8) + (
                -4 * self.calculate_euler_number(game_state, player)) + (
                        5 * (player_count - opponent_count)) - (9 * player_edge_count * (center_unoccupied_count / 9))
        if self.player == WHITE:
            score += KOMI
        return score

    def move(self, game_state, player, move):
        new_game_state = copy.deepcopy(game_state)
        new_game_state[move[0]][move[1]] = player
        for index in range(len(X_CHANGES)):
            new_row = move[0] + X_CHANGES[index]
            new_col = move[1] + Y_CHANGES[index]
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                opponent_player = self.get_opponent_player(player)
                if new_game_state[new_row][new_col] == opponent_player:
                    stack = [(new_row, new_col)]
                    visited = set()
                    opponent_group_should_be_deleted = True
                    while stack:
                        top_node = stack.pop()
                        visited.add(top_node)
                        for index in range(len(X_CHANGES)):
                            new_new_row = top_node[0] + X_CHANGES[index]
                            new_new_col = top_node[1] + Y_CHANGES[index]
                            if 0 <= new_new_row < BOARD_SIZE and 0 <= new_new_col < BOARD_SIZE:
                                if (new_new_row, new_new_col) in visited:
                                    continue
                                elif new_game_state[new_new_row][new_new_col] == UNOCCUPIED:
                                    opponent_group_should_be_deleted = False
                                    break
                                elif new_game_state[new_new_row][new_new_col] == opponent_player and \
                                        (new_new_row, new_new_col) not in visited:
                                    stack.append((new_new_row, new_new_col))

                    if opponent_group_should_be_deleted:
                        for stone in visited:
                            new_game_state[stone[0]][stone[1]] = UNOCCUPIED
        return new_game_state

    def calculate_euler_number(self, game_state, player):
        opponent_player = self.get_opponent_player(player)
        new_game_state = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                new_game_state[row + 1][col + 1] = game_state[row][col]

        q1_player = 0
        qd_player = 0
        q3_player = 0
        q1_opponent_player = 0
        q2_opponent_player = 0
        q3_opponent_player = 0

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                new_game_sub_state = new_game_state[row: row + 2, col: col + 2]
                q1_player += self.count_q1(new_game_sub_state, player)
                qd_player += self.count_qd(new_game_sub_state, player)
                q3_player += self.count_q3(new_game_sub_state, player)
                q1_opponent_player += self.count_q1(new_game_sub_state, opponent_player)
                q2_opponent_player += self.count_qd(new_game_sub_state, opponent_player)
                q3_opponent_player += self.count_q3(new_game_sub_state, opponent_player)

        return (q1_player - q3_player + 2 * qd_player - (q1_opponent_player - q3_opponent_player + 2 * q2_opponent_player)) / 4

    def count_q1(self, window_state, player):
        '''
        Counts if a 2x2 window state has only one cell occupied by the player.

        Parameters:
        - window_state (2x2 list): A 2x2 matrix representing a section of the game board.
        - player (int): 1/2

        Returns:
        - 1: If only one cell in the window state is occupied by the player.
        - 0: Otherwise.
        '''
        first = window_state[0][0]
        second = window_state[0][1]
        third = window_state[1][0]
        fourth = window_state[1][1]
        if ((first == player and second != player
             and third != player and fourth != player)
                or (first != player and second == player
                    and third != player and fourth != player)
                or (first != player and second != player
                    and third == player and fourth != player)
                or (first != player and second != player
                    and third != player and fourth == player)):
            return 1
        else:
            return 0

    def count_qd(self, window_state, player):
        '''
        Counts if a 2x2 window state has a diagonal occupied by the player and the other diagonal unoccupied.

        Parameters:
        - window_state (2x2 list): A 2x2 matrix representing a section of the game board.
        - player (int): 1/2

        Returns:
        - 1: If one diagonal is occupied by the player and the other diagonal is unoccupied.
        - 0: Otherwise.
        '''
        first = window_state[0][0]
        second = window_state[0][1]
        third = window_state[1][0]
        fourth = window_state[1][1]
        if ((first == player and second != player
             and third != player and fourth == player)
                or (first != player and second == player
                    and third == player and fourth != player)):
            return 1
        else:
            return 0

    def count_q3(self, window_state, player):
        '''
        Counts if a 2x2 window state has three cells occupied by the player.

        Parameters:
        - window_state (2x2 list): A 2x2 matrix representing a section of the game board.
        - player (int): 1/2

        Returns:
        - 1: If three cells in the window state are occupied by the player.
        - 0: Otherwise.
        '''
        first = window_state[0][0]
        second = window_state[0][1]
        third = window_state[1][0]
        fourth = window_state[1][1]
        if ((first == player and second == player
             and third == player and fourth != player)
                or (first != player and second == player
                    and third == player and fourth == player)
                or (first == player and second != player
                    and third == player and fourth == player)
                or (first != player and second == player
                    and third == player and fourth== player)):
            return 1
        else:
            return 0

    def find_legal_moves(self, game_state, player):
        legal_moves = {'3player': [], '1capturing': [], '2regular': []}
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if game_state[row][col] == UNOCCUPIED:
                    if self.check_for_liberty(game_state, row, col, player):
                        if not self.ko_check(row, col):
                            if row == 0 or col == 0 or row == BOARD_SIZE - 1 or col == BOARD_SIZE - 1:
                                legal_moves.get('3player').append((row, col))
                            else:
                                legal_moves.get('2regular').append((row, col))
                    else:
                        for index in range(len(X_CHANGES)):
                            new_row = row + X_CHANGES[index]
                            new_col = col + Y_CHANGES[index]
                            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                                opponent_player = self.get_opponent_player(player)
                                if game_state[new_row][new_col] == opponent_player:
                                    new_game_state = copy.deepcopy(game_state)
                                    new_game_state[row][col] = player
                                    if not self.check_for_liberty(new_game_state, new_row, new_col,
                                                                  opponent_player):
                                        if not self.ko_check(row, col):
                                            legal_moves.get('1capturing').append((row, col))
                                        break

                
        legal_moves_list = []
        for valid_move in legal_moves.get('1capturing'):
            legal_moves_list.append(valid_move)
        for valid_move in legal_moves.get('2regular'):
            legal_moves_list.append(valid_move)
        for valid_move in legal_moves.get('3player'):
            legal_moves_list.append(valid_move)

        return legal_moves_list

    def check_for_liberty(self, game_state, i, j, player):
        stack = [(i, j)]
        visited = set()
        while stack:
            top_node = stack.pop()
            visited.add(top_node)
            for index in range(len(X_CHANGES)):
                new_row = top_node[0] + X_CHANGES[index]
                new_col = top_node[1] + Y_CHANGES[index]
                if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                    if (new_row, new_col) in visited:
                        continue
                    elif game_state[new_row][new_col] == UNOCCUPIED:
                        return True
                    elif game_state[new_row][new_col] == player and (new_row, new_col) not in visited:
                        stack.append((new_row, new_col))
        return False

    def get_opponent_player(self, player):
        return WHITE if player == BLACK else BLACK

    def ko_check(self, i, j):
        '''
        This function checks for the 'Ko' rule in the game of Go, ensuring that a move doesn't recreate a previous board position.

        Parameters:
        - row (int): The row index of the player's intended move.
        - col (int): The column index of the player's intended move.

        Returns:
        - True: If the move violates the 'Ko' rule by recreating a previous board state.
        - False: If the move doesn't violate the 'Ko' rule.
        '''
        if self.previous_board[i][j] != self.player:
            return False
        new_game_state = copy.deepcopy(self.current_board)
        new_game_state[i][j] = self.player
        opponent_i, opponent_j = self.opponent_move()
        for index in range(len(X_CHANGES)):
            new_row = i + X_CHANGES[index]
            new_col = j + Y_CHANGES[index]
            if new_row == opponent_i and new_col == opponent_j:
                if not self.check_for_liberty(new_game_state, new_row, new_col, self.opponent_player):
                    self.delete_group(new_game_state, new_row, new_col, self.opponent_player)
        return np.array_equal(new_game_state, self.previous_board)

    def opponent_move(self):
        if np.array_equal(self.current_board, self.previous_board):
            return None
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.current_board[row][col] != self.previous_board[row][col] \
                        and self.current_board[row][col] != UNOCCUPIED:
                    return row, col

    def delete_group(self, game_state, i, j, player):
        stack = [(i, j)]
        visited = set()
        while stack:
            top_node = stack.pop()
            visited.add(top_node)
            game_state[top_node[0]][top_node[1]] = UNOCCUPIED
            for index in range(len(X_CHANGES)):
                new_row = top_node[0] + X_CHANGES[index]
                new_col = top_node[1] + Y_CHANGES[index]
                if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                    if (new_row, new_col) in visited:
                        continue
                    elif game_state[new_row][new_col] == player:
                        stack.append((new_row, new_col))
        return game_state

if __name__ == '__main__':
    player, previous_board, current_board = read_input()
    step_count = calculate_step_count(previous_board, current_board)
    my_player = MyPlayer(player, previous_board, current_board)
    my_player.alphaBeta_search(4, 20, step_count)