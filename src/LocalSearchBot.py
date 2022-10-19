from copy import deepcopy
from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import random
import numpy as np


class LocalSearchBot(Bot):

    def getPlayerValue(self, state: GameState):
        if state.player1_turn:
            return -1
        else:
            return 1

    def get_next_position_with_zero_value(self, matrix: np.ndarray):
        [ny, nx] = matrix.shape
        x = -1
        y = -1
        valid = False
        while not valid:
            x += 1
            y = 0
            valid = matrix[y, x] == 0
        return (x, y)

    # the objective value is the sum of all the squares completed by the agent
    def get_objective_value(self, state: GameState) -> int:
        total = 0
        for i in state.board_status:
            for j in i:
                if abs(j) == 4:
                    total += j//4
        return total

    # hill climbing with sideways move
    def generate_successor(self, state: GameState):
        return {**self.generate_successor_row(state), **self.generate_successor_col(state)}

    # all possible row coordinate that can be filled
    def generate_successor_row(self, state: GameState):
        [ny, nx] = state.row_status.shape
        row_and_value = {}
        for i in range(ny):
            for j in range(nx):
                if state.row_status[i, j] == 0:
                    new_game_state = deepcopy(state)
                    new_game_state.row_status[i, j] = 1
                    # not top edge
                    if i > 0:
                        new_game_state.board_status[i - 1, j] = self.getPlayerValue(
                            state)*abs(new_game_state.board_status[i - 1, j]) + self.getPlayerValue(state)
                    # not bottom edge
                    if i < ny - 1:
                        new_game_state.board_status[i, j] = self.getPlayerValue(
                            state)*abs(new_game_state.board_status[i, j]) + self.getPlayerValue(state)
                    row_and_value["row", (j, i)] = self.get_objective_value(
                        new_game_state)
        return row_and_value

    # all possible col coordinate that can be filled
    def generate_successor_col(self, state: GameState):
        [ny, nx] = state.col_status.shape
        col_and_value = {}
        for i in range(ny):
            for j in range(nx):
                if state.col_status[i, j] == 0:
                    new_game_state = deepcopy(state)
                    new_game_state.col_status[i, j] = 1
                    # not left edge
                    if j > 0:
                        new_game_state.board_status[i, j - 1] = self.getPlayerValue(
                            state)*abs(new_game_state.board_status[i, j - 1]) + self.getPlayerValue(state)
                    # not right edge
                    if j < nx - 1:
                        new_game_state.board_status[i, j] = self.getPlayerValue(
                            state)*abs(new_game_state.board_status[i, j]) + self.getPlayerValue(state)
                    col_and_value["col", (j, i)] = self.get_objective_value(
                        new_game_state)
        return col_and_value

    # heuristic not to choose action that will give enemy chance to close the square
    def evaluate_keys(self, list_keys: list, state: GameState, maximize = True):
        # list of heuristic applied choices
        available_choices = []
        for choice in list_keys:
            state_copy : GameState
            state_copy = deepcopy(state)

            movement_type = choice[0]
            x, y = choice[1]
            i = y ; j = x

            if movement_type == 'row':
                ny = state.row_status.shape[0]
                # not top edge
                if i > 0:
                    state_copy.board_status[i - 1, j] = self.getPlayerValue(
                        state)*abs(state_copy.board_status[i - 1, j]) + self.getPlayerValue(state)
                # not bottom edge
                if i < ny - 1:
                    state_copy.board_status[i, j] = self.getPlayerValue(
                        state)*abs(state_copy.board_status[i, j]) + self.getPlayerValue(state)
                    
            elif movement_type == 'col':
                nx = state.col_status.shape[1]
                # not left edge
                if j > 0:
                    state_copy.board_status[i, j - 1] = self.getPlayerValue(
                        state)*abs(state_copy.board_status[i, j - 1]) + self.getPlayerValue(state)
                # not right edge
                if j < nx - 1:
                    state_copy.board_status[i, j] = self.getPlayerValue(
                        state)*abs(state_copy.board_status[i, j]) + self.getPlayerValue(state)

            else:
                raise Exception("Unknown movement type")
            
            # Do not choose if it makes the board status element is 3 (maximize) or -3 (minimize) 
            # because it will give enemy chance to close the square
            chosen = True
            sy, sx = state_copy.board_status.shape
            for i in range(sy):
                for j in range(sx):
                    if maximize and state_copy.board_status[i,j] == 3:
                        chosen = False
                    elif not maximize and state_copy.board_status[i,j] == -3:
                        chosen = False

            # add to available_choices if chosen
            if chosen:
                available_choices.append(choice)

        if len(available_choices) != 0:
            # randomize when available choices are more than one
            return random.choice(available_choices)
        else:
            # choose initial list_keys random if available_choices are not available
            return random.choice(list_keys)

    # get the best successor which has the highest objective value
    def get_neighbour(self, state: GameState):
        successors = self.generate_successor(state)
        # if bot is player 2, then it is maximizing
        if (not state.player1_turn):
            max_value = max(successors.values())
            max_keys = [k for k, v in successors.items() if v == max_value]
            
            # evaluate max_keys with heuristic
            neighbor = self.evaluate_keys(max_keys, state, maximize=True)

            return neighbor

        # if bot is player 1, then it is minimizing
        else:
            min_value = min(successors.values())
            min_keys = [k for k, v in successors.items() if v == min_value]

            # evaluate min_keys with heuristic
            neighbor = self.evaluate_keys(min_keys, state, maximize=False)

            return neighbor

    def get_action(self, state: GameState) -> GameAction:
        lit, pos = self.get_neighbour(state)
        return GameAction(lit, pos)
