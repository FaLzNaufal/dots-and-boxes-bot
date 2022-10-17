from copy import deepcopy
from Bot import Bot
from GameAction import GameAction
from GameState import GameState
import random
import numpy as np

class MinimaxBot(Bot):

    def getPlayerValue(self, state: GameState):
        if state.player1_turn:
            return -1
        else:
            return 1

    # the objective value is the sum of all the squares completed by the agent
    def get_objective_value(self, state: GameState) -> int:
        total = 0
        for i in state.board_status:
            for j in i:
                if abs(j) == 4:
                    total += j//4
        return total

    #check if current position is final position
    def finalPos(self, state: GameState):
        column = np.all(np.ravel(state.col_status) == 1)
        row = np.all(np.ravel(state.row_status) == 1)
        return column & row

    # generate successor for a certain state
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
                    row_and_value["row", (j, i)] = (new_game_state, 0)
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
                    col_and_value["col", (j, i)] = (new_game_state, 0)
        return col_and_value

    #check if get box from line input
    def checkConsecutiveTurn(self, iState: GameState, fState: GameState):
        iBoard = np.ravel(iState.board_status)
        fBoard = np.ravel(fState.board_status)
        i = 0
        consTurn = False
        while i < 9 and not consTurn:
            if iBoard[i] < fBoard[i] and fBoard[i] == abs(4):
                consTurn = True
            i += 1
        return consTurn

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
                    
            else:
                nx = state.col_status.shape[1]
                # not left edge
                if j > 0:
                    state_copy.board_status[i, j - 1] = self.getPlayerValue(
                        state)*abs(state_copy.board_status[i, j - 1]) + self.getPlayerValue(state)
                # not right edge
                if j < nx - 1:
                    state_copy.board_status[i, j] = self.getPlayerValue(
                        state)*abs(state_copy.board_status[i, j]) + self.getPlayerValue(state)

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

    #minimax algorithm
    def minimax (self, state: GameState, depth, alpha, beta, player1):
        if depth == 0 or self.finalPos(state):
            return self.get_objective_value(state)

        if not player1:
            path = {}
            maxValue = -100
            successor = self.generate_successor(state)
            for key in successor:
                if self.checkConsecutiveTurn(state, successor.get(key)[0]):
                    player1Turn = False
                else:
                    player1Turn = True
                value = self.minimax(successor.get(key)[0], depth-1, alpha, beta, player1Turn)
                maxValue = max(maxValue, value)
                alpha = max(alpha, value)
                if depth == 5:
                    path[key[0],key[1]] = value
                if beta <= alpha:
                    break
            if depth == 5:
                return path
            else:
                return maxValue

        else:
            path = {}
            minValue = 100
            successor = self.generate_successor(state)
            for key in successor:
                if self.checkConsecutiveTurn(state, successor.get(key)[0]):
                    player1Turn = True
                else:
                    player1Turn = False
                value = self.minimax(successor.get(key)[0], depth-1, alpha, beta, player1Turn)
                minValue = min(minValue, value)
                beta = min(beta, value)
                if depth == 5:
                    path[key[0],key[1]] = value
                if beta <= alpha:
                    break
            if depth == 5:
                optimal = min(path.values())
                optimalAct = [k for k, v in path.items() if v == optimal]
                act = random.choice(optimalAct)
                return act
            else:
                return minValue

    def get_neighbor(self, state: GameState):
        successors = self.minimax(state, 5, -100, 100, state.player1_turn)
        if state.player1_turn:
            value = min(successors.values())
            min_keys = [k for k, v in successors.items() if v == value]
            act = self.evaluate_keys(min_keys, state, maximize=False)
        else:
            value = max(successors.values())
            max_keys = [k for k, v in successors.items() if v == value]
            act = self.evaluate_keys(max_keys, state, maximize=True)
        return act

    def get_action(self, state: GameState) -> GameAction:
        lit, pos = self.get_neighbor(state)
        return GameAction(lit, pos)