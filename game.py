import numpy as np
import itertools
import encoder_decoder as ed
import copy


class Game:

    def __init__(self):
        self.current_board = Board(np.zeros([8, 8]).astype(str), initial=True)
        self.gameState = GameState(self.current_board, -1)
        self.actionSpace = np.zeros([4672]).astype(int)
        self.pieces = {0: "R", 1: "N", 2: "B", 3: "Q", 4: "K", 5: "P", 6: "r", 7: "n", 8: "b", 9: "q", 10: "k", 11: "p"}
        self.grid_shape = (8, 8)
        self.input_shape = (22, 8, 8)  # input for NN
        self.name = 'chess'
        self.state_size = len(self.gameState.binary)
        self.action_size = len(self.actionSpace)  # output for NN

    def reset(self):
        self.current_board = Board(np.zeros([8, 8]).astype(str), initial=True)
        self.gameState = GameState(self.current_board, -1)
        return self.gameState

    def step(self, action):
        next_state, value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        info = None
        return ((next_state, value, done, info))

    def identities(self, state, actionValues):
        identities = [(state, actionValues)]

        # currentBoard = state.board
        # currentAV = actionValues
        #
        # currentBoard = np.array([
        #     currentBoard[6], currentBoard[5], currentBoard[4], currentBoard[3], currentBoard[2], currentBoard[1],
        #     currentBoard[0]
        #     , currentBoard[13], currentBoard[12], currentBoard[11], currentBoard[10], currentBoard[9], currentBoard[8],
        #     currentBoard[7]
        #     , currentBoard[20], currentBoard[19], currentBoard[18], currentBoard[17], currentBoard[16],
        #     currentBoard[15], currentBoard[14]
        #     , currentBoard[27], currentBoard[26], currentBoard[25], currentBoard[24], currentBoard[23],
        #     currentBoard[22], currentBoard[21]
        #     , currentBoard[34], currentBoard[33], currentBoard[32], currentBoard[31], currentBoard[30],
        #     currentBoard[29], currentBoard[28]
        #     , currentBoard[41], currentBoard[40], currentBoard[39], currentBoard[38], currentBoard[37],
        #     currentBoard[36], currentBoard[35]
        # ])
        #
        # currentAV = np.array([
        #     currentAV[6], currentAV[5], currentAV[4], currentAV[3], currentAV[2], currentAV[1], currentAV[0]
        #     , currentAV[13], currentAV[12], currentAV[11], currentAV[10], currentAV[9], currentAV[8], currentAV[7]
        #     , currentAV[20], currentAV[19], currentAV[18], currentAV[17], currentAV[16], currentAV[15], currentAV[14]
        #     , currentAV[27], currentAV[26], currentAV[25], currentAV[24], currentAV[23], currentAV[22], currentAV[21]
        #     , currentAV[34], currentAV[33], currentAV[32], currentAV[31], currentAV[30], currentAV[29], currentAV[28]
        #     , currentAV[41], currentAV[40], currentAV[39], currentAV[38], currentAV[37], currentAV[36], currentAV[35]
        # ])
        #
        # identities.append((GameState(currentBoard, state.playerTurn), currentAV))

        return identities


class Board:
    def __init__(self, board, initial=False):
        self.init_board = board
        if initial:
            self.init_board[0, 0] = "r"
            self.init_board[0, 1] = "n"
            self.init_board[0, 2] = "b"
            self.init_board[0, 3] = "q"
            self.init_board[0, 4] = "k"
            self.init_board[0, 5] = "b"
            self.init_board[0, 6] = "n"
            self.init_board[0, 7] = "r"
            self.init_board[1, 0:8] = "p"
            self.init_board[7, 0] = "R"
            self.init_board[7, 1] = "N"
            self.init_board[7, 2] = "B"
            self.init_board[7, 3] = "Q"
            self.init_board[7, 4] = "K"
            self.init_board[7, 5] = "B"
            self.init_board[7, 6] = "N"
            self.init_board[7, 7] = "R"
            self.init_board[6, 0:8] = "P"
            self.init_board[self.init_board == "0.0"] = " "
        self.current_board = self.init_board
        self.en_passant = -0.1
        self.en_passant_move = 0  # returns j index of last en_passant pawn
        self.r1_move_count = 0  # black's queenside rook
        self.r2_move_count = 0  # black's kingside rook
        self.k_move_count = 0
        self.R1_move_count = 0  # white's queenside rook
        self.R2_move_count = 0  # white's kingside rook
        self.K_move_count = 0
        self.move_count = 0

    def move_rules_P(self, current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        # to calculate allowed moves for king
        threats = []
        if 0 <= i - 1 <= 7 and 0 <= j + 1 <= 7:
            threats.append((i - 1, j + 1))
        if 0 <= i - 1 <= 7 and 0 <= j - 1 <= 7:
            threats.append((i - 1, j - 1))
        # at initial position
        if i == 6:
            if board_state[i - 1, j] == " ":
                next_positions.append((i - 1, j))
                if board_state[i - 2, j] == " ":
                    next_positions.append((i - 2, j))
        # en passant capture
        elif i == 3 and self.en_passant != -0.1:
            if j - 1 == self.en_passant and abs(self.en_passant_move - self.move_count) == 1:
                next_positions.append((i - 1, j - 1))
            elif j + 1 == self.en_passant and abs(self.en_passant_move - self.move_count) == 1:
                next_positions.append((i - 1, j + 1))
        if i in [1, 2, 3, 4, 5] and board_state[i - 1, j] == " ":
            next_positions.append((i - 1, j))
        if j == 0 and board_state[i - 1, j + 1] in ["r", "n", "b", "q", "k", "p"]:
            next_positions.append((i - 1, j + 1))
        elif j == 7 and board_state[i - 1, j - 1] in ["r", "n", "b", "q", "k", "p"]:
            next_positions.append((i - 1, j - 1))
        elif j in [1, 2, 3, 4, 5, 6]:
            if board_state[i - 1, j + 1] in ["r", "n", "b", "q", "k", "p"]:
                next_positions.append((i - 1, j + 1))
            if board_state[i - 1, j - 1] in ["r", "n", "b", "q", "k", "p"]:
                next_positions.append((i - 1, j - 1))
        return next_positions, threats

    def move_rules_p(self, current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        # to calculate allowed moves for king
        threats = []
        if 0 <= i + 1 <= 7 and 0 <= j + 1 <= 7:
            threats.append((i + 1, j + 1))
        if 0 <= i + 1 <= 7 and 0 <= j - 1 <= 7:
            threats.append((i + 1, j - 1))
        # at initial position
        if i == 1:
            if board_state[i + 1, j] == " ":
                next_positions.append((i + 1, j))
                if board_state[i + 2, j] == " ":
                    next_positions.append((i + 2, j))
        # en passant capture
        elif i == 4 and self.en_passant != -0.1:
            if j - 1 == self.en_passant and abs(self.en_passant_move - self.move_count) == 1:
                next_positions.append((i + 1, j - 1))
            elif j + 1 == self.en_passant and abs(self.en_passant_move - self.move_count) == 1:
                next_positions.append((i + 1, j + 1))
        if i in [2, 3, 4, 5, 6] and board_state[i + 1, j] == " ":
            next_positions.append((i + 1, j))
        if j == 0 and board_state[i + 1, j + 1] in ["R", "N", "B", "Q", "K", "P"]:
            next_positions.append((i + 1, j + 1))
        elif j == 7 and board_state[i + 1, j - 1] in ["R", "N", "B", "Q", "K", "P"]:
            next_positions.append((i + 1, j - 1))
        elif j in [1, 2, 3, 4, 5, 6]:
            if board_state[i + 1, j + 1] in ["R", "N", "B", "Q", "K", "P"]:
                next_positions.append((i + 1, j + 1))
            if board_state[i + 1, j - 1] in ["R", "N", "B", "Q", "K", "P"]:
                next_positions.append((i + 1, j - 1))
        return next_positions, threats

    def move_rules_r(self, current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = []
        a = i
        while a != 0:
            if board_state[a - 1, j] != " ":
                if board_state[a - 1, j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a - 1, j))
                break
            next_positions.append((a - 1, j))
            a -= 1
        a = i
        while a != 7:
            if board_state[a + 1, j] != " ":
                if board_state[a + 1, j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a + 1, j))
                break
            next_positions.append((a + 1, j))
            a += 1
        a = j
        while a != 7:
            if board_state[i, a + 1] != " ":
                if board_state[i, a + 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i, a + 1))
                break
            next_positions.append((i, a + 1))
            a += 1
        a = j
        while a != 0:
            if board_state[i, a - 1] != " ":
                if board_state[i, a - 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i, a - 1))
                break
            next_positions.append((i, a - 1))
            a -= 1
        return next_positions

    def move_rules_R(self, current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = []
        a = i
        while a != 0:
            if board_state[a - 1, j] != " ":
                if board_state[a - 1, j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a - 1, j))
                break
            next_positions.append((a - 1, j))
            a -= 1
        a = i
        while a != 7:
            if board_state[a + 1, j] != " ":
                if board_state[a + 1, j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a + 1, j))
                break
            next_positions.append((a + 1, j))
            a += 1
        a = j
        while a != 7:
            if board_state[i, a + 1] != " ":
                if board_state[i, a + 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i, a + 1))
                break
            next_positions.append((i, a + 1))
            a += 1
        a = j
        while a != 0:
            if board_state[i, a - 1] != " ":
                if board_state[i, a - 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i, a - 1))
                break
            next_positions.append((i, a - 1))
            a -= 1
        return next_positions

    def move_rules_n(self, current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        for a, b in [(i + 2, j - 1), (i + 2, j + 1), (i + 1, j - 2), (i - 1, j - 2), (i - 2, j + 1), (i - 2, j - 1),
                     (i - 1, j + 2), (i + 1, j + 2)]:
            if 0 <= a <= 7 and 0 <= b <= 7:
                if board_state[a, b] in ["R", "N", "B", "Q", "K", "P", " "]:
                    next_positions.append((a, b))
        return next_positions

    def move_rules_N(self, current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        for a, b in [(i + 2, j - 1), (i + 2, j + 1), (i + 1, j - 2), (i - 1, j - 2), (i - 2, j + 1), (i - 2, j - 1),
                     (i - 1, j + 2), (i + 1, j + 2)]:
            if 0 <= a <= 7 and 0 <= b <= 7:
                if board_state[a, b] in ["r", "n", "b", "q", "k", "p", " "]:
                    next_positions.append((a, b))
        return next_positions

    def move_rules_b(self, current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        a = i
        b = j
        while a != 0 and b != 0:
            if board_state[a - 1, b - 1] != " ":
                if board_state[a - 1, b - 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a - 1, b - 1))
                break
            next_positions.append((a - 1, b - 1))
            a -= 1
            b -= 1
        a = i
        b = j
        while a != 7 and b != 7:
            if board_state[a + 1, b + 1] != " ":
                if board_state[a + 1, b + 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a + 1, b + 1))
                break
            next_positions.append((a + 1, b + 1))
            a += 1
            b += 1
        a = i
        b = j
        while a != 0 and b != 7:
            if board_state[a - 1, b + 1] != " ":
                if board_state[a - 1, b + 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a - 1, b + 1))
                break
            next_positions.append((a - 1, b + 1))
            a -= 1
            b += 1
        a = i
        b = j
        while a != 7 and b != 0:
            if board_state[a + 1, b - 1] != " ":
                if board_state[a + 1, b - 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a + 1, b - 1))
                break
            next_positions.append((a + 1, b - 1))
            a += 1
            b -= 1
        return next_positions

    def move_rules_B(self, current_position):
        i, j = current_position
        next_positions = []
        board_state = self.current_board
        a = i
        b = j
        while a != 0 and b != 0:
            if board_state[a - 1, b - 1] != " ":
                if board_state[a - 1, b - 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a - 1, b - 1))
                break
            next_positions.append((a - 1, b - 1))
            a -= 1
            b -= 1
        a = i
        b = j
        while a != 7 and b != 7:
            if board_state[a + 1, b + 1] != " ":
                if board_state[a + 1, b + 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a + 1, b + 1))
                break
            next_positions.append((a + 1, b + 1))
            a += 1
            b += 1
        a = i
        b = j
        while a != 0 and b != 7:
            if board_state[a - 1, b + 1] != " ":
                if board_state[a - 1, b + 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a - 1, b + 1))
                break
            next_positions.append((a - 1, b + 1))
            a -= 1
            b += 1
        a = i
        b = j
        while a != 7 and b != 0:
            if board_state[a + 1, b - 1] != " ":
                if board_state[a + 1, b - 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a + 1, b - 1))
                break
            next_positions.append((a + 1, b - 1))
            a += 1
            b -= 1
        return next_positions

    def move_rules_q(self, current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = []
        a = i
        # bishop-like moves
        while a != 0:
            if board_state[a - 1, j] != " ":
                if board_state[a - 1, j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a - 1, j))
                break
            next_positions.append((a - 1, j))
            a -= 1
        a = i
        while a != 7:
            if board_state[a + 1, j] != " ":
                if board_state[a + 1, j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a + 1, j))
                break
            next_positions.append((a + 1, j))
            a += 1
        a = j
        while a != 7:
            if board_state[i, a + 1] != " ":
                if board_state[i, a + 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i, a + 1))
                break
            next_positions.append((i, a + 1))
            a += 1
        a = j
        while a != 0:
            if board_state[i, a - 1] != " ":
                if board_state[i, a - 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i, a - 1))
                break
            next_positions.append((i, a - 1))
            a -= 1
        # rook-like moves
        a = i
        b = j
        while a != 0 and b != 0:
            if board_state[a - 1, b - 1] != " ":
                if board_state[a - 1, b - 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a - 1, b - 1))
                break
            next_positions.append((a - 1, b - 1))
            a -= 1
            b -= 1
        a = i
        b = j
        while a != 7 and b != 7:
            if board_state[a + 1, b + 1] != " ":
                if board_state[a + 1, b + 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a + 1, b + 1))
                break
            next_positions.append((a + 1, b + 1))
            a += 1
            b += 1
        a = i
        b = j
        while a != 0 and b != 7:
            if board_state[a - 1, b + 1] != " ":
                if board_state[a - 1, b + 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a - 1, b + 1))
                break
            next_positions.append((a - 1, b + 1))
            a -= 1
            b += 1
        a = i
        b = j
        while a != 7 and b != 0:
            if board_state[a + 1, b - 1] != " ":
                if board_state[a + 1, b - 1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a + 1, b - 1))
                break
            next_positions.append((a + 1, b - 1))
            a += 1
            b -= 1
        return next_positions

    def move_rules_Q(self, current_position):
        i, j = current_position
        board_state = self.current_board
        next_positions = []
        a = i
        # bishop moves
        while a != 0:
            if board_state[a - 1, j] != " ":
                if board_state[a - 1, j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a - 1, j))
                break
            next_positions.append((a - 1, j))
            a -= 1
        a = i
        while a != 7:
            if board_state[a + 1, j] != " ":
                if board_state[a + 1, j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a + 1, j))
                break
            next_positions.append((a + 1, j))
            a += 1
        a = j
        while a != 7:
            if board_state[i, a + 1] != " ":
                if board_state[i, a + 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i, a + 1))
                break
            next_positions.append((i, a + 1))
            a += 1
        a = j
        while a != 0:
            if board_state[i, a - 1] != " ":
                if board_state[i, a - 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i, a - 1))
                break
            next_positions.append((i, a - 1))
            a -= 1
        # rook moves
        a = i
        b = j
        while a != 0 and b != 0:
            if board_state[a - 1, b - 1] != " ":
                if board_state[a - 1, b - 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a - 1, b - 1))
                break
            next_positions.append((a - 1, b - 1))
            a -= 1
            b -= 1
        a = i
        b = j
        while a != 7 and b != 7:
            if board_state[a + 1, b + 1] != " ":
                if board_state[a + 1, b + 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a + 1, b + 1))
                break
            next_positions.append((a + 1, b + 1))
            a += 1
            b += 1
        a = i
        b = j
        while a != 0 and b != 7:
            if board_state[a - 1, b + 1] != " ":
                if board_state[a - 1, b + 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a - 1, b + 1))
                break
            next_positions.append((a - 1, b + 1))
            a -= 1
            b += 1
        a = i
        b = j
        while a != 7 and b != 0:
            if board_state[a + 1, b - 1] != " ":
                if board_state[a + 1, b - 1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a + 1, b - 1))
                break
            next_positions.append((a + 1, b - 1))
            a += 1
            b -= 1
        return next_positions

    # does not include king, castling
    def possible_W_moves(self, threats=False):
        board_state = self.current_board
        rooks = {}
        knights = {}
        bishops = {}
        queens = {}
        pawns = {}
        i, j = np.where(board_state == "R")
        for rook in zip(i, j):
            rooks[tuple(rook)] = self.move_rules_R(rook)
        i, j = np.where(board_state == "N")
        for knight in zip(i, j):
            knights[tuple(knight)] = self.move_rules_N(knight)
        i, j = np.where(board_state == "B")
        for bishop in zip(i, j):
            bishops[tuple(bishop)] = self.move_rules_B(bishop)
        i, j = np.where(board_state == "Q")
        for queen in zip(i, j):
            queens[tuple(queen)] = self.move_rules_Q(queen)
        i, j = np.where(board_state == "P")
        for pawn in zip(i, j):
            if threats == False:
                pawns[tuple(pawn)], _ = self.move_rules_P(pawn)
            else:
                _, pawns[tuple(pawn)] = self.move_rules_P(pawn)
        c_dict = {"R": rooks, "N": knights, "B": bishops, "Q": queens, "P": pawns}
        c_list = []
        c_list.extend(list(itertools.chain(*list(rooks.values()))))
        c_list.extend(list(itertools.chain(*list(knights.values()))))
        c_list.extend(list(itertools.chain(*list(bishops.values()))))
        c_list.extend(list(itertools.chain(*list(queens.values()))))
        c_list.extend(list(itertools.chain(*list(pawns.values()))))
        return c_list, c_dict

    def move_rules_k(self):
        current_position = np.where(self.current_board == "k")
        i, j = current_position
        i, j = i[0], j[0]
        next_positions = []
        c_list, _ = self.possible_W_moves(threats=True)
        for a, b in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1), (i + 1, j + 1), (i - 1, j - 1), (i + 1, j - 1),
                     (i - 1, j + 1)]:
            if 0 <= a <= 7 and 0 <= b <= 7:
                if self.current_board[a, b] in [" ", "Q", "B", "N", "P", "R"] and (a, b) not in c_list:
                    next_positions.append((a, b))
        if self.castle(1, "queenside") == True and self.king_in_check_status(1) == False:
            next_positions.append((0, 2))
        if self.castle(1, "kingside") == True and self.king_in_check_status(1) == False:
            next_positions.append((0, 6))
        return next_positions

        # does not include king, castling

    def possible_B_moves(self, threats=False):
        rooks = {}
        knights = {}
        bishops = {}
        queens = {}
        pawns = {}
        board_state = self.current_board
        i, j = np.where(board_state == "r")
        for rook in zip(i, j):
            rooks[tuple(rook)] = self.move_rules_r(rook)
        i, j = np.where(board_state == "n")
        for knight in zip(i, j):
            knights[tuple(knight)] = self.move_rules_n(knight)
        i, j = np.where(board_state == "b")
        for bishop in zip(i, j):
            bishops[tuple(bishop)] = self.move_rules_b(bishop)
        i, j = np.where(board_state == "q")
        for queen in zip(i, j):
            queens[tuple(queen)] = self.move_rules_q(queen)
        i, j = np.where(board_state == "p")
        for pawn in zip(i, j):
            if threats == False:
                pawns[tuple(pawn)], _ = self.move_rules_p(pawn)
            else:
                _, pawns[tuple(pawn)] = self.move_rules_p(pawn)
        c_dict = {"r": rooks, "n": knights, "b": bishops, "q": queens, "p": pawns}
        c_list = []
        c_list.extend(list(itertools.chain(*list(rooks.values()))))
        c_list.extend(list(itertools.chain(*list(knights.values()))))
        c_list.extend(list(itertools.chain(*list(bishops.values()))))
        c_list.extend(list(itertools.chain(*list(queens.values()))))
        c_list.extend(list(itertools.chain(*list(pawns.values()))))
        return c_list, c_dict

    def move_rules_K(self):
        current_position = np.where(self.current_board == "K")
        i, j = current_position
        i, j = i[0], j[0]
        next_positions = []
        c_list, _ = self.possible_B_moves(threats=True)
        for a, b in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1), (i + 1, j + 1), (i - 1, j - 1), (i + 1, j - 1),
                     (i - 1, j + 1)]:
            if 0 <= a <= 7 and 0 <= b <= 7:
                if self.current_board[a, b] in [" ", "q", "b", "n", "p", "r"] and (a, b) not in c_list:
                    next_positions.append((a, b))
        if self.castle(-1, "queenside") == True and self.king_in_check_status(-1) == False:
            next_positions.append((7, 2))
        if self.castle(-1, "kingside") == True and self.king_in_check_status(-1) == False:
            next_positions.append((7, 6))
        return next_positions

    # player = "w" or "b", side="queenside" or "kingside"
    def castle(self, playerTurn, side, inplace=False):
        if playerTurn == -1 and self.K_move_count == 0:
            if side == "queenside" and self.R1_move_count == 0 and self.current_board[7, 1] == " " and \
                    self.current_board[7, 2] == " " \
                    and self.current_board[7, 3] == " ":
                if inplace == True:
                    self.current_board[7, 0] = " "
                    self.current_board[7, 3] = "R"
                    self.current_board[7, 4] = " "
                    self.current_board[7, 2] = "K"
                    self.K_move_count += 1
                return True
            elif side == "kingside" and self.R2_move_count == 0 and self.current_board[7, 5] == " " and \
                    self.current_board[7, 6] == " ":
                if inplace == True:
                    self.current_board[7, 7] = " "
                    self.current_board[7, 5] = "R"
                    self.current_board[7, 4] = " "
                    self.current_board[7, 6] = "K"
                    self.K_move_count += 1
                return True
        if playerTurn == 1 and self.k_move_count == 0:
            if side == "queenside" and self.r1_move_count == 0 and self.current_board[0, 1] == " " and \
                    self.current_board[0, 2] == " " \
                    and self.current_board[0, 3] == " ":
                if inplace == True:
                    self.current_board[0, 0] = " "
                    self.current_board[0, 3] = "r"
                    self.current_board[0, 4] = " "
                    self.current_board[0, 2] = "k"
                    self.k_move_count += 1
                return True
            elif side == "kingside" and self.r2_move_count == 0 and self.current_board[0, 5] == " " and \
                    self.current_board[0, 6] == " ":
                if inplace == True:
                    self.current_board[0, 7] = " "
                    self.current_board[0, 5] = "r"
                    self.current_board[0, 4] = " "
                    self.current_board[0, 6] = "k"
                    self.k_move_count += 1
                return True
        return False

    def check_num_pieces(self):
        num_pieces = 0
        for row in self.current_board:
            for square in row:
                if square != ' ':
                    num_pieces += 1
        return num_pieces

    # check if current player's king is in check
    def king_in_check_status(self, playerTurn):
        if playerTurn == -1:
            c_list, _ = self.possible_B_moves(threats=True)
            king_position = np.where(self.current_board == "K")
            opposite_king_position = np.where(self.current_board == "k")
            i, j = king_position
            x, y = opposite_king_position
            # x, y = x[0], y[0]
            kings_encroaching = False
            if (i, j) in [(x + 1, y),
                          (x - 1, y),
                          (x, y + 1),
                          (x, y - 1),
                          (x + 1, y + 1),
                          (x - 1, y - 1),
                          (x + 1, y - 1),
                          (x - 1, y + 1)]:
                kings_encroaching = True

            if (i, j) in c_list or kings_encroaching:
                return True
        elif playerTurn == 1:
            c_list, _ = self.possible_W_moves(threats=True)
            king_position = np.where(self.current_board == "k")
            opposite_king_position = np.where(self.current_board == "K")
            i, j = king_position
            x, y = opposite_king_position
            # x, y = x[0], y[0]
            kings_encroaching = False
            if (i, j) in [(x + 1, y),
                          (x - 1, y),
                          (x, y + 1),
                          (x, y - 1),
                          (x + 1, y + 1),
                          (x - 1, y - 1),
                          (x + 1, y - 1),
                          (x - 1, y + 1)]:
                kings_encroaching = True
            if (i, j) in c_list or kings_encroaching:
                return True
        return False

    def move_piece(self, playerTurn, initial_position, final_position, promoted_piece="Q"):
        if playerTurn == -1:
            promoted = False
            i, j = initial_position
            piece = self.current_board[i, j]
            self.current_board[i, j] = " "
            i, j = final_position
            if piece == "R" and initial_position == (7, 0):
                self.R1_move_count += 1
            if piece == "R" and initial_position == (7, 7):
                self.R2_move_count += 1
            if piece == "K":
                self.K_move_count += 1
            x, y = initial_position
            if piece == "P":
                if abs(x - i) > 1:
                    self.en_passant = j
                    self.en_passant_move = self.move_count
                if abs(y - j) == 1 and self.current_board[i, j] == " ":  # En passant capture
                    self.current_board[i + 1, j] = " "
                if i == 0 and promoted_piece in ["R", "B", "N", "Q"]:
                    self.current_board[i, j] = promoted_piece
                    promoted = True
            if promoted == False:
                self.current_board[i, j] = piece
            self.move_count += 1

        elif playerTurn == 1:
            promoted = False
            i, j = initial_position
            piece = self.current_board[i, j]
            self.current_board[i, j] = " "
            i, j = final_position
            if piece == "r" and initial_position == (0, 0):
                self.r1_move_count += 1
            if piece == "r" and initial_position == (0, 7):
                self.r2_move_count += 1
            if piece == "k":
                self.k_move_count += 1
            x, y = initial_position
            if piece == "p":
                if abs(x - i) > 1:
                    self.en_passant = j
                    self.en_passant_move = self.move_count
                if abs(y - j) == 1 and self.current_board[i, j] == " ":  # En passant capture
                    self.current_board[i - 1, j] = " "
                if i == 7 and promoted_piece in ["r", "b", "n", "q"]:
                    self.current_board[i, j] = promoted_piece
                    promoted = True
            if promoted == False:
                self.current_board[i, j] = piece
            self.move_count += 1

        else:
            print("Invalid move: ", initial_position, final_position, promoted_piece)


class GameState():
    def __init__(self, board, playerTurn):
        self.board = board
        self.pieces = {0: "R", 1: "N", 2: "B", 3: "Q", 4: "K", 5: "P", 6: "r", 7: "n", 8: "b", 9: "q", 10: "k", 11: "p"}
        self.playerTurn = playerTurn
        self.states = []
        self.no_progress_count = 0
        self.draw_counter = 0
        self.move_history = None
        self.repetitions_b = 0
        self.move_count_copy = None
        self.en_passant_move_copy = None
        self.copy_board = None
        self.en_passant_copy = None
        self.r1_move_count_copy = None
        self.r2_move_count_copy = None
        self.k_move_count_copy = None
        self.R1_move_count_copy = None
        self.R2_move_count_copy = None
        self.K_move_count_copy = None
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()

    def in_check_possible_moves(self):
        # return only moves that move king out of check:
        self.copy_board = np.copy(self.board.current_board)
        self.move_count_copy = self.board.move_count  # backup board state
        self.en_passant_copy = self.board.en_passant
        self.r1_move_count_copy = self.board.r1_move_count
        self.r2_move_count_copy = self.board.r2_move_count
        self.en_passant_move_copy = self.board.en_passant_move
        self.k_move_count_copy = self.board.k_move_count
        self.R1_move_count_copy = self.board.R1_move_count
        self.R2_move_count_copy = self.board.R2_move_count
        self.K_move_count_copy = self.board.K_move_count
        # if white's turn
        if self.playerTurn == -1:
            possible_moves = []
            _, c_dict = self.board.possible_W_moves()
            current_position = np.where(self.board.current_board == "K")
            i, j = current_position
            i, j = i[0], j[0]
            c_dict["K"] = {(i, j): self.board.move_rules_K()}
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        self.board.move_piece(self.playerTurn, initial_pos, final_pos)
                        if self.board.king_in_check_status(-1) == False:
                            if key in ["P", "p"] and final_pos[0] in [0, 7]:
                                for p in ["queen", "rook", "knight", "bishop"]:
                                    possible_moves.append([initial_pos, final_pos, p])
                            else:
                                possible_moves.append([initial_pos, final_pos, None])
                        self.board.current_board = np.copy(self.copy_board)
                        self.board.en_passant = self.en_passant_copy
                        self.board.en_passant_move = self.en_passant_move_copy
                        self.board.R1_move_count = self.R1_move_count_copy
                        self.board.R2_move_count = self.R2_move_count_copy
                        self.board.K_move_count = self.K_move_count_copy
                        self.board.move_count = self.move_count_copy
            return possible_moves
        if self.playerTurn == 1:
            possible_moves = []
            _, c_dict = self.board.possible_B_moves()
            current_position = np.where(self.board.current_board == "k")
            i, j = current_position
            i, j = i[0], j[0]
            c_dict["k"] = {(i, j): self.board.move_rules_k()}
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        self.board.move_piece(self.playerTurn, initial_pos, final_pos)
                        if self.board.king_in_check_status(1) == False:
                            if key in ["P", "p"] and final_pos[0] in [0, 7]:
                                for p in ["queen", "rook", "knight", "bishop"]:
                                    possible_moves.append([initial_pos, final_pos, p])
                            else:
                                possible_moves.append([initial_pos, final_pos, None])
                        self.board.current_board = np.copy(self.copy_board)
                        self.board.en_passant = self.en_passant_copy
                        self.board.en_passant_move = self.en_passant_move_copy
                        self.board.r1_move_count = self.r1_move_count_copy
                        self.board.r2_move_count = self.r2_move_count_copy
                        self.board.k_move_count = self.k_move_count_copy
                        self.board.move_count = self.move_count_copy
            return possible_moves

    # functions used elsewhere:
    def _allowedActions(self):
        allowed = []

        if self.board.king_in_check_status(self.playerTurn):
            allowed = self.in_check_possible_moves()
        else:
            acts = []
            if self.playerTurn == -1:
                _, c_dict = self.board.possible_W_moves()  # all non-king moves except castling
                current_position = np.where(self.board.current_board == "K")
                i, j = current_position
                i, j = i[0], j[0]
                c_dict["K"] = {(i, j): self.board.move_rules_K()}  # all king moves
                for key in c_dict.keys():
                    for initial_pos in c_dict[key].keys():
                        for final_pos in c_dict[key][initial_pos]:
                            if key in ["P", "p"] and final_pos[0] in [0, 7]:
                                for p in ["queen", "rook", "knight", "bishop"]:
                                    acts.append([initial_pos, final_pos, p])
                            else:
                                acts.append([initial_pos, final_pos, None])
                actss = []
                for act in acts:  # after move, check that its not check ownself, else illegal move
                    i, f, p = act
                    self.copy_board = np.copy(self.board.current_board)
                    self.move_count_copy = self.board.move_count  # backup board state
                    self.en_passant_copy = self.board.en_passant
                    self.r1_move_count_copy = self.board.r1_move_count
                    self.r2_move_count_copy = self.board.r2_move_count
                    self.en_passant_move_copy = self.board.en_passant_move
                    self.k_move_count_copy = self.board.k_move_count
                    self.R1_move_count_copy = self.board.R1_move_count
                    self.R2_move_count_copy = self.board.R2_move_count
                    self.K_move_count_copy = self.board.K_move_count

                    self.board.move_piece(self.playerTurn, i, f, p)
                    if self.board.king_in_check_status(self.playerTurn) == False:
                        actss.append(act)
                    self.board.current_board = np.copy(self.copy_board)
                    self.board.en_passant = self.en_passant_copy
                    self.board.en_passant_move = self.en_passant_move_copy
                    self.board.R1_move_count = self.R1_move_count_copy
                    self.board.R2_move_count = self.R2_move_count_copy
                    self.board.K_move_count = self.K_move_count_copy
                    self.board.move_count = self.move_count_copy
                allowed = actss
            if self.playerTurn == 1:
                _, c_dict = self.board.possible_B_moves()  # all non-king moves except castling
                current_position = np.where(self.board.current_board == "k")
                i, j = current_position
                i, j = i[0], j[0]
                c_dict["k"] = {(i, j): self.board.move_rules_k()}  # all king moves
                for key in c_dict.keys():
                    for initial_pos in c_dict[key].keys():
                        for final_pos in c_dict[key][initial_pos]:
                            if key in ["P", "p"] and final_pos[0] in [0, 7]:
                                for p in ["queen", "rook", "knight", "bishop"]:
                                    acts.append([initial_pos, final_pos, p])
                            else:
                                acts.append([initial_pos, final_pos, None])
                actss = []
                for act in acts:  # after move, check that its not check ownself, else illegal move
                    i, f, p = act
                    self.copy_board = np.copy(self.board.current_board)
                    self.move_count_copy = self.board.move_count  # backup board state
                    self.en_passant_copy = self.board.en_passant
                    self.r1_move_count_copy = self.board.r1_move_count
                    self.r2_move_count_copy = self.board.r2_move_count
                    self.en_passant_move_copy = self.board.en_passant_move
                    self.k_move_count_copy = self.board.k_move_count
                    self.R1_move_count_copy = self.board.R1_move_count
                    self.R2_move_count_copy = self.board.R2_move_count
                    self.K_move_count_copy = self.board.K_move_count

                    self.board.move_piece(self.playerTurn, i, f, p)
                    if self.board.king_in_check_status(1) == False:
                        actss.append(act)
                    self.board.current_board = np.copy(self.copy_board)
                    self.board.en_passant = self.en_passant_copy
                    self.board.en_passant_move = self.en_passant_move_copy
                    self.board.R1_move_count = self.R1_move_count_copy
                    self.board.R2_move_count = self.R2_move_count_copy
                    self.board.K_move_count = self.K_move_count_copy
                    self.board.move_count = self.move_count_copy
                allowed = actss

        encoded_allowed = []
        for action in allowed:
            encoded_allowed.append(ed.encode_action(self, action[0], action[1], underpromote=action[2]))
        return encoded_allowed

    def decodeAllowedMoves(self):
        def changeCoords(coords):
            yCordDict = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
            xCordDict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
            return xCordDict[coords[1]] + str(yCordDict[coords[0]])
        for action in self.allowedActions:
            decoded = ed.decode_action(self.board, self.playerTurn, action)
            i, j = decoded[0][0]
            print(self.board.current_board[i][j] + ' to: ' + changeCoords(decoded[1][0]) + ' as: ' + str(action))

    def decodeMove(self, action):
        def changeCoords(coords):
            yCordDict = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
            xCordDict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
            return xCordDict[coords[1]] + str(yCordDict[coords[0]])

        decoded = ed.decode_action(self.board, self.playerTurn, action)
        i, j = decoded[0][0]
        return self.board.current_board[i][j] + ' to ' + changeCoords(decoded[1][0])

    def _binary(self):
        encoded_s = ed.encode_board(self)
        encoded_s = encoded_s.transpose(2, 0, 1)
        return encoded_s

    def _convertStateToId(self):
        pieces = {1: "R", 2: "N", 3: "B", 4: "Q", "ef": "K", 5: "P", 6: "r", 7: "n", 8: "b", "ab": "q", "cd": "k", "9": "p", 0: " "}
        flat_board = list(np.concatenate(self.board.current_board).flat)
        rev_subs = {v: k for k, v in pieces.items()}
        position = [rev_subs.get(item, item) for item in flat_board]
        id = ''.join(map(str, position)) + str(self.draw_counter) + str(self.board.R1_move_count) \
             + str(self.board.R2_move_count) + str(self.board.r1_move_count) + str(self.board.r2_move_count) \
             + str(self.board.K_move_count) + str(self.board.k_move_count) + str(self.playerTurn)

        return id

    def _checkForEndGame(self):
        # check for draw by 3 repetitions:
        for s in self.states:
            if np.array_equal(self.board.current_board, s):
                self.draw_counter += 1
        if self.draw_counter >= 3:
            return 1

        # check for draw by insufficient material:
        # pseudo: if both side have only a k, or a k and b, or a k and n
        if self.board.check_num_pieces() < 6:
            num_mating_pieces = 0
            white_bishops = 0
            black_bishops = 0
            white_knights = 0
            black_knights = 0
            for row in self.board.current_board:
                for square in row:
                    # check for pieces that could mate on own
                    if square in ['Q', 'q', 'R', 'r', 'P', 'p']:
                        num_mating_pieces += 1
                    # check for combinations of knight and bishop:
                    if square == 'B':
                        white_bishops += 1
                    if square == 'b':
                        black_bishops += 1
                    if square == 'N':
                        white_knights += 1
                    if square == 'n':
                        black_knights += 1
            if num_mating_pieces == 0 and \
                    (white_bishops + white_knights < 2) and \
                    (black_bishops + black_knights < 2):
                return 1

        # check for draw by stalemate or checkmate:
        if self.allowedActions == []:
            return 1

        # check for draw by 50 moves without pawn move or capture:
        if self.no_progress_count >= 50:
            return 1

        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        if self.board.king_in_check_status(self.playerTurn) and self.in_check_possible_moves() == []:
            return (-1, -1, 1)

        return (0, 0, 0)

    def _getScore(self):
        tmp = self.value
        return (tmp[1], tmp[2])

    def takeAction(self, action):
        self.move_count_copy = self.board.move_count  # backup board state
        self.en_passant_copy = self.board.en_passant
        self.r1_move_count_copy = self.board.r1_move_count
        self.r2_move_count_copy = self.board.r2_move_count
        self.en_passant_move_copy = self.board.en_passant_move
        self.k_move_count_copy = self.board.k_move_count
        self.R1_move_count_copy = self.board.R1_move_count
        self.R2_move_count_copy = self.board.R2_move_count
        self.K_move_count_copy = self.board.K_move_count
        no_progress = self.no_progress_count
        if self.states == []:
            self.states = [self.board.current_board]
        states_copy = self.states
        newBoard = Board(np.copy(self.board.current_board))
        playerTurn = self.playerTurn
        oldState = self

        i_pos, f_pos, prom = ed.decode_action(newBoard, playerTurn, action)
        newBoard.en_passant = oldState.en_passant_copy
        newBoard.en_passant_move = oldState.en_passant_move_copy
        newBoard.R1_move_count = oldState.R1_move_count_copy
        newBoard.R2_move_count = oldState.R2_move_count_copy
        newBoard.K_move_count = oldState.K_move_count_copy
        newBoard.r1_move_count = oldState.r1_move_count_copy
        newBoard.r2_move_count = oldState.r2_move_count_copy
        newBoard.k_move_count = oldState.k_move_count_copy
        newBoard.move_count = oldState.move_count_copy

        for i, f, p in zip(i_pos, f_pos, prom):
            newBoard.move_piece(playerTurn, i, f, p)  # move piece to get next board state s
            a, b = i
            c, d = f
            if newBoard.current_board[c, d] in ["K", "k"] and abs(
                    d - b) == 2:  # if king moves 2 squares, then move rook too for castling
                if a == 7 and d - b > 0:  # castle kingside for white
                    self.playerTurn = -1
                    newBoard.move_piece(self.playerTurn, (7, 7), (7, 5), None)
                if a == 7 and d - b < 0:  # castle queenside for white
                    self.playerTurn = -1
                    newBoard.move_piece(self.playerTurn, (7, 0), (7, 3), None)
                if a == 0 and d - b > 0:  # castle kingside for black
                    self.playerTurn = 1
                    newBoard.move_piece(self.playerTurn, (0, 7), (0, 5), None)
                if a == 0 and d - b < 0:  # castle queenside for black
                    self.playerTurn = 1
                    newBoard.move_piece(self.playerTurn, (0, 0), (0, 3), None)

        value = 0
        done = 0

        newState = GameState(newBoard, -playerTurn)
        newState.states = np.concatenate((states_copy, [newState.board.current_board]), axis=0)

        newState.no_progress_count = no_progress

        if (newState.board.check_num_pieces() != oldState.board.check_num_pieces()) or \
                newState.board.current_board[i_pos[0][0], i_pos[0][1]] in ["P", "p"]:
            newState.no_progress_count = 0
        else:
            newState.no_progress_count += 1

        newState.isEndGame = newState._checkForEndGame()

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return newState, value, done

    def render(self, logger):
        for r in range(8):
            logger.info(self.board.current_board[r])
        logger.info('Endgame: ' + str(self.isEndGame))
        logger.info('--------------')
