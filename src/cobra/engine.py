import chess
import tensorflow as tf
import time
import numpy as np
import chess.engine

from cobra import helpers
from cobra.controller import Controller
from cobra.transposition import TranspositionTable, TranspositionTableEntry, EXACT, UPPER, LOWER


class CobraEngine:
    __slots__ = ('model', 'controller', 'transposition', 'history', 'butterfly')
    def __init__(self):
        # Load neural network model to predict evaluations
        self.model = tf.keras.models.load_model('C:/Source Code/Code/chess_nn/src/nn/chess_nn_model.h5')

        # Controller to make and unmake moves while also updating the zobrist key
        self.controller = Controller()

        # Transposition table
        self.transposition = TranspositionTable()

        # Relative history heuristic
        self.history = [[[0] * 64 for _ in range(64)] for _ in range(2)]
        self.butterfly = [[[0] * 64 for _ in range(64)] for _ in range(2)]

    def get_move(self, board):
        """Return the best move given a chess board"""
        self.controller.set_board(board)
        return self._IDS(board)

    def _IDS(self, board, depth_limit=4, time_limit=5):
        """
        Iterative deepening search algorithm to find 
        best chess move for specified colour within depth limit and time limit
        """
        start_time = time.time()

        for depth in range(1, depth_limit + 1):
            evaluation, best_move = self._negamax(board, float('-inf'), float('inf'), depth)
            if time.time() - start_time > time_limit:
                break
        
        print('Evaluation:', evaluation)
        print('Depth searched:', depth)
        print('Time taken:', time.time() - start_time)
        
        return best_move

    def _negamax(self, board, alpha, beta, depth):
        alpha_orig = alpha

        # See if same position has been reached before in transposition table
        entry = self.transposition.lookup(self.controller.zobrist.key)
        if entry is not None and entry.depth >= depth:
            if entry.flag == EXACT:
                return entry.score, entry.move
            elif entry.flag == LOWER:
                alpha = max(alpha, entry.score)
            elif entry.flag == UPPER:
                beta = min(beta, entry.score)

            if alpha >= beta:
                return entry.score, entry.move

        if depth == 0 or board.is_game_over():
            return self.nn_evaluation(board) - depth, None

        best_move = None
        best_score = float('-inf')

        def move_score(move):
            if (capture_square := helpers.captured_piece_square(board, move)) is not None:
                capture = board.piece_at(capture_square)
                piece_scores = [1, 3, 3, 5, 9, 10000]  # Pawn, Knight, Bishop, Rook, Queen, King
                attacker = board.piece_at(move.from_square)
                exchange = piece_scores[capture.piece_type-1] - piece_scores[attacker.piece_type-1]

                return 30 if exchange < 0 else (10+exchange) * 5

            # Quiet move, use relative history heuristic
            hh = self.history[board.turn][move.from_square][move.to_square]
            bf = self.butterfly[board.turn][move.from_square][move.to_square]
            return 0 if bf == 0 else hh / bf

        moves = list(board.legal_moves)
        moves.sort(key=move_score, reverse=True)

        for move in moves:
            self.controller.move(move)
            score = -self._negamax(board, -beta, -alpha, depth-1)[0]
            self.controller.unmove()

            if score > best_score:
                best_score = score
                best_move = move

            is_capture = board.is_capture(move)
            alpha = max(alpha, best_score)
            
            if alpha >= beta:
                if not is_capture:
                    self.history[board.turn][move.from_square][move.to_square] += depth * depth
                break
            else:
                if not is_capture:
                    self.butterfly[board.turn][move.from_square][move.to_square] += 1

        # Store result in transposition table
        if score <= alpha_orig:
            flag = UPPER
        elif score >= beta:
            flag = LOWER
        else:
            flag = EXACT

        entry = TranspositionTableEntry(flag, depth, best_move, best_score)
        self.transposition.store(self.controller.zobrist.key, entry)

        return best_score, best_move

    def nn_evaluation(self, board):
        """Return the evaluation of a chess position"""
        if (outcome := board.outcome()) is not None:
            if outcome.winner is None:
                return 0
            elif board.turn == outcome.winner:
                return 100000
            else: 
                return -100000

        return self.model(np.array([helpers.bitboard(board)]))[0][0]
    
    def static_evaluation(self, board):
        if (outcome := board.outcome()) is not None:
            if outcome.winner is None:
                return 0
            elif board.turn == outcome.winner:
                return 100000
            else: 
                return -100000
        
        piece_scores = [1, 3, 3, 5, 9, 10000]
        white_score = 0
        black_score = 0
        for piece in chess.PIECE_TYPES:
            for _ in board.pieces(piece, chess.WHITE):
                white_score += piece_scores[piece-1]
            for _ in board.pieces(piece, chess.BLACK):
                black_score += piece_scores[piece-1]
        if board.turn == chess.WHITE:
            return white_score - black_score
        else:
            return black_score - white_score


# board = chess.Board()
# bot = CobraEngine()

# import cProfile, pstats
# profiler = cProfile.Profile()
# profiler.enable()

# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats()

# bot.get_move(board)
# while not board.is_game_over():
#     if board.turn == chess.BLACK:
#         move = bot.get_move(board)
#         board.push(move)
#         print(board, move)
#     else:
#         while True:
#             try:
#                 move = chess.Move.from_uci(input("Move: "))
#                 board.push(move)
#                 break
#             except Exception as e:
#                 print(e)
#         # move = bot.get_move(board)
#         # board.push(move)
#         # print(board, move)