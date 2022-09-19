import chess
import tensorflow as tf
import time
import numpy as np
import chess.engine

from cobra import helpers
from cobra.controller import Controller
from cobra.transposition import TranspositionTable, TranspositionTableEntry, EXACT, UPPER, LOWER


class CobraEngine:
    __slots__ = ('model', 'controller', 'transposition', 'history', 'butterfly', 'killer', 'positions_evaluated')
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

        # Killer heuristic
        self.killer = [[None] * 20 for _ in range(2)]

    def get_move(self, board):
        """Return the best move given a chess board"""
        self.controller.set_board(board)
        self.positions_evaluated = 0
        return self._IDS(board)

    def _IDS(self, board, depth_limit=10, time_limit=5):
        """
        Iterative deepening search algorithm to find 
        best chess move for specified colour within depth limit and time limit
        """
        start_time = time.time()

        for depth in range(1, depth_limit + 1):
            evaluation, best_move = self._negamax(board, float('-inf'), float('inf'), depth, True)
            print('Depth searched:', depth, end=', ')
            print('Best move:', best_move, end=', ')
            print('Evaluation', evaluation, end=', ')
            print('Time taken:', time.time() - start_time, end=', ')
            print('Positions evaluated:', self.positions_evaluated)
            if time.time() - start_time > time_limit:
                break
        
        print('\n')
        
        # print('Best move:', best_move)
        # print('Evaluation:', evaluation)
        # print('Depth searched:', depth)
        # print('Time taken:', time.time() - start_time)
        # print('Positions evaluated:', self.positions_evaluated)
        
        return best_move

    def _quiescence(self, depth, board):
        if depth <= 0 or board.is_game_over():
            return

    def _negamax(self, board, alpha, beta, depth, do_null):
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

        if depth <= 0 or board.is_game_over():
            return self.nn_evaluation(board) - depth, None

        # Null move pruning
        if do_null and not board.is_check():
            self.controller.make_null_move()
            R = 2
            score = -self._negamax(board, -beta, -beta+1, depth-R, False)[0]
            self.controller.unmake_null_move()
            
            if score >= beta:
                return score, None

        best_move = None
        best_score = float('-inf')

        def move_score(move):
            # Pv node
            if entry is not None and entry.flag == EXACT and entry.move == move:
                return 10000

            # Captures
            if (capture_square := helpers.captured_piece_square(board, move)) is not None:
                piece_scores = [1, 3, 3, 5, 9, 10000]  # Pawn, Knight, Bishop, Rook, Queen, King
                capture = board.piece_at(capture_square)
                attacker = board.piece_at(move.from_square)
                exchange = piece_scores[capture.piece_type-1] - piece_scores[attacker.piece_type-1]

                # Losing captures are after killers and winning captures are first
                return 100 if exchange < 0 else (1000+exchange) * 5

            # Killer moves
            if move == self.killer[0][depth]:
                return 500
            elif move == self.killer[1][depth]:
                return 400
                
            # Quiet move, use relative history heuristic
            hh = self.history[board.turn][move.from_square][move.to_square]
            bf = self.butterfly[board.turn][move.from_square][move.to_square]
            return 0 if bf == 0 else hh / bf

        moves = list(board.legal_moves)
        moves.sort(key=move_score, reverse=True)

        for move in moves:
            self.controller.move(move)
            score = -self._negamax(board, -beta, -alpha, depth-1, True)[0]
            self.controller.unmove()

            if score > best_score:
                best_score = score
                best_move = move

            is_capture = board.is_capture(move)
            alpha = max(alpha, best_score)
            
            if alpha >= beta:
                if not is_capture:
                    self.history[board.turn][move.from_square][move.to_square] += depth * depth
                    if self.killer[0][depth] != move:
                        self.killer[1][depth] = self.killer[0][depth]
                        self.killer[0][depth] = move
                break
            else:
                if not is_capture:
                    self.butterfly[board.turn][move.from_square][move.to_square] += depth

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
        """Predict evaluation of a chess position with a neural network"""
        self.positions_evaluated += 1
        if (outcome := board.outcome()) is not None:
            if outcome.winner is None:
                return 0
            elif board.turn == outcome.winner:
                return 100000
            else: 
                return -100000

        return self.model(np.array([helpers.bitboard(board)]))[0][0]
    
    def static_evaluation(self, board):
        """Return the evaluation in terms of material"""
        self.positions_evaluated += 1
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