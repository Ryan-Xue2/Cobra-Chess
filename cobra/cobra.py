import chess
import tensorflow as tf
import time
import numpy as np
import chess.engine

from random import randint
from transposition import TranspositionTable, TranspositionTableEntry, EXACT, UPPER, LOWER


class Cobra:
    __slots__ = ('model', 'zobrist_key', 'zobrist_active_colour', 'zobrist_castling', 'zobrist_enpassant', 'zobrist_pieces', 'transposition', 'history', 'butterfly')
    def __init__(self):
        # Load neural network model to predict evaluations
        self.model = tf.keras.models.load_model('chess_nn_model.h5')

        # Initialize zobrist key and zobrist tables
        self.zobrist_key = 0
        self.zobrist_active_colour = randint(0, 2**64)
        self.zobrist_castling = [randint(0, 2**64) for _ in range(4)]
        self.zobrist_enpassant = [randint(0, 2**64) for _ in range(8)]
        self.zobrist_pieces = [[[randint(0, 2**64) for _ in range(12)] for _ in range(8)] for _ in range(8)]

        # Transposition table
        self.transposition = TranspositionTable()

        # Relative history heuristic
        self.history = np.zeros((2, 64, 64))
        self.butterfly = np.zeros((2, 64, 64))

    def get_move(self, board):
        """Return the best move given a chess board"""
        self._calculate_zobrist_key(board)
        return self._IDS(board)

    def _IDS(self, board, depth_limit=4, time_limit=10):
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
        entry = self.transposition.lookup(self.zobrist_key)
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
            return self.evaluate_position(board) - depth, None
        
        best_move = None
        best_score = float('-inf')

        def move_score(move):
            if (capture := board.piece_at(move.to_square)) is not None:
                piece_scores = [1, 3, 3, 5, 9, 10000]  # Pawn, Knight, Bishop, Rook, Queen, King
                attacker = board.piece_at(move.from_square)
                exchange = piece_scores[capture.piece_type-1] - piece_scores[attacker.piece_type-1]

                return 30 if exchange < 0 else (10+exchange) * 5

            # Quiet move, use relative history heuristic
            hh = self.history[int(board.turn), move.from_square, move.to_square]
            bf = self.butterfly[int(board.turn), move.from_square, move.to_square]
            return 0 if bf == 0 else hh / bf

        moves = list(board.legal_moves)
        moves.sort(key=move_score, reverse=True)

        for move in moves:
            capture = board.piece_at(move.to_square)
            self._make_move(board, move)
            score = -self._negamax(board, -beta, -alpha, depth-1)[0]
            self._unmake_move(board, move, capture)

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)
            if alpha >= beta:
                if capture is None:
                    self.history[int(board.turn), move.from_square, move.to_square] += depth * depth
                break
            else:
                if capture is None:
                    self.butterfly[int(board.turn), move.from_square, move.to_square] += 1
        
        # Store result in transposition table
        if score <= alpha_orig:
            flag = UPPER
        elif score >= beta:
            flag = LOWER
        else:
            flag = EXACT

        entry = TranspositionTableEntry(flag, depth, best_move, best_score)
        self.transposition.store(self.zobrist_key, entry)

        return best_score, best_move

    def _make_move(self, board, move):
        """Make the move and update the zobrist hash accordingly"""
        y_to, x_to = divmod(move.to_square, 8)
        y_from, x_from = divmod(move.from_square, 8)

        # Piece was captured
        if (capture := board.piece_at(move.to_square)) is not None:
            self.zobrist_key ^= self.zobrist_pieces[y_to][x_to][self._piece_index(capture)]  # XOR out the captured piece

        piece = board.piece_at(move.from_square)
        idx = self._piece_index(piece)
        self.zobrist_key ^= self.zobrist_pieces[y_from][x_from][idx]  # XOR out the piece from the original square
        self.zobrist_key ^= self.zobrist_pieces[y_to][x_to][idx]  # XOR in the piece to the new square

        # Switch the active colour
        self.zobrist_key ^= self.zobrist_active_colour

        # Update zobrist key in the case of promotion
        if move.promotion is not None:
            self.zobrist_key ^= self.zobrist_pieces[y_to][x_to][idx]  # Remove the pawn from the eighth rank
            # Replace the pawn with the new piece
            if board.turn == chess.WHITE:
                self.zobrist_key ^= self.zobrist_pieces[y_to][x_to][move.promotion-1]
            else:
                self.zobrist_key ^= self.zobrist_pieces[y_to][x_to][move.promotion+5]

        # If move is a castling move, move the rook as well
        if board.is_castling(move):
            if board.is_queenside_castling(move):
                if board.turn == chess.WHITE:
                    self.zobrist_key ^= self.zobrist_pieces[0][0][3]  # XOR rook out of a1
                    self.zobrist_key ^= self.zobrist_pieces[0][3][3]  # XOR rook into d1
                else:
                    self.zobrist_key ^= self.zobrist_pieces[7][0][9]  # XOR rook out of a8
                    self.zobrist_key ^= self.zobrist_pieces[7][3][9]  # XOR rook into d8
            elif board.turn == chess.WHITE:
                self.zobrist_key ^= self.zobrist_pieces[0][7][3]  # XOR rook out of h1
                self.zobrist_key ^= self.zobrist_pieces[0][5][3]  # XOR rook into f1
            else:
                self.zobrist_key ^= self.zobrist_pieces[7][7][9]  # XOR rook out of h8
                self.zobrist_key ^= self.zobrist_pieces[7][5][9]  # XOR rook into f8

        ep_square_before = board.ep_square
        castling_rights_before = board.castling_rights
        board.push(move)

        # Update the zobrist key if the castling rights or if the en passant square has changed
        self._update_zobrist_castling_rights(board, castling_rights_before)
        self._update_zobrist_en_passant(board, ep_square_before)

    def _unmake_move(self, board, move, capture):
        """Unmake the last move and update the zobrist key accordingly"""
        y_to, x_to = divmod(move.to_square, 8)
        y_from, x_from = divmod(move.from_square, 8)

        if capture is not None:
            self.zobrist_key ^= self.zobrist_pieces[y_to][x_to][self._piece_index(capture)]  # XOR the captured piece back in

        piece = board.piece_at(move.to_square)
        idx = self._piece_index(piece)

        self.zobrist_key ^= self.zobrist_pieces[y_to][x_to][idx]  # XOR out the piece from the new square
        self.zobrist_key ^= self.zobrist_pieces[y_from][x_from][idx]  # XOR in the piece to the original square

        # Switch the active colour
        self.zobrist_key ^= self.zobrist_active_colour

        # Update zobrist key in the case of promotion
        if move.promotion is not None:
            self.zobrist_key ^= self.zobrist_pieces[y_from][x_from][idx]  # Remove the pawn from the eighth rank
            # Replace the pawn with the new piece
            if board.turn == chess.WHITE:
                self.zobrist_key ^= self.zobrist_pieces[y_from][x_from][0]
            else:
                self.zobrist_key ^= self.zobrist_pieces[y_from][x_from][6]
        
        ep_square_before = board.ep_square
        castling_rights_before = board.castling_rights
        board.pop()

        # Update the zobrist key if the castling rights or if the en passant square has changed
        self._update_zobrist_castling_rights(board, castling_rights_before)
        self._update_zobrist_en_passant(board, ep_square_before)

        # This has to be done afer the move is unmade, otherwise, 
        # Board.is_castling() will return False even if the move was a castling move
        # If the move was castling, unmove the rooks in the zobrist key
        if board.is_castling(move):
            if board.is_queenside_castling(move):
                if board.turn == chess.WHITE:
                    self.zobrist_key ^= self.zobrist_pieces[0][0][3]  # XOR rook back to a1
                    self.zobrist_key ^= self.zobrist_pieces[0][3][3]  # XOR rook out of d1
                else:
                    self.zobrist_key ^= self.zobrist_pieces[7][0][9]  # XOR rook back to a8
                    self.zobrist_key ^= self.zobrist_pieces[7][3][9]  # XOR rook out of d8
            elif board.turn == chess.WHITE:
                self.zobrist_key ^= self.zobrist_pieces[0][7][3]  # XOR rook back to h1
                self.zobrist_key ^= self.zobrist_pieces[0][5][3]  # XOR rook out of f1
            else:
                self.zobrist_key ^= self.zobrist_pieces[7][7][9]  # XOR rook back to h8
                self.zobrist_key ^= self.zobrist_pieces[7][5][9]  # XOR rook out of f8

    def _update_zobrist_castling_rights(self, board, castling_rights_before):
        """Update the zobrist key if any castling rights have changed"""
        # Board.castling rights is a bitmask and by using the 
        # XOR operator with the new bitmask after making the move, 
        # we can see if any castling rights have changed
        castling_rights_change = castling_rights_before ^ board.castling_rights

        # Kingside castling for white
        if castling_rights_change & chess.BB_H1:
            self.zobrist_key ^= self.zobrist_castling[0]
        # Queenside castling for white
        if castling_rights_change & chess.BB_A1:
            self.zobrist_key ^= self.zobrist_castling[1]
        # Kingside castling for black
        if castling_rights_change & chess.BB_H8:
            self.zobrist_key ^= self.zobrist_castling[2]
        # Queenside castling for black
        if castling_rights_change & chess.BB_A8:
            self.zobrist_key ^= self.zobrist_castling[3]

    def _update_zobrist_en_passant(self, board, ep_square_before):
        """Update the zobrist key if the en passant square has changed"""
        if board.ep_square != ep_square_before:
            # En passant no longer available
            if board.ep_square is None:
                self.zobrist_key ^= self.zobrist_enpassant[ep_square_before % 8]
            # New en passant possibility
            elif ep_square_before is None:
                self.zobrist_key ^= self.zobrist_enpassant[board.ep_square % 8]
            # Opportunity to take en passant is gone 
            # but now en passant is available to the opponent
            else:
                self.zobrist_key ^= self.zobrist_enpassant[board.ep_square % 8]
                self.zobrist_key ^= self.zobrist_enpassant[ep_square_before % 8]

    def _calculate_zobrist_key(self, board):
        """Calculate the zobrist key of the current board state"""
        self.zobrist_key = 0

        # Piece positions
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, chess.WHITE):
                i, j = divmod(square, 8)
                self.zobrist_key ^= self.zobrist_pieces[i][j][piece-1]
            for square in board.pieces(piece, chess.BLACK):
                i, j = divmod(square, 8)
                self.zobrist_key ^= self.zobrist_pieces[i][j][piece+5]
        
        # Active color
        if board.turn == chess.BLACK:
            self.zobrist_key ^= self.zobrist_active_colour

        # Enpassant
        if board.has_legal_en_passant():
            self.zobrist_key ^= self.zobrist_enpassant[board.ep_square % 8]

        # Kingside castling rights for white
        if board.castling_rights & chess.BB_H1:
            self.zobrist_key ^= self.zobrist_castling[0]
        # Queenside castling rights for white
        if board.castling_rights & chess.BB_A1:
            self.zobrist_key ^= self.zobrist_castling[1]
        # Kingside castling rights for black
        if board.castling_rights & chess.BB_H8:
            self.zobrist_key ^= self.zobrist_castling[2]
        # Queenside castling rights for black
        if board.castling_rights & chess.BB_A8:
            self.zobrist_key ^= self.zobrist_castling[3]

    def _bitboard(self, board):
        """Generate a boolean array representing a chess board"""
        # 768 bits for pieces, 8 bits for en passant, 4 bits for castling rights, and 1 bit to represent whose turn it is
        bitboard = np.zeros(781, dtype=bool)

        # Bits representing the pieces
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, chess.WHITE):
                bitboard[64 * (piece-1) + square] = True

            for square in board.pieces(piece, chess.BLACK):
                bitboard[64 * (piece+5) + square] = True
        
        # Bit representing whose turn it is
        if board.turn == chess.BLACK:
            bitboard[768] = True

        # Bits to represent the castling rights
        # Kingside castling for white
        if board.castling_rights & chess.BB_H1:
            bitboard[769] = True
        # Queenside castling for white
        if board.castling_rights & chess.BB_A1:
            bitboard[770] = True
        # Kingside castling for black
        if board.castling_rights & chess.BB_H8:
            bitboard[771] = True
        # Queenside castling for black
        if board.castling_rights & chess.BB_A8:
            bitboard[772] = True

        # 8 bits to represent the en passant row, if there is one
        if board.has_legal_en_passant():
            bitboard[773 + board.ep_square % 8] = True

        return bitboard

    def evaluate_position(self, board):
        """Return the evaluation of a chess position"""
        if (outcome := board.outcome()) is not None:
            if outcome.winner is None:
                return 0
            elif board.turn == outcome.winner:
                return 100000
            else: 
                return -100000

        return self.model(np.array([self._bitboard(board)]))[0][0]
    
    def _piece_index(self, piece):
        """Takes as input a chess.Piece instance and outputs a value from 0 to 11"""
        if piece.color == chess.WHITE:
            return piece.piece_type - 1
        return piece.piece_type + 5


# board = chess.Board()
# bot = ChessEngine()

# import cProfile, pstats
# profiler = cProfile.Profile()
# profiler.enable()

# bot.get_move(board)

# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats()

# # bot.get_move(board)
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