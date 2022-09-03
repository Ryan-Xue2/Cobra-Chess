import numpy as np
import chess
from random import randint


class Zobrist:
    __slots__ = ('key', 'active_colour', 'en_passant', 'castling_rights', 'pieces')
    def __init__(self):
        self.key = 0
        self.active_colour = randint(0, 2**64)
        self.en_passant = [randint(0, 2**64) for _ in range(8)]
        self.castling_rights = [randint(0, 2**64) for _ in range(4)]
        self.pieces = [[[randint(0, 2**64) for _ in range(12)] for _ in range(8)] for _ in range(8)]

    def calculate_zobrist_key(self, board):
        """Calculate the zobrist key of the current board state"""
        self.key = 0
        
        # Piece positions
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, chess.WHITE):
                i, j = divmod(square, 8)
                self.key ^= self.pieces[i][j][piece-1]
            for square in board.pieces(piece, chess.BLACK):
                i, j = divmod(square, 8)
                self.key ^= self.pieces[i][j][piece+5]
        
        # Active color
        if board.turn == chess.BLACK:
            self.key ^= self.active_colour

        # Enpassant
        if board.has_legal_en_passant():
            self.key ^= self.en_passant[board.ep_square % 8]

        # Kingside castling rights for white
        if board.castling_rights & chess.BB_H1:
            self.key ^= self.zobrist_castling[0]
        # Queenside castling rights for white
        if board.castling_rights & chess.BB_A1:
            self.key ^= self.zobrist_castling[1]
        # Kingside castling rights for black
        if board.castling_rights & chess.BB_H8:
            self.key ^= self.zobrist_castling[2]
        # Queenside castling rights for black
        if board.castling_rights & chess.BB_A8:
            self.key ^= self.zobrist_castling[3]
    
    def _piece_index(self, piece):
        """Takes as input a chess.Piece instance and outputs a value from 0 to 11"""
        if piece.color == chess.WHITE:
            return piece.piece_type - 1
        return piece.piece_type + 5


