import chess
import numpy as np


def captured_piece_square(board, move):
    if board.is_capture(move):
        if board.is_en_passant(move):
            return board.ep_square - (8 if board.turn == chess.WHITE else -8)
        else:
            return move.to_square
    return None


def bitboard(board):
    """Generate a boolean array representing a chess board"""
    # 768 bits for pieces, 8 bits for en passant, 4 bits for castling rights, and 1 bit to represent whose turn it is
    bitboard = np.zeros(781, dtype=bool)

    # Bits representing the pieces
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            bitboard[64 * (piece-1) + square] = 1

        for square in board.pieces(piece, chess.BLACK):
            bitboard[64 * (piece+5) + square] = 1
    
    # Bit representing whose turn it is
    if board.turn == chess.BLACK:
        bitboard[768] = 1

    # Bits to represent the castling rights
    # Kingside castling for white
    if board.castling_rights & chess.BB_H1:
        bitboard[769] = 1
    # Queenside castling for white
    if board.castling_rights & chess.BB_A1:
        bitboard[770] = 1
    # Kingside castling for black
    if board.castling_rights & chess.BB_H8:
        bitboard[771] = 1
    # Queenside castling for black
    if board.castling_rights & chess.BB_A8:
        bitboard[772] = 1

    # 8 bits to represent the en passant row, if there is one
    if board.has_legal_en_passant():
        bitboard[773 + board.ep_square % 8] = 1

    return bitboard