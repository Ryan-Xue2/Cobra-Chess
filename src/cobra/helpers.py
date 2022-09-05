import chess


def captured_piece_square(board, move):
    if board.is_capture(move):
        if board.is_en_passant(move):
            return board.ep_square - (8 if board.turn == chess.WHITE else -8)
        else:
            return move.to_square
    return None