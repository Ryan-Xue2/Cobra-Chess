from cobra.zobrist import Zobrist
from cobra import helpers


class Controller:
    __slots__ = ('board', 'captures', 'zobrist')
    def __init__(self, board=None):
        self.board = board
        self.captures = []
        self.zobrist = Zobrist()
        if board is not None:
            self.zobrist.calculate_zobrist_key(board)

    def set_board(self, board):
        """
        Set the board to the new board, clear the list of captures, 
        and recalculate the zobrist key
        """
        self.board = board
        self.captures.clear()
        self.zobrist.calculate_zobrist_key(board)

    def move(self, move):
        """Make the move passed in and update the zobrist key accordingly"""
        self.zobrist.move(self.board, move)

        capture_square = helpers.captured_piece_square(self.board, move)
        if capture_square is not None:
            captured_pc = self.board.piece_at(capture_square)
        else:
            captured_pc = None
        self.captures.append((capture_square, captured_pc))

        # Update zobrist key in case of a captured piece
        if capture_square is not None:
            self.zobrist.update_capture(capture_square, captured_pc)
        # Update zobrist key in case of promotion
        if move.promotion is not None:
            self.zobrist.promote(self.board, move)
        # If the move is a castling move, update the position of the rook
        if self.board.is_castling(move):
            self.zobrist.move_rook_if_castle(self.board, move)

        # Keep track of the en passant possibilities and castling rights before moving
        ep_square_before = self.board.ep_square
        ep_available_before = self.board.has_legal_en_passant()
        castling_rights_before = self.board.castling_rights

        self.board.push(move)

        # Update the castling rights and en passant rights if necessary
        self.zobrist.update_castling_rights(self.board, castling_rights_before)
        self.zobrist.update_en_passant(self.board, ep_square_before, ep_available_before)


    def unmove(self):
        """Undo the last move and update zobrist key accordingly"""
        move = self.board.peek()
        self.zobrist.unmove(self.board, move)

        # Uncapture the captured piece if there is one
        capture_square, captured_pc = self.captures.pop()
        if capture_square is not None:
            self.zobrist.update_capture(capture_square, captured_pc)
        # Update zobrist key in case of promotion
        if move.promotion is not None:
            self.zobrist.unpromote(self.board, move)
        
        # Keep track of the en passant possibilities and castling rights before undoing the move
        ep_square_before = self.board.ep_square
        ep_available_before = self.board.has_legal_en_passant()
        castling_rights_before = self.board.castling_rights

        self.board.pop()

        # If the move is a castling move, then update the position of the rook in the zobrist key
        if self.board.is_castling(move):
            self.zobrist.move_rook_if_castle(self.board, move)

        # Update the castling rights and en passant rights if necessary
        self.zobrist.update_castling_rights(self.board, castling_rights_before)
        self.zobrist.update_en_passant(self.board, ep_square_before, ep_available_before)
