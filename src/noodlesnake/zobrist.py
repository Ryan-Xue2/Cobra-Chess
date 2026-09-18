import chess
from random import randint, seed

class Zobrist:
    __slots__ = ('key', 'turn', 'castling', 'en_passant', 'pieces', 'rook_squares')
    def __init__(self):
        self.key = 0

        seed(1)
        self.turn = randint(0, 2**64)
        self.castling = [randint(0, 2**64) for _ in range(4)]
        self.en_passant = [randint(0, 2**64) for _ in range(8)]
        self.pieces = [[[randint(0, 2**64) for _ in range(6)] for _ in range(64)] for _ in range(2)]

        self.rook_squares = [chess.BB_H1, chess.BB_A1, chess.BB_H8, chess.BB_A8]

    def move(self, board, move):
        """
        Updates the position of the piece moved in the zobrist key 
        as well as the active colour.
        It is assumed that this method is called before the move is made.
        """
        # Update location of the piece
        piece = board.piece_at(move.from_square)
        self.key ^= self.pieces[piece.color][move.from_square][piece.piece_type-1]
        self.key ^= self.pieces[piece.color][move.to_square][piece.piece_type-1]

        # Update whose turn it is
        self.key ^= self.turn

    def unmove(self, board, move):
        """
        Updates the position of the piece to be unmoved in the zobrist key
        as well as the active colour.
        It is assumed that the this method is called before the move is undone.
        """
        # Update location of the piece
        piece = board.piece_at(move.to_square)
        self.key ^= self.pieces[piece.color][move.to_square][piece.piece_type-1]
        self.key ^= self.pieces[piece.color][move.from_square][piece.piece_type-1]

        # Update whose turn it is
        self.key ^= self.turn
    
    def update_capture(self, capture_square, captured_pc):
        """Update the zobrist key accordingly for a capture"""
        self.key ^= self.pieces[captured_pc.color][capture_square][captured_pc.piece_type-1]

    def promote(self, board, move):
        """
        Update the zobrist key to account for a pawn promoting.
        It is assumed that this method is called before the piece is promoted.
        """
        # Remove pawn from 8th or 1st rank
        self.key ^= self.pieces[board.turn][move.to_square][chess.PAWN-1]
        # Put the promoted piece where the pawn was
        self.key ^= self.pieces[board.turn][move.to_square][move.promotion-1]

    def unpromote(self, board, move):
        """
        Update the zobrist key to account for a piece unpromoting. 
        It is assumed that this method is called before the piece is unpromoted.
        """
        # Remove promoted piece from the board
        self.key ^= self.pieces[not board.turn][move.from_square][move.promotion-1]
        # Put pawn where promoted piece was
        self.key ^= self.pieces[not board.turn][move.from_square][chess.PAWN-1]

    def move_rook_if_castle(self, board, move):
        """
        This method should be called if the move made is a castling move.
        This method will update the rook's position in the zobrist
        key after being castled.
        """
        if board.is_queenside_castling(move):
            if board.turn == chess.WHITE:
                self.key ^= self.pieces[1][chess.A1][chess.ROOK-1]
                self.key ^= self.pieces[1][chess.D1][chess.ROOK-1]
            else:
                self.key ^= self.pieces[0][chess.A8][chess.ROOK-1]
                self.key ^= self.pieces[0][chess.D8][chess.ROOK-1]
        elif board.turn == chess.WHITE:
            self.key ^= self.pieces[1][chess.H1][chess.ROOK-1]
            self.key ^= self.pieces[1][chess.F1][chess.ROOK-1]
        else:
            self.key ^= self.pieces[0][chess.H8][chess.ROOK-1]
            self.key ^= self.pieces[0][chess.F8][chess.ROOK-1]

    def update_en_passant(self, board, ep_square_before, ep_available_before):
        """Update the zobrist key if the en passant availibility has changed"""
        ep_available = board.has_legal_en_passant()

        # En passant no longer available
        if not ep_available and ep_available_before:
            self.key ^= self.en_passant[ep_square_before % 8]
        # En passant was not available but is now
        elif not ep_available_before and ep_available:
            self.key ^= self.en_passant[board.ep_square % 8]
        # En passant was available and now is available for the opponent
        elif ep_available:
            self.key ^= self.en_passant[board.ep_square % 8]
            self.key ^= self.en_passant[ep_square_before % 8]
    
    def update_castling_rights(self, board, castling_rights_before):
        """Update the zobrist key if any castling rights have changed"""
        # Bitmask of the rooks whose castling rights have changed
        castling_rights_change = castling_rights_before ^ board.castling_rights

        for i, rook in enumerate(self.rook_squares):
            if castling_rights_change & rook:
                self.key ^= self.castling[i]

    def calculate_zobrist_key(self, board):
        """Calculate the zobrist key of the current board state"""
        self.key = 0
        
        # Piece positions
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, chess.WHITE):
                self.key ^= self.pieces[1][square][piece-1]
            for square in board.pieces(piece, chess.BLACK):
                self.key ^= self.pieces[0][square][piece-1]
        
        # Active color
        if board.turn == chess.BLACK:
            self.key ^= self.turn

        # Enpassant
        if board.has_legal_en_passant():
            self.key ^= self.en_passant[board.ep_square % 8]

        # Castling rights
        for i, rook in enumerate(self.rook_squares):
            if board.castling_rights & rook:
                self.key ^= self.castling[i]
    
    def update_null_move(self, board, ep_square_before, ep_available_before):
        """Update the zobrist key in the case of a null move"""
        # Change turn
        self.key ^= self.turn

        # Update en passant
        self.update_en_passant(board, ep_square_before, ep_available_before)