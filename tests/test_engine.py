import pytest
from cobra.cobra import Cobra
import chess

def test_zobrist_hash():
    bot = Cobra()
    board = chess.Board()
    bot._calculate_zobrist_key(board)
    key_before = bot.zobrist_key

    bot.get_move(board)
    
    assert key_before == bot.zobrist_key

    