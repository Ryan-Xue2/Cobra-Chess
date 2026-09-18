from noodlesnake.engine import NoodlesnakeEngine
import chess


engine = NoodlesnakeEngine()
board = chess.Board()

for _ in range(10):
    move = engine.get_move(board)
    board.push(move)
    engine.transposition.clear()
print(engine.killer)
    

