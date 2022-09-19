from cobra.engine import CobraEngine
import chess


engine = CobraEngine()
board = chess.Board()

for _ in range(10):
    move = engine.get_move(board)
    board.push(move)
    engine.transposition.clear()
print(engine.killer)
    

