from gui import Gui 
import chess
from cobra.engine import CobraEngine
from time import sleep

board = chess.Board()
print(board)
gui = Gui(board)
engine = CobraEngine()

while True:
    gui.check_events()
    if board.turn == chess.BLACK:
        move = engine.get_move(board)
        gui.make_move(move)
    if board.is_game_over():
        sleep(3)
        print('Game over')
        break