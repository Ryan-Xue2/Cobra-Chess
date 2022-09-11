import chess
import chess.engine
import random
import numpy as np

from time import perf_counter
from multiprocessing import Pool
from cobra import helpers


def main():
    """Generate training data for the chess neural network"""
    datapoints = 5000000
    y = np.empty(datapoints, dtype=np.int32)

    start = perf_counter()
    print('Generating boards...')

    with Pool(processes=8) as p:
        boards = p.map(random_board, [random.randint(5, 150) for _ in range(datapoints)])
        # x = p.map(bitboard, boards)

    print(perf_counter() - start)

    x = [helpers.bitboard(board) for board in boards]
    print('Finished generating boards')
    print(perf_counter() - start)

    start = perf_counter()
    print('Analyzing with stockfish...')

    with chess.engine.SimpleEngine.popen_uci(r'C:\Users\16477\Downloads\stockfish_15_win_x64_avx2\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe') as sf:
            for i, board in enumerate(boards):
                # Get the evaluation from stockfish at depth 0 in centipawns
                info = sf.analyse(board, chess.engine.Limit(depth=0))
                score = info['score'].pov(board.turn).score(mate_score=100000)
                y[i] = max(min(score, 1500), -1500)  # Max the evals at 1500 and min them at -1500 so that values aren't too extreme

    print('Done analyzing')
    print(perf_counter() - start)

    # Save x and y to a npz file
    np.savez('dataset.npz', np.asarray(x), y)


def random_board(depth):
    """Make x number of random moves from the starting chess position and return the result"""
    board = chess.Board()

    for _ in range(depth):
        random_move = random.choice(list(board.legal_moves))
        board.push(random_move)
        if board.is_game_over():
            break
    
    board.clear_stack()  # Clear the stack to not have the nn thinking position is draw when it is not (repetition)
    return board


if __name__ == '__main__':
    main()