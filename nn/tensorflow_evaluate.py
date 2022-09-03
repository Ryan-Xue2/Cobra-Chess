#%% Import modules
import tensorflow as tf
import numpy as np
import chess
import chess.engine


#%% Load data into np arrays
data = np.load('dataset.npz')
train_boards = data['arr_0']
train_evals = data['arr_1']

#%% Build neural network model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(781)))
model.add(tf.keras.layers.Dense(300, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(300, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(300, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(300, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(300, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(300, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['accuracy'])

#%% Train the model on the training data and test the loss and accuracy
model.fit(train_boards, train_evals, epochs=1, batch_size=1024)

# %%
retrain_x = []
retrain_y = []

#%%
def bitboard(board):
    """Generate a boolean array representing a chess board"""
    # 768 bits for pieces, 8 bits for en passant, 4 bits for castling rights, and 1 bit to represent whose turn it is
    bitboard = np.zeros(781, dtype=bool)

    # Pieces
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            bitboard[64 * (piece-1) + square] = True

        for square in board.pieces(piece, chess.BLACK):
            bitboard[64 * (piece+5) + square] = True
    
    # Turn
    if board.turn == chess.BLACK:
        bitboard[768] = True

    # Castling rights
    if board.castling_rights & chess.BB_H1:
        bitboard[769] = True
    if board.castling_rights & chess.BB_A1:
        bitboard[770] = True
    if board.castling_rights & chess.BB_H8:
        bitboard[771] = True
    if board.castling_rights & chess.BB_A8:
        bitboard[772] = True

    # En passant
    if board.has_legal_en_passant():
        bitboard[773 + board.ep_square % 8] = True

    return bitboard

import random
def random_board(max_depth=150):
    """Generate a random chess board"""
    board = chess.Board()
    depth = random.randrange(0, max_depth)

    for _ in range(depth):
        all_moves = list(board.legal_moves)
        random_move = random.choice(all_moves)
        board.push(random_move)
        if board.is_game_over():
            break
    
    board.clear_stack()  # Clear the stack to not have the nn thinking position is draw when it is not
    return board

with chess.engine.SimpleEngine.popen_uci(r'C:\Users\16477\Downloads\stockfish_15_win_x64_avx2\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe') as sf:
    for _ in range(1000):
        board = random_board()
        info = sf.analyse(board, chess.engine.Limit(depth=0))
        score = info['score'].pov(board.turn).score(mate_score=100000)
        true = max(min(1500, score), -1500)
        pred = model(np.asarray([bitboard(board)]))
        if true < 0 and pred > 0:
            retrain_x.append(bitboard(board))
            retrain_y.append(true)

#%% 
retrain_x = np.array(retrain_x)
retrain_y = np.array(retrain_y)
# %% Save the trained model
model.save('chess_nn_model.h5')