import sqlite3
import numpy as np
import json


data = np.load('dataset.npz')
x = data['arr_0']
y = data['arr_1']


con = sqlite3.connect('dataset.db')
cur = con.cursor()
# cur.execute('DELETE FROM scores')
# # cur.execute('CREATE TABLE scores(board, score)')

# data = []
# for board, score in zip(x, y):
#     board = json.dumps(board.tolist())
#     score = int(score)
#     data.append((board, score))

# cur.executemany('INSERT INTO scores VALUES(?, ?)', data)
# con.commit()

res = cur.execute('SELECT * FROM scores')
board, score = res.fetchone()
print(board, score)
print(type(board))