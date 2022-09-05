UPPER = 0
LOWER = 1
EXACT = 2

# TODO: Draw detection

class TranspositionTable:
    __slots__ = ('transposition',)
    def __init__(self):
        self.transposition = {}
    
    def lookup(self, key):
        return self.transposition.get(key)
    
    def store(self, key, entry):
        self.transposition[key] = entry
    
    def clear(self):
        self.transposition.clear()


class TranspositionTableEntry:
    __slots__ = ('move', 'depth', 'flag', 'score')
    def __init__(self, flag, depth, move, score):
        self.move = move
        self.depth = depth
        self.flag = flag
        self.score = score