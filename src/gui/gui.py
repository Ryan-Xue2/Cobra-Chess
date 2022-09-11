import pygame
import chess
import sys



class Gui:
    def __init__(self, board):
        pygame.init()
        self.SQUARE_SIZE = 100
        self.WIDTH = self.SQUARE_SIZE * 8
        self.HEIGHT = self.SQUARE_SIZE * 8
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.board = board
        self.dark_sq_colour = (180, 135, 102)
        self.light_sq_colour = (240, 217, 183)

        self.background = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.init_background()

        self.display_pieces()
        
        self.selected_sq = None

        pygame.display.set_caption('Chess GUI')

    def init_background(self):
        for y in range(0, self.HEIGHT, self.SQUARE_SIZE):
            for x in range(0, self.WIDTH, self.SQUARE_SIZE):
                color = self.light_sq_colour if y % (2*self.SQUARE_SIZE) == x % (2*self.SQUARE_SIZE) else self.dark_sq_colour
                rect = x, y, self.SQUARE_SIZE, self.SQUARE_SIZE
                pygame.draw.rect(self.background, color, rect)
    
    def display_pieces(self):
        piece_dict = {
            1: 'pawn',
            2: 'knight',
            3: 'bishop',
            4: 'rook', 
            5: 'queen',
            6: 'king'
        }
        for piece in chess.PIECE_TYPES:
            for square in self.board.pieces(piece, chess.WHITE):
                square = 63-square
                y, x = divmod(square, 8)

                img = pygame.image.load(f'images/white_{piece_dict[piece]}.png')
                img = pygame.transform.smoothscale(img, (self.SQUARE_SIZE, self.SQUARE_SIZE))
                self.screen.blit(img, (x*self.SQUARE_SIZE, y*self.SQUARE_SIZE))
                
            for square in self.board.pieces(piece, chess.BLACK):
                square = 63-square
                y, x = divmod(square, 8)

                img = pygame.image.load(f'images/black_{piece_dict[piece]}.png')
                img = pygame.transform.smoothscale(img, (self.SQUARE_SIZE, self.SQUARE_SIZE))
                self.screen.blit(img, (x*self.SQUARE_SIZE, y*self.SQUARE_SIZE))
                
    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._check_mousedown_events(event)
            if event.type == pygame.QUIT:
                sys.exit()
        
    def _check_mousedown_events(self, event):
        x, y = event.pos
        square = y // self.SQUARE_SIZE * 8 + x // self.SQUARE_SIZE
        
        if self.selected_sq is not None:
            if square != self.selected_sq:
                from_sq = 63-self.selected_sq
                to_sq = 63-square
                
                move = chess.Move.from_uci(self.to_uci(from_sq, to_sq))
                print(move)
                if move in self.board.legal_moves:
                    self.make_move(move)
                    self.selected_sq = None
                else:
                    self.selected_sq = square
                
            else:
                self.selected_sq = None
        else:
            self.selected_sq = square
        print('Currect selected square:', self.selected_sq)
    
    def to_uci(self, from_sq, to_sq):
        col_to_alpha = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        from_uci = col_to_alpha[from_sq % 8] + str(from_sq // 8 + 1)
        to_uci = col_to_alpha[to_sq % 8] + str(to_sq // 8 + 1)
        return from_uci + to_uci

    def make_move(self, move):
        if self.board.move_stack:
            last_move = self.board.peek()
            self.unhighlight_square(63-last_move.from_square)
            self.unhighlight_square(63-last_move.to_square)
        self.board.push(move)
        square = 63-move.from_square
        self.highlight_square(square)

        square = 63-move.to_square
        self.highlight_square(square)
        self._update_screen()

    def highlight_square(self, square):
        y, x = divmod(square, 8)
        rect = (x * self.SQUARE_SIZE, y * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE)
        dark_highlight = (170,163,59)
        light_highlight = (204,211,103)
        colour = light_highlight if y % 2 == x % 2 else dark_highlight
        pygame.draw.rect(self.background, colour, rect)
    
    def unhighlight_square(self, square):
        y, x = divmod(square, 8)
        rect = (x * self.SQUARE_SIZE, y * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE)
        colour = self.light_sq_colour if y % 2 == x % 2 else self.dark_sq_colour
        pygame.draw.rect(self.background, colour, rect)

    def _update_screen(self):
        self.screen.blit(self.background, (0, 0))
        self.display_pieces()
        pygame.display.flip()