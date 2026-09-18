import sys
import threading

import chess

from noodlesnake.engine import NoodlesnakeEngine


class UCIAdapter:
    """Expose Noodlesnake through the UCI commands needed to play games."""

    def __init__(self):
        """Initialize the engine and starting position."""
        self.board = chess.Board()
        self.engine = NoodlesnakeEngine()
        self.search_thread = None
        self.stop_event = None

    @staticmethod
    def send(message):
        """Write a UCI response immediately."""
        print(message, flush=True)

    def run(self):
        """Read UCI commands until the client sends quit."""
        for line in sys.stdin:
            if not self.handle(line.strip()):
                break

    def handle(self, line):
        """Handle one UCI command."""
        command, *arguments = line.split()

        if command == 'uci':
            self.send('id name Noodlesnake')
            self.send('id author Ryan Xue')
            self.send('uciok')
        elif command == 'isready':
            self.send('readyok')
        elif command == 'ucinewgame':
            self._stop_search(wait=True)
            self.engine = NoodlesnakeEngine()
        elif command == 'position':
            self._stop_search(wait=True)
            self._set_position(arguments)
        elif command == 'go':
            self._start_search(arguments)
        elif command == 'stop':
            self._stop_search()
        elif command == 'quit':
            self._stop_search(wait=True)
            return False
        else:
            raise ValueError(f'unsupported command: {command}')

        return True

    def _set_position(self, arguments):
        """Replace the current board from a UCI position command."""
        if arguments[0] == 'startpos':
            board = chess.Board()
            move_index = 1
        elif arguments[0] == 'fen':
            move_index = arguments.index('moves') if 'moves' in arguments else len(arguments)
            board = chess.Board(' '.join(arguments[1:move_index]))

        for move in arguments[move_index + 1:]:
            board.push_uci(move)

        self.board = board

    def _start_search(self, arguments):
        """Start a cancellable search from a UCI go command."""
        self._stop_search(wait=True)
        time_limit = self._parse_go(arguments)
        board = self.board.copy()
        self.stop_event = threading.Event()
        stop_event = self.stop_event

        def search():
            move = self.engine.get_move(
                board,
                time_limit=time_limit,
                stop_event=stop_event,
            )
            self.send(f'bestmove {move.uci()}')

        self.search_thread = threading.Thread(target=search, name='noodlesnake-search')
        self.search_thread.start()

    def _stop_search(self, wait=False):
        """Cancel the current search and optionally wait for it."""
        if self.stop_event is not None:
            self.stop_event.set()
        if not wait:
            return
        if self.search_thread is not None:
            self.search_thread.join()
        self.search_thread = None
        self.stop_event = None

    def _parse_go(self, arguments):
        """Convert Lichess clock fields into a move time."""
        values = {}
        for index in range(0, len(arguments), 2):
            values[arguments[index]] = int(arguments[index + 1])

        if 'movetime' in values:
            return values['movetime'] / 1000

        side = 'w' if self.board.turn == chess.WHITE else 'b'
        remaining_seconds = values[f'{side}time'] / 1000
        increment_seconds = values[f'{side}inc'] / 1000

        # Budget 30 moves plus 80% of the increment, capped at a quarter of the clock.
        return min(
            remaining_seconds * 0.25,
            remaining_seconds / 30 + increment_seconds * 0.8,
        )


def main():
    UCIAdapter().run()


if __name__ == '__main__':
    main()
