import os
import sys
import threading

import berserk
import chess
import chess.engine
from dotenv import load_dotenv


class LichessBot:
    """Connect the Noodlesnake UCI engine to Lichess through Berserk."""

    def __init__(self, token):
        """Create the Lichess client and game state."""
        self.client = berserk.Client(session=berserk.TokenSession(token))
        self.pending_challenge_id = None
        self.game_id = None
        self.state_lock = threading.Lock()

    def run(self):
        """Process challenges and game starts from Lichess."""
        for event in self.client.bots.stream_incoming_events():
            if event['type'] == 'challenge':
                self._handle_challenge(event['challenge'])
            elif event['type'] == 'challengeCanceled':
                self._handle_canceled_challenge(event['challenge']['id'])
            elif event['type'] == 'gameStart':
                self._start_game(event['game'])

    def _handle_challenge(self, challenge):
        """Accept one casual standard clock challenge at a time."""
        if challenge['variant']['key'] != 'standard':
            self.client.bots.decline_challenge(challenge['id'], reason='variant')
            return
        if challenge['rated']:
            self.client.bots.decline_challenge(challenge['id'], reason='casual')
            return
        if challenge['timeControl']['type'] != 'clock':
            self.client.bots.decline_challenge(challenge['id'], reason='timeControl')
            return

        with self.state_lock:
            busy = (
                self.pending_challenge_id is not None
                or self.game_id is not None
            )

        if busy:
            self.client.bots.decline_challenge(challenge['id'], reason='later')
            return

        self.client.bots.accept_challenge(challenge['id'])
        with self.state_lock:
            self.pending_challenge_id = challenge['id']

    def _handle_canceled_challenge(self, challenge_id):
        """Release a challenge slot when its sender cancels."""
        with self.state_lock:
            if challenge_id == self.pending_challenge_id:
                self.pending_challenge_id = None

    def _start_game(self, game):
        """Start the game stream without blocking the event stream."""
        with self.state_lock:
            if self.game_id is not None:
                raise RuntimeError('Lichess started a second game while Noodlesnake was busy')
            self.pending_challenge_id = None
            self.game_id = game['gameId']

        threading.Thread(
            target=self._play_game,
            args=(game['gameId'], game['color']),
            name=f"lichess-game-{game['gameId']}",
        ).start()

    def _play_game(self, game_id, color):
        """Play moves until the Lichess game stream reports completion."""
        our_color = chess.WHITE if color == 'white' else chess.BLACK

        try:
            with chess.engine.SimpleEngine.popen_uci(
                [sys.executable, '-m', 'noodlesnake.uci']
            ) as engine:
                for event in self.client.bots.stream_game_state(game_id):
                    if event['type'] == 'gameFull':
                        state = event['state']
                        clocks_are_milliseconds = True
                    elif event['type'] == 'gameState':
                        state = event
                        clocks_are_milliseconds = False
                    else:
                        continue

                    if state['status'] != 'started':
                        break

                    board = chess.Board()
                    for move in state['moves'].split():
                        board.push_uci(move)

                    if board.turn != our_color:
                        continue

                    if clocks_are_milliseconds:
                        white_clock = state['wtime'] / 1000
                        black_clock = state['btime'] / 1000
                        white_increment = state['winc'] / 1000
                        black_increment = state['binc'] / 1000
                    else:
                        # Berserk converts clocks on gameState events to timedeltas.
                        white_clock = state['wtime'].total_seconds()
                        black_clock = state['btime'].total_seconds()
                        white_increment = state['winc'].total_seconds()
                        black_increment = state['binc'].total_seconds()

                    result = engine.play(
                        board,
                        chess.engine.Limit(
                            white_clock=white_clock,
                            black_clock=black_clock,
                            white_inc=white_increment,
                            black_inc=black_increment,
                        ),
                    )
                    self.client.bots.make_move(game_id, result.move.uci())
        finally:
            with self.state_lock:
                self.game_id = None


def main():
    """Run Noodlesnake using the token from the environment."""
    load_dotenv()
    LichessBot(os.environ['LICHESS_BOT_TOKEN']).run()


if __name__ == '__main__':
    main()
