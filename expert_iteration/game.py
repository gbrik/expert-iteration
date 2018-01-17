import numpy as np
from typing import *

BoardState = TypeVar('BoardState')
Action = NewType('Action', int)
Player = NewType('Player', int)

class State(Generic[BoardState]):
    def __init__(self, board_state: BoardState, player: Player, prev_action: Action) -> None:
        self.board_state = board_state
        self.player = player
        self.prev_action = prev_action

class Game(Generic[BoardState]):
    def __init__(self) -> None:
        pass

    def gen_roots(self, num: int) -> np.ndarray:
        return np.array([self.gen_root() for _ in range(num)])
    def gen_root(self) -> State[BoardState]:
        return self.gen_roots(1)[0]

    def do_actions(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.array([self.do_action(state, action) for state, action in zip(states, actions)])
    def do_action(self, state: State[BoardState], action: Action) -> State[BoardState]:
        return self.do_actions(np.array([state]), np.array([action]))[0]

    def valid_actionses(self, states: np.ndarray) -> np.ndarray:
        return np.array([self.valid_actions(state) for state in states])
    def valid_actions(self, state: State[BoardState]) -> np.ndarray:
        return self.valid_actionses(np.array([state]))[0]

    def check_ends(self, states: np.ndarray) -> np.ndarray:
        return np.array([self.check_end(state) for state in states])
    def check_end(self, state: State[BoardState]) -> np.ndarray:
        return self.check_ends(np.array([state]))[0]

    @property
    def num_actions(self) -> int:
        raise NotImplemented
    @property
    def num_players(self) -> int:
        raise NotImplemented

    def render(self, state: State[BoardState]):
        print('Player %d\'s turn:' % state.player)
        print(state.board_state)

    def parse(self, s: str):
        return int(s)

class GamePlayer(Generic[BoardState]):
    """
    Plays games.

    Since some GamePlayers need to maintain internal state, it is expected that next_turn
    is called exactly once, in order, for each turn of the game, regardless of whether it's
    the current player's turn.
    """

    def next_turn(self, state: State[BoardState]) -> Action:
        """
        Should be called every turn, regardless of whether it is the player's turn.
        Returns the player's move if it's their turn, otherwise returns None.
        """
        if state.player in self.players:
            return self._take_turn(state)
        else:
            self._watch_turn(state)
            return None

    def _take_turn(self, state: State[BoardState]) -> Action: #type: ignore
        """
        Select the best move.
        """
        raise NotImplemented

    def _watch_turn(self, state: State[BoardState]):
        """
        Keep the player's internal state up to date with the game state.
        Unnecessary if the player has no internal state.
        """
        pass

    def __init__(self, players: Set[Player], take_turn = None) -> None:
        """
        Which players this Player object is playing as.
        """
        self.players = players
        if take_turn:
            self._take_turn = take_turn #type: ignore

class GameAlgorithm(Generic[BoardState]):
    """
    Instantiates GamePlayers with a strategy.
    """

    def __init__(self, game: Game[BoardState]) -> None:
        self.game = game

    def mk_player(self, players: Set[Player]) -> GamePlayer[BoardState]:
        raise NotImplemented

    def __call__(self, players: Set[Player]) -> GamePlayer[BoardState]:
        """
        Simply calls mk_player, provided for convenience.
        """
        return self.mk_player(players)

def play_game(game: Game[BoardState], mk_players: List[Tuple[Set[Player], GameAlgorithm[BoardState]]]):
    players = [ alg.mk_player(playing_as) for playing_as, alg in mk_players ]
    cur_state = game.gen_root()
    cur_ends = game.check_end(cur_state)
    while not np.any(cur_ends):
        cur_action = None
        for player in players:
            act = player.next_turn(cur_state)
            if act != None:
                cur_action = act
        if cur_action == None:
            print(cur_state.board_state)
            raise ValueError('No player played a valid action.')
        if(cur_state.board_state[cur_action] != 0): #type: ignore
            print('play_game invalid action: ', cur_state.board_state, cur_state.player, cur_action)
        cur_state = game.do_action(cur_state, cur_action)
        cur_ends = game.check_end(cur_state)
    if np.sum(np.abs(cur_state.board_state)) < 3:
        print('ending early: ', cur_state.board_state)
    return cur_state, cur_ends, players

class UserAlgorithm(GameAlgorithm[BoardState]):
    def __init__(self, game: Game[BoardState]) -> None:
        self.game = game

    def mk_player(self, players: Set[Player]):
        def take_turn(state: State[BoardState]) -> Action:
            self.game.render(state)
            ret = input('What is your move? ')
            while True:
                try:
                    parsed = self.game.parse(ret)
                except ValueError:
                    ret = input('What is your move? ')
                else:
                    return parsed

        return GamePlayer(players, take_turn)
