import numpy as np
from typing import *

BoardState = TypeVar('BoardState')
Action = NewType('Action', int)
Player = NewType('Player', int)

class State(Generic[BoardState]):
    def __init__(self, board_state: BoardState, player: Player, prev_action: Action, turn_num: int) -> None:
        self.board_state = board_state
        self.player = player
        self.prev_action = prev_action
        self.turn_num = turn_num

    def __hash__(self) -> int:
        return hash((self.board_state, self.player, self.prev_action, self.turn_num))

    def __eq__(self, other) -> bool:
        raise (self.board_state == other.board_state
               and self.player == other.player
               and self.prev_action == other.prev_action
               and self.turn_num == other.turn_num)

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

    def parse(self, s: str, state: State[BoardState]):
        return int(s)

class GamePlayer(Generic[BoardState]):
    """
    Plays games.

    Since some GamePlayers need to maintain internal state, it is expected that next_turn
    is called exactly once, in order, for each turn of the game, regardless of whether it's
    the current player's turn.
    """

    def next_turn(self, state: State[BoardState], game_idx: int = 0) -> Action:
        """
        Should be called every turn, regardless of whether it is the player's turn.
        Returns the player's move if it's their turn, otherwise returns None.
        """
        if state.player in self.players:
            return self._take_turn(state, game_idx)
        else:
            self._watch_turn(state, game_idx)
            return None

    def next_turns(self, states: np.ndarray, game_idxs: np.ndarray) -> np.ndarray:
        """
        Should be called every turn with each remaining game, regardless of whether it is the player's turn.
        Returns the player's move if it's their turn, otherwise returns None.
        """
        actions = np.full(game_idxs.size, -1)
        take_turn = np.array([ state.player in self.players for state in states ])
        if np.any(take_turn):
            actions[take_turn] = self._take_turns(states[take_turn], game_idxs[take_turn])
        if np.any(~take_turn):
            self._watch_turns(states[~take_turn], game_idxs[~take_turn])
        return actions

    def _take_turn(self, state: State[BoardState], game_idx: int) -> Action:
        """
        Select the best move.
        """
        return self._take_turns(np.array([state]), np.array([game_idx]))[0]

    def _take_turns(self, states: np.ndarray, game_idxs: np.ndarray) -> np.ndarray:
        """
        Select the best move.
        """
        return np.array([ self._take_turn(state, game_idx) for state, game_idx in zip(states, game_idxs) ])

    def _watch_turn(self, state: State[BoardState], game_idx: int):
        """
        Keep the player's internal state up to date with the game state.
        Unnecessary if the player has no internal state.
        """
        self._watch_turns(np.array([state]), np.array([game_idx]))

    def _watch_turns(self, states: np.ndarray, game_idxs: np.ndarray):
        """
        Keep the player's internal state up to date with the game state.
        Unnecessary if the player has no internal state.
        """
        for state, game_idx in zip(states, game_idxs):
            self._watch_turn(state, game_idx)

    def __init__(self, players: Set[Player], num_games: int,
                 take_turn = None, take_turns = None, watch_turn = None, watch_turns = None) -> None:
        """
        Which players this Player object is playing as.
        """
        self.players = players
        self.num_games = num_games
        if take_turn:
            self._take_turn = take_turn #type: ignore
        if take_turns:
            self._take_turns = take_turns #type: ignore
        if watch_turn:
            self._watch_turn = watch_turn #type: ignore
        if watch_turns:
            self._watch_turns = watch_turns #type: ignore

class GameAlgorithm(Generic[BoardState]):
    """
    Instantiates GamePlayers with a strategy.
    """

    def __init__(self, game: Game[BoardState]) -> None:
        self.game = game

    def mk_player(self, players: Set[Player], num_games: int = 1) -> GamePlayer[BoardState]:
        raise NotImplemented

def play_game(game: Game[BoardState], mk_players: List[Tuple[Set[Player], GameAlgorithm[BoardState]]]):
    states, ends, players = play_games(1, game, mk_players)
    return states[0], ends[0], players

def play_games(num_games: int, game: Game[BoardState], mk_players: List[Tuple[Set[Player], GameAlgorithm[BoardState]]]):
    players = [ alg.mk_player(playing_as, num_games) for playing_as, alg in mk_players ]
    states = game.gen_roots(num_games)
    results = np.empty((num_games, game.num_players), dtype=np.bool)
    ongoing_games = np.arange(num_games)
    while ongoing_games.size > 0:
        cur_actions = np.full(ongoing_games.size, -1, dtype=np.int)
        for player in players:
            acts = player.next_turns(states[ongoing_games], ongoing_games)
            np.maximum(cur_actions, acts, out=cur_actions)
        states[ongoing_games] = game.do_actions(states[ongoing_games], cur_actions)
        check_ends = game.check_ends(states[ongoing_games])
        ended = np.any(check_ends, axis=1)
        results[ongoing_games[ended]] = check_ends[ended]
        ongoing_games = ongoing_games[~ended]

    return states, results, players

class UserAlgorithm(GameAlgorithm[BoardState]):
    def __init__(self, game: Game[BoardState]) -> None:
        self.game = game

    def mk_player(self, players: Set[Player], num_games: int = 1):
        def take_turn(state: State[BoardState], game_idx: int) -> Action:
            self.game.render(state)
            ret = input('What is your move? ')
            while True:
                try:
                    parsed = self.game.parse(ret, state)
                except ValueError:
                    ret = input('What is your move? ')
                else:
                    return parsed

        def watch_turn(state: State[BoardState], game_idx: int):
            pass

        return GamePlayer(players, num_games, take_turn=take_turn, watch_turn=watch_turn)
