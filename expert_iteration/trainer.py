import numpy as np
import threading
import queue
from typing import TypeVar, Generic, List, Tuple
from .game import State, play_game, play_games
from .model import Model
from . import mcts
from . import utils

_example_search_size = 20

BoardState = TypeVar('BoardState')

class Trainer(Generic[BoardState]):
    def __init__(self,
                 model: Model[BoardState],
                 num_iterations: int,
                 iteration_size: int,
                 train_opts: mcts.Opts = mcts.Opts(temp_fn=lambda _: 1.0),
                 compare_opts: mcts.Opts = mcts._default_opts) -> None:
        self.model = model
        self.game = self.model.game
        self.num_iterations = num_iterations
        self.iteration_size = iteration_size
        self.train_opts = train_opts
        self.compare_opts = compare_opts
        self.example_opts = mcts.Opts(search_size=_example_search_size)

    def train_player(self):
        for i in range(1, self.num_iterations + 1):
            print('starting step %d' % i)
            self.model.add_data(self.play_games())
            print('finished self play, training on game data')
            self.model.train()

            print('finished training, evaluating new checkpoint: ')
            if self.train_is_better():
                print('improved model' % i)
                self.model.new_checkpoint()
            else:
                print('keeping checkpoint' %i)
                self.model.restore_checkpoint()

            print('finished step %d' % i)

        return example_games

    def train_is_better(self):
        tot_games = 10
        best_alg = mcts.Algorithm(self.game, self.model.best_evaluator, self.compare_opts)
        train_alg = mcts.Algorithm(self.game, self.model.train_evaluator, self.compare_opts)

        results = play_games(tot_games, self.game, [({0}, best_alg), ({1}, train_alg)])[1]
        x_wins = sum([ (mcts.rewards_from_result(result)[1] + 1) // 2 for result in results ])
        results = play_games(tot_games, self.game, [({0}, train_alg), ({1}, best_alg)])[1]
        o_wins = sum([ (mcts.rewards_from_result(result)[0] + 1) // 2 for result in results ])

        win_pct = (x_wins + o_wins) / tot_games
        print('win percent against previous best checkpoint: %d', int(win_pct * 100))
        return win_pct > 0.55


    def play_games(self) -> List[Tuple[State[BoardState], np.ndarray, np.ndarray]]:
        return mcts.play_selfs(self.iteration_size, self.game, self.model.best_evaluator, self.train_opts)
