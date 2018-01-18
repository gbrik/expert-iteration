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
                 iteration_size: int = 100,
                 search_size: int = 100) -> None:
        self.model = model
        self.game = self.model.game
        self.num_iterations = num_iterations
        self.iteration_size = iteration_size
        self.search_size = search_size

    def train_player(self):
        play_example_game = lambda: mcts.play_self(self.game, self.model.train_evaluator, _example_search_size)
        example_games = [play_example_game()]

        for i in range(1, self.num_iterations + 1):
            example_games.append(play_example_game())
            self.model.add_data(self.play_games())
            self.model.train()

            if self.train_is_better():
                self.model.new_checkpoint()
            else:
                self.model.restore_checkpoint()

            print('finished step %d' % i)

        example_games.append(play_example_game())

        return example_games

    def train_is_better(self):
        tot_games = 10
        reward = 0
        best_alg = mcts.Algorithm(self.game, self.model.best_evaluator, self.search_size)
        train_alg = mcts.Algorithm(self.game, self.model.train_evaluator, self.search_size)

        results = play_games(tot_games, self.game, [({0}, best_alg), ({1}, train_alg)])[1]
        reward += sum([ mcts.rewards_from_result(result)[1] for result in results ])
        results = play_games(tot_games, self.game, [({0}, train_alg), ({1}, best_alg)])[1]
        reward += sum([ mcts.rewards_from_result(result)[0] for result in results ])

        avg_reward = reward / (2 * tot_games)
        print(avg_reward)
        return avg_reward > 0.1


    def play_games(self) -> List[Tuple[State[BoardState], np.ndarray, np.ndarray]]:
        result = mcts.play_selfs(self.iteration_size, self.game, self.model.best_evaluator, self.search_size)
        return sum(result, [])
