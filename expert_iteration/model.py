import numpy as np
from .utils import *
from .game import *
from . import mcts

class Model(Generic[BoardState]):
    def __init__(self, game: Game[BoardState]) -> None:
        self.game = game
        self.best_evaluator = mcts.Evaluator[BoardState](
            eval_state=lambda s: self.eval_state(s, using=Model.PARAMS_BEST),
            eval_states=lambda s: self.eval_states(s, using=Model.PARAMS_BEST))
        self.train_evaluator = mcts.Evaluator[BoardState](
            eval_state=lambda s: self.eval_state(s, using=Model.PARAMS_BEST),
            eval_states=lambda s: self.eval_states(s, using=Model.PARAMS_BEST))

    PARAMS_TRAIN = 'PARAMS_TRAIN'
    PARAMS_BEST = 'PARAMS_BEST'

    def eval_states(self, states: List[State[BoardState]], using=PARAMS_BEST) -> Tuple[np.ndarray, np.ndarray]:
        probs, values = unzip([self.eval_state(state, using) for state in states])
        return np.array(probs), np.array(values)

    def eval_state(self, state: State[BoardState], using=PARAMS_BEST) -> Tuple[np.ndarray, np.ndarray]:
        a, v = self.eval_states([state], using)
        return a[0], v[0]

    def train(self, batch_size=64, num_iters=1000):
        pass

    def add_data(self, data: List[Tuple[State[BoardState], np.ndarray, np.ndarray]]):
        pass
