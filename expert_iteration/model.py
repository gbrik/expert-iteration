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

    def train(self):
        raise NotImplemented

    def add_data(self, data: List[Tuple[State[BoardState], np.ndarray, np.ndarray]]):
        raise NotImplemented

    def new_checkpoint(self):
        raise NotImplemented

    def restore_checkpoint(self):
        raise NotImplemented

class SupervisedOpts:
    def __init__(self,
                 batch_size: int = 64,
                 train_iters: int = 1000,
                 history_size: int = 10) -> None:
        self.batch_size = 64
        self.train_iters = 1000
        self.history_size = history_size

_default_supervised_opts = SupervisedOpts()

class Supervised(Model[BoardState]):
    def __init__(self, game: Game[BoardState], opts: SupervisedOpts = _default_supervised_opts) -> None:
        super().__init__(game)
        self.opts = opts
        self.cutoffs: List[int] = []
        self.data: Dict[Any, np.ndarray] = dict()
        self._num_data: int = 0

    def _parse_data(self, data: List[Tuple[State[BoardState], np.ndarray, np.ndarray]]) -> Dict[Any, np.ndarray]:
        raise NotImplemented

    def _gen_batch(self):
        select = np.random.choice(np.arange(self._num_data),
                                  min(self._num_data, self.opts.batch_size),
                                  replace=False)

        return { key: val[select] for key, val in self.data.items() }

    def add_data(self, data: List[Tuple[State[np.ndarray], np.ndarray, np.ndarray]]):
        parsed = self._parse_data(data)
        start_i = 0
        self._num_data += len(data)
        if len(self.cutoffs) > self.opts.history_size:
            start_i = self.cutoffs[0]
            self._num_data -= start_i
            self.cutoffs = self.cutoffs[1:] + [ len(data) ]
        self.data = { key: np.concatenate([val[start_i:], parsed[key]]) for key, val in self.data.items() }
