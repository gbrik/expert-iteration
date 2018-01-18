import numpy as np
import tensorflow as tf
import os
from .utils import *
from .game import *
from . import mcts
from . import model

class TTTState(State[np.ndarray]):
    def __init__(self, board_state: np.ndarray, player: Player, prev_action: Action) -> None:
        super().__init__(board_state, player, prev_action)
        self._hash = hash_arr(board_state)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return isinstance(other, TTTState) and np.all(self.board_state == other.board_state)

class _TTT(Game[np.ndarray]):
    _board_shape = (9,)
    num_actions = 9
    num_players = 2

    def __init__(self) -> None:
        super().__init__()

    def gen_root(self) -> State[np.ndarray]:
        return TTTState(np.zeros(_TTT._board_shape, dtype=np.int), cast(Player, 1), None)

    def do_action(self, state: State[np.ndarray], action: Action) -> State[np.ndarray]:
        new_board = np.copy(state.board_state)
        new_board[action] = -1 + 2 * state.player
        return TTTState(new_board, cast(Player, 1 - state.player), action)

    def valid_actions(self, state: State[np.ndarray]) -> np.ndarray:
        return state.board_state == 0

    _end_idxs: List[int] = sum([[3 * i, 3 * i + 1, 3 * i + 2, i, i + 3, i + 6 ] for i in range(3)], [])
    _end_idx = np.array(_end_idxs + [2, 4, 6, 0, 4, 8])
    _inv_end_idx = np.array(cast(List[int], sum([[3 * i, i] for i in range(3)], [])) + [2, 0])
    def check_end(self, state: State[np.ndarray]) -> np.ndarray:
        ret = np.zeros(2, dtype=np.bool)
        z = np.flatnonzero(np.abs(np.sum(state.board_state[_TTT._end_idx].reshape(8, 3), axis=1)) == 3)
        if z.size:
            ret[(state.board_state[_TTT._inv_end_idx[z[0]]] + 1) // 2] = True
        elif np.all(state.board_state):
            ret[:] = True
        return ret

    _ch = {-1: 'o', 0: ' ', 1: 'x'}
    def render(self, state: State[np.ndarray]):
        for i in range(3):
            if i != 0:
                print('-----')
            for j in range(3):
                if j != 0:
                    print('|', end='')
                print(_TTT._ch[state.board_state[3 * i + j]], end='')
            print('')
        print('')

    def parse(self, s: str) -> Action:
        try:
            x, y = s.split()
            res = 3 * (int(x) - 1) + int(y) - 1
        except:
            raise ValueError
        return cast(Action, res)

game = _TTT()

def _ttt_eval_state(s: State[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    acts = np.zeros(game._board_shape)
    valid_acts = game.valid_actions(s)
    acts[valid_acts] = 1.0 / np.count_nonzero(valid_acts)
    return (acts, np.zeros(2))

rand_evaluator = mcts.Evaluator[np.ndarray](eval_state=_ttt_eval_state)


class Model(model.Supervised[np.ndarray]):
    l2_loss_coeff = 0.01
    hidden_size = 100

    graph = tf.Graph()
    with graph.as_default():
        ttt_hw = tf.Variable(tf.truncated_normal([9, hidden_size]))
        ttt_hb = tf.Variable(tf.zeros([hidden_size]))

        ttt_aw = tf.Variable(tf.truncated_normal([hidden_size, 9]))
        ttt_ab = tf.Variable(tf.zeros([9]))

        ttt_vw = tf.Variable(tf.truncated_normal([hidden_size, 2]))
        ttt_vb = tf.Variable(tf.zeros([2]))

        weights = [ttt_hw, ttt_hb, ttt_aw, ttt_ab, ttt_vw, ttt_vb]
        saver = tf.train.Saver(weights)

        tf_boards = tf.placeholder(tf.float32, shape=[None, 9])
        hidden = tf.nn.relu(tf.matmul(tf_boards, ttt_hw) + ttt_hb)
        actions = tf.matmul(hidden, ttt_aw) + ttt_ab
        values = tf.tanh(tf.matmul(hidden, ttt_vw) + ttt_vb)

        tf_probs = tf.placeholder(tf.float32, shape=[None, 9])
        tf_rewards = tf.placeholder(tf.float32, shape=[None, 2])

        loss = tf.reduce_mean(tf.square(tf_rewards - values))
        loss = loss + tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_probs, logits=actions))
        for w in weights:
            loss = loss + l2_loss_coeff * tf.nn.l2_loss(w)

        optimizer = tf.train.AdamOptimizer().minimize(loss)

    chkpnt_folder = os.path.join('.', 'checkpoints')
    if not os.path.exists(chkpnt_folder):
        os.makedirs(chkpnt_folder)
    chkpnt_file = os.path.join(chkpnt_folder, 'ttt')


    def __init__(self, opts: model.SupervisedOpts = model._default_supervised_opts) -> None:
        super().__init__(game, opts)

        self.data = {
            self.tf_boards: np.empty((0, 9)),
            self.tf_probs: np.empty((0, 9)),
            self.tf_rewards: np.empty((0, 2))
        }

        self.train_step = 0
        self.train_sess = None
        self.best_sess: tf.Session = None
        self.best_chkpnt: tf.Session = None

    def __enter__(self):
        self.train_sess = tf.Session(graph=self.graph)
        self.best_sess = tf.Session(graph=self.graph)
        self.train_sess.__enter__()
        self.best_sess.__enter__()

        self.train_sess.run(tf.global_variables_initializer())
        self.best_chkpnt = self.saver.save(self.train_sess, self.chkpnt_file, 0)
        self.saver.restore(self.best_sess, self.best_chkpnt)

        return self

    def __exit__(self, tp, val, traceback):
        self.best_sess.__exit__(tp, val, traceback)
        self.train_sess.__exit__(tp, val, traceback)

    def eval_states(self,
                    states: List[State[np.ndarray]],
                    using=model.Model.PARAMS_BEST) -> Tuple[np.ndarray, np.ndarray]:
        feed = { self.tf_boards: self._parse_states(states) }
        sess = self.best_sess if using == Model.PARAMS_BEST else self.train_sess
        actions, values = sess.run([self.actions, self.values], feed_dict=feed)
        for i in range(actions.shape[0]):
            action = actions[i]
            poss_actions = self.game.valid_actions(states[i])
            action[~poss_actions] = 0.0
            action[poss_actions] = softmax(action[poss_actions])

        return (actions, values)

    def train(self):
        self.train_step += 1

        for i in range(self.opts.train_iters):
            self.train_sess.run(self.optimizer, feed_dict=self._gen_batch())

    def new_checkpoint(self):
        self.best_chkpnt = self.saver.save(self.train_sess, self.chkpnt_file, self.train_step)
        self.saver.restore(self.best_sess, self.best_chkpnt)

    def restore_checkpoint(self):
        self.saver.restore(self.train_sess, self.best_chkpnt)

    def _parse_states(self, states: List[State[np.ndarray]]) -> np.ndarray:
        return np.array([ state.board_state for state in states ])

    def _parse_data(self, data: List[List[Tuple[State[np.ndarray], np.ndarray, np.ndarray]]]) -> Tuple[Dict[Any, np.ndarray], int]:
        states, probs, rewards = unzip(sum(data, []))
        parsed = {
            self.tf_boards: self._parse_states(states),
            self.tf_probs: np.array(probs),
            self.tf_rewards: np.array(rewards)
        }
        return (parsed, len(states))
