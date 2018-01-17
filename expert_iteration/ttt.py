import numpy as np
import tensorflow as tf
import os
from .utils import *
from .game import *
from . import mcts
from . import model

class _TTT(Game[np.ndarray]):
    _board_shape = (9,)
    num_actions = 9
    num_players = 2

    def __init__(self) -> None:
        super().__init__()

    def gen_root(self) -> State[np.ndarray]:
        return State(np.zeros(_TTT._board_shape, dtype=np.int), cast(Player, 1), None)

    def do_action(self, state: State[np.ndarray], action: Action) -> State[np.ndarray]:
        new_board = np.copy(state.board_state)
        new_board[action] = -1 + 2 * state.player
        return State(new_board, cast(Player, 1 - state.player), action)

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
        print('%s\' turn' % ('x' if state.player == 1 else 'o'))

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


class Model(model.Model[np.ndarray]):
    l2_loss_coeff = 0.01
    hidden_size = 100
    search_size = 100

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
        tf_zs = tf.placeholder(tf.float32, shape=[None, 2])

        loss = tf.reduce_mean(tf.square(tf_zs - values))
        loss = loss + tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_probs, logits=actions))
        for w in weights:
            loss = loss + l2_loss_coeff * tf.nn.l2_loss(w)

        optimizer = tf.train.AdamOptimizer().minimize(loss)

    chkpnt_folder = os.path.join('.', 'checkpoints')
    if not os.path.exists(chkpnt_folder):
        os.makedirs(chkpnt_folder)
    chkpnt_file = os.path.join(chkpnt_folder, 'ttt')


    def __init__(self):
        super().__init__(game)

        self.states = np.empty((0, 9))
        self.probs = np.empty((0, 9))
        self.rewards = np.empty((0, 2))
        self.players = np.empty((0,), dtype=np.int)
        self.cutoffs = []

        self.train_step = 0
        self.train_sess = None
        self.best_sess = None
        self.best_chkpnt = None

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
        feed = { self.tf_boards: np.array([state.board_state for state in states]) }
        sess = self.best_sess if using == Model.PARAMS_BEST else self.train_sess
        actions, values = sess.run([self.actions, self.values], feed_dict=feed)
        for i in range(actions.shape[0]):
            action = actions[i]
            poss_actions = self.game.valid_actions(states[i])
            action[~poss_actions] = 0.0
            action[poss_actions] = softmax(action[poss_actions])

        return (actions, values)

    def train(self, batch_size=64, num_iters=1000):
        self.train_step += 1

        for i in range(1000):
            select = np.random.choice(np.arange(len(self.states)), 64)

            feed = {
                self.tf_boards: self.states[select],
                self.tf_probs: self.probs[select],
                self.tf_zs: self.rewards[select]
            }

            self.train_sess.run(self.optimizer, feed_dict=feed)

        if self.train_is_better():
            self.best_chkpnt = self.saver.save(self.train_sess, self.chkpnt_file, self.train_step)
            self.saver.restore(self.best_sess, self.best_chkpnt)
        else:
            self.saver.restore(self.train_sess, self.best_chkpnt)

    def train_is_better(self):
        tot_games = 10
        reward = 0
        best_alg = mcts.Algorithm(self.game, self.best_evaluator, self.search_size)
        train_alg = mcts.Algorithm(self.game, self.train_evaluator, self.search_size)
        for i in range(tot_games):
            reward += mcts.rewards_from_result(play_game(self.game, [({0}, best_alg), ({1}, train_alg)])[1])[1]
        print(reward)
        for i in range(tot_games):
            reward += mcts.rewards_from_result(play_game(self.game, [({0}, train_alg), ({1}, best_alg)])[1])[0]

        avg_reward = reward / (2 * tot_games)
        print(avg_reward)
        return avg_reward > 0.1


    def add_data(self, data: List[Tuple[State[np.ndarray], np.ndarray, np.ndarray]]):
        states, probs, rewards = unzip(data)
        board_states, players = unzip([(state.board_state, state.player) for state in states])
        start_i = 0
        if len(self.cutoffs) > 3:
            start_i = self.cutoffs[0]
            self.cutoffs = self.cutoffs[1:]
        self.states = np.concatenate([self.states[start_i:], board_states])
        self.probs = np.concatenate([self.probs[start_i:], probs])
        self.rewards = np.concatenate([self.rewards[start_i:], rewards])
        self.players = np.concatenate([self.players[start_i:], players])
        self.cutoffs.append(len(players))
