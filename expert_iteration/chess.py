import numpy as np
import os
import chess
from keras import layers, regularizers
import keras.models
from .utils import *
from .game import *
from . import mcts, model

_action_shape = (64, 64, 2)

def _action_from_move(move: chess.Move) -> Action:
    return np.ravel_multi_index((move.from_square, move.to_square, move.promotion != None), _action_shape)

def _move_from_action(action: Action) -> chess.Move:
    from_square, to_square, promotion = np.unravel_index(action, _action_shape)
    return chess.Move(from_square, to_square, promotion = chess.QUEEN if promotion else None)

class _State(State[chess.Board]):
    def __init__(self, board: chess.Board):
        super().__init__(board,
                         cast(Player, int(board.turn)),
                         _action_from_move(board.peek()) if len(board.move_stack) > 0 else None,
                         board.fullmove_number * 2 - 1 - int(board.turn))
        self._hash = hash(board.fen())

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return isinstance(other, TTTState) and self.board_state.fen() == other.board_state.fen()

class _Chess(Game[chess.Board]):
    num_players = 2
    num_actions = np.product(_action_shape)

    def __init__(self) -> None:
        super().__init__()

    def gen_root(self) -> State[chess.Board]:
        return _State(chess.Board())

    def do_action(self, state: State[chess.Board], action: Action) -> State[chess.Board]:
        new_board = state.board_state.copy()
        new_board.push(_move_from_action(action))
        return _State(new_board)

    def valid_actions(self, state: State[chess.Board]) -> np.ndarray:
        ret = np.zeros(self.num_actions, dtype=np.bool)
        for move in state.board_state.legal_moves:
            ret[_action_from_move(move)] = True
        return ret

    def check_end(self, state: State[chess.Board]) -> np.ndarray:
        if state.board_state.is_checkmate():
            return np.arange(2) != state.player
        elif state.board_state.is_stalemate() or state.turn_num >= 10:
            return np.array([True, True])
        else:
            return np.array([False, False])

        result = state.board_state.result(claim_draw=True)
        if result == '1-0':
            return np.array([False, True])
        elif result == '0-1':
            return np.array([True, False])
        elif result == '1/2-1/2' or state.turn_num >= 50:
            return np.array([True, True])
        else:
            return np.array([False, False])

    def render(self, state: State[chess.Board]):
        print(state.board_state)

    def parse(self, s: str, state: State[chess.Board]) -> Action:
        legal = list(state.board_state.legal_moves)
        if s in [ move.uci() for move in legal ]:
            return _action_from_move(chess.Move.from_uci(s))
        else:
            raise ValueError

game = _Chess()

_num_piece_types = 6
_board_dim = 8
_input_filters = 2 * _num_piece_types + 1
_input_shape = (_board_dim, _board_dim, _input_filters)
_regularizer = regularizers.l2(0.001)
_depth = 64
_conv_layer = lambda: layers.Conv2D(_depth, 3, padding='same', kernel_regularizer=_regularizer)
_norm_layer = lambda: layers.BatchNormalization()
_activation_layer = lambda: layers.Activation('relu')
_residual_tower_height = 5

def _keras_model():
    board = layers.Input(shape=_input_shape)
    conv = _conv_layer()(board)
    norm = _norm_layer()(conv)
    bottom = _activation_layer()(norm)
    for i in range(_residual_tower_height):
        conv = _conv_layer()(bottom)
        norm = _norm_layer()(conv)
        mid = _activation_layer()(norm)
        conv = _conv_layer()(mid)
        norm = _norm_layer()(conv)
        add = layers.Add()([bottom, norm])
        bottom = _activation_layer()(add)

    policy = layers.Conv2D(2, 1, padding='same', kernel_regularizer=_regularizer)(bottom)
    policy = _norm_layer()(policy)
    policy = _activation_layer()(policy)
    policy = layers.Flatten()(policy)
    policy = layers.Dense(game.num_actions, kernel_regularizer=_regularizer)(policy)

    value = layers.Conv2D(1, 1, padding='same', kernel_regularizer=_regularizer)(bottom)
    value = _norm_layer()(value)
    value = _activation_layer()(value)
    value = layers.Flatten()(value)
    value = layers.Dense(_depth, kernel_regularizer=_regularizer)(value)
    value = _activation_layer()(value)
    value = layers.Dense(1, activation='tanh')(value)

    model = keras.models.Model(inputs=board, outputs=[policy, value])
    model.compile(optimizer='adagrad',
                  loss=['categorical_crossentropy', 'mean_squared_error'],
                  loss_weights=[1.0, 1.0])

    return model

_boards = 'boards'
_probs = 'probs'
_rewards = 'rewards'


class Model(model.Supervised[np.ndarray]):
    def __init__(self, opts: model.SupervisedOpts = model._default_supervised_opts) -> None:
        super().__init__(game, opts=opts)
        checkpoint_folder = os.path.join('.', 'checkpoints')
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        self.checkpoint_file = os.path.join(checkpoint_folder, 'chess-weights')

        self.data = {
            _boards: np.empty((0,) + _input_shape),
            _probs: np.empty((0, game.num_actions)),
            _rewards: np.empty((0,))
        }

        self.best_model = _keras_model()
        self.train_model = _keras_model()
        self.train_model.save_weights(self.checkpoint_file)
        self.best_model.load_weights(self.checkpoint_file)

    def eval_states(self,
                    states: List[State[np.ndarray]],
                    using=model.Model.PARAMS_BEST) -> Tuple[np.ndarray, np.ndarray]:

        model = self.best_model if using == Model.PARAMS_BEST else self.train_model
        actions, values = model.predict(self._parse_states(states), batch_size=self.opts.batch_size)
        poss_actions = self.game.valid_actionses(states)
        actions = softmax(actions, poss_actions)

        return (actions, values)

    def train(self):
        self.train_model.fit(x=self.data[_boards],
                             y=[self.data[_probs], self.data[_rewards]],
                             batch_size=self.opts.batch_size)

    def new_checkpoint(self):
        self.train_model.save_weights(self.checkpoint_file)
        self.best_model.load_weights(self.checkpoint_file)

    def restore_checkpoint(self):
        self.train_model.load_weights(self.checkpoint_file)

    def _parse_states(self, states: List[State[np.ndarray]]) -> np.ndarray:
        piece_types = range(1, _num_piece_types + 1)
        colors = [False, True]
        boards = np.zeros((len(states),) + _input_shape, dtype=np.float)
        for i, state in enumerate(states):
            for square, piece in state.board_state.piece_map().items():
                x, y = np.unravel_index(square, (8, 8))
                val = piece.piece_type - 1 + 6 * int(piece.color)
                boards[i, x, y, val] = 1.0
            if state.board_state.turn == chess.WHITE:
                boards[i, :, :, -1] = 1.0

        return boards

    def _parse_data(self, data: List[List[Tuple[State[np.ndarray], np.ndarray, np.ndarray]]]) -> Tuple[Dict[Any, np.ndarray], int]:
        states, probs, rewards = unzip(sum(data, []))
        parsed = {
            _boards: self._parse_states(states),
            _probs: np.array(probs),
            _rewards: np.array([reward[1] for reward in rewards])
        }
        return (parsed, len(states))
