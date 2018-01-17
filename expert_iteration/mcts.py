from .utils import *
from .game import *
import numpy as np
import math

class Evaluator(Generic[BoardState]):
    def eval_state(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        res = self.eval_state(np.array([state]))
        return res[0], res[1]

    def eval_states(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs, vals = unzip([self.eval_state(state) for state in states])
        return np.array(probs), np.array(vals)

    def __init__(self, eval_state = None, eval_states = None) -> None:
        if eval_state != None:
            self.eval_state = eval_state #type: ignore
        if eval_states != None:
            self.eval_states = eval_states #type: ignore

c_puct = 1.0
dirichlet_eps = 0.25
dirichlet_alpha = 0.3

class MCTSNode(Generic[BoardState]):
    def __init__(self, state: State[BoardState]) -> None:
        self.state = state

    @property
    def terminal(self):
        raise NotImplemented

class InternalMCTSNode(MCTSNode[BoardState]):
    terminal = False

    def __init__(self, state: State[BoardState], probs: np.ndarray) -> None:
        super().__init__(state)

        self.actions = np.flatnonzero(probs)
        self.inv_actions = np.empty_like(probs, dtype=np.int)
        self.inv_actions[self.actions] = np.arange(self.actions.size)

        self.N: np.ndarray = np.zeros_like(self.actions, dtype=np.int)
        self.W: np.ndarray = np.zeros_like(self.actions, dtype=np.float)
        self.Q: np.ndarray = np.zeros_like(self.actions, dtype=np.float)
        self.P: np.ndarray = probs[self.actions]

        self.total_visits: int = 1
        self.children: List[MCTSNode[BoardState]] = [None] * probs.size

    def select_action(self) -> Action:
        if not self.actions.size:
            return None

        root_total = math.sqrt(self.total_visits)

        selection_priorities = self.Q + c_puct * root_total * self.P / (1 + self.N)
        return self.actions[np.argmax(selection_priorities)]

    def backup(self, a, vs):
        self.total_visits = self.total_visits + 1
        idx = self.inv_actions[a]
        self.N[idx] = self.N[idx] + 1
        self.W[idx] = self.W[idx] + vs[self.state.player]
        self.Q[idx] = self.W[idx] / self.N[idx]

    def probs(self, temp) -> np.ndarray:
        if temp == 0.0:
            ret = np.zeros_like(self.inv_actions)
            ret[self.actions[np.argmax(self.N)]] = 1.0
            return ret
        exps = self.N ** (1.0 / temp)
        ret = np.zeros_like(self.inv_actions, dtype=np.float)
        ret[self.actions] = exps / np.sum(exps)
        return ret

class LeafMCTSNode(MCTSNode[BoardState]):
    terminal = True

    def __init__(self, state: State[BoardState], values: np.ndarray) -> None:
        super().__init__(state)
        self.values = values

class Algorithm(GameAlgorithm[BoardState]):
    def __init__(self,
                 game: Game[BoardState],
                 evaluator: Evaluator[BoardState],
                 search_size: int,
                 temp: float = 0.0) -> None:
        self.game = game
        self.evaluator = evaluator
        self.search_size = search_size
        self.temp = temp

    def mk_player(self, players: Set[Player]) -> GamePlayer[BoardState]:
        return MCTSPlayer(self, players)

    def _do_search(self, node: InternalMCTSNode[BoardState]):
        for _ in range(self.search_size):
            cur_node: MCTSNode[BoardState] = node
            cur_action = None
            history: List[InternalMCTSNode[BoardState]] = []

            while cur_node and isinstance(cur_node, InternalMCTSNode):
                cur_action = cur_node.select_action()
                history.append(cur_node)
                cur_node = cur_node.children[cur_action]

            rewards: float
            if isinstance(cur_node, LeafMCTSNode):
                rewards = cur_node.values
            else:
                prev_node = history[-1]
                new_state = self.game.do_action(prev_node.state, cur_action)
                result = self.game.check_end(new_state)
                if np.any(result):
                    rewards = rewards_from_result(result)
                    cur_node = LeafMCTSNode(new_state, rewards)
                else:
                    probs, rewards = self.evaluator.eval_state(new_state)
                    cur_node = InternalMCTSNode(new_state, probs)

                prev_node.children[cur_action] = cur_node

            acts = [ node.state.prev_action for node in history[1:] ] + [ cur_action ]

            for prev_node, act in zip(history, acts):
                prev_node.backup(act, rewards)

        return node.probs(self.temp)

class MCTSPlayer(GamePlayer[BoardState]):
    def __init__(self,
                 alg: Algorithm[BoardState],
                 players: Set[Player]) -> None:
        super().__init__(players)
        self.alg = alg
        self.node: InternalMCTSNode[BoardState] = None
        self.hist: List[Tuple[State[BoardState], np.ndarray]] = []

    def _take_turn(self, state: State[BoardState]) -> Action:
        if self.node != None:
            self.node = cast(InternalMCTSNode[BoardState], self.node.children[state.prev_action])
        if self.node == None:
            probs, _ = self.alg.evaluator.eval_state(state)
            if state.prev_action == None:
                mask = probs != 0
                probs[mask] = ((1 - dirichlet_eps) * probs[mask] +
                         dirichlet_eps * np.random.dirichlet(np.full_like(probs[mask], dirichlet_alpha)))
            self.node = InternalMCTSNode(state, probs)

        if self.node.terminal:
            print(state.board_state, self.node.state.board_state)
            for state, probs in self.hist:
                print(state.board_state, state.prev_action, probs)
        probs = self.alg._do_search(self.node)
        self.hist.append((state, probs))
        act = sample(probs)
        if act == None or state.board_state[act] != 0: #type: ignore
            print('_take_turn invalid action: ', act, state.board_state, self.node.state.board_state)
        return cast(Action, act)

    def _watch_turn(self, state: State[BoardState]):
        if self.node == None:
            return

        self.hist.append((state, self.node.probs(self.alg.temp)))
        self.node = cast(InternalMCTSNode[BoardState], self.node.children[state.prev_action])

def rewards_from_result(game_result: np.ndarray) -> np.ndarray:
    winners = np.count_nonzero(game_result)
    if winners == game_result.size:
        return np.zeros_like(game_result, dtype=np.float)
    losers = game_result.size - winners
    values = np.full_like(game_result, -1.0 / losers, dtype=np.float)
    values[game_result] = 1.0 / winners
    return values

def play_self(game: Game[BoardState],
              evaluator: Evaluator[BoardState],
              search_size: int,
              temp: float = 0.0) -> List[Tuple[State[BoardState], np.ndarray, np.ndarray]]:
    end_state, result, players = play_game(game, [(cast(Set[Player], {0, 1}), Algorithm(game, evaluator, search_size, temp))])
    hist = players[0].hist
    rewards = rewards_from_result(result)

    return [(state, probs, rewards) for state, probs in hist]

def play_selfs(num_games: int,
               game: Game[BoardState],
               evaluator: Evaluator[BoardState],
               search_size: int,
               temp: float = 0.0) -> List[List[Tuple[State[BoardState], np.ndarray, np.ndarray]]]:
    turn_summaries: List[List[Tuple[State[BoardState], np.ndarray, np.ndarray]]] = []
    _, results, playerses = play_games(num_games, game, [(cast(Set[Player], {0, 1}), Algorithm(game, evaluator, search_size, temp))])

    for result, players in zip(results, playerses):
        hist = players[0].hist
        rewards = rewards_from_result(result)

        turn_summaries.append([(state, probs, rewards) for state, probs in hist])

    return turn_summaries
