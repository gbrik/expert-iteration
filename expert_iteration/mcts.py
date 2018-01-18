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

    def mk_player(self, players: Set[Player], num_games: int = 1) -> GamePlayer[BoardState]:
        return MCTSPlayer(self, players, num_games)

    def _do_searches(self, nodes: np.ndarray) -> np.ndarray:
        for _ in range(self.search_size):
            cur_nodes = np.copy(nodes)
            final_actions = np.empty(nodes.size, dtype=np.int)
            ongoing_idxs = np.arange(nodes.size)
            histories = np.empty(nodes.size, dtype=np.object)
            histories[:] = [ [] for _ in range(nodes.size) ]

            while ongoing_idxs.size > 0:
                cur_actions = np.array([ node.select_action() for node in cur_nodes[ongoing_idxs] ])
                for hist, node in zip(histories[ongoing_idxs], cur_nodes[ongoing_idxs]):
                    hist.append(node)
                cur_nodes[ongoing_idxs] = [ node.children[action]
                                            for node, action
                                            in zip(cur_nodes[ongoing_idxs], cur_actions) ]
                continue_search = np.array([ node != None and isinstance(node, InternalMCTSNode)
                                             for node in cur_nodes[ongoing_idxs] ])

                final_actions[ongoing_idxs[~continue_search]] = cur_actions[~continue_search]
                ongoing_idxs = ongoing_idxs[continue_search]

            rewardses = np.empty((nodes.size, self.game.num_players), dtype=np.float)
            existing_leaves = np.array([ isinstance(node, LeafMCTSNode) for node in cur_nodes ])
            if np.any(existing_leaves):
                rewardses[existing_leaves] = [ node.values for node in cur_nodes[existing_leaves] ]
            if np.any(~existing_leaves):
                idxs = np.arange(nodes.size)[~existing_leaves]
                prev_nodes = [ history[-1] for history in histories[idxs] ]
                new_states = np.array([ self.game.do_action(prev_node.state, final_action)
                                        for prev_node, final_action in zip(prev_nodes, final_actions[idxs]) ])
                results = np.array([ self.game.check_end(new_state) for new_state in new_states ])
                is_over = np.any(results, axis=1)
                if np.any(is_over):
                    over_idxs = idxs[is_over]
                    rewards = np.array([ rewards_from_result(result) for result in results[is_over] ])
                    rewardses[over_idxs] = rewards
                    cur_nodes[over_idxs] = [ LeafMCTSNode(new_state, reward)
                                             for new_state, reward in zip(new_states[is_over], rewards) ]
                if np.any(~is_over):
                    not_over_idxs = idxs[~is_over]
                    probses, rewards = self.evaluator.eval_states(new_states[~is_over])
                    rewardses[not_over_idxs] = rewards
                    cur_nodes[not_over_idxs] = [ InternalMCTSNode(new_state, probs)
                                                 for new_state, probs in zip(new_states[~is_over], probses) ]

                for prev_node, act, new_node in zip(prev_nodes, final_actions[idxs], cur_nodes[idxs]):
                    prev_node.children[act] = new_node

            for rewards, final_action, history in zip(rewardses, final_actions, histories):
                acts = [ node.state.prev_action for node in history[1:] ] + [ final_action ]

                for prev_node, act in zip(history, acts):
                    prev_node.backup(act, rewards)

        return np.array([ node.probs(self.temp) for node in nodes ])

    def _setup_nodes(self, states: np.ndarray, nodes: np.ndarray):
        existing = nodes != None
        if np.any(existing):
            nodes[existing] = [ cast(InternalMCTSNode[BoardState], node.children[state.prev_action])
                                for state, node in zip(states, nodes[existing])]

        non_existing = nodes == None
        if np.any(non_existing):
            probses, _ = self.evaluator.eval_states(states[non_existing])
            add_dirichlet = np.array([ state.prev_action == None for state in states[non_existing] ])
            if np.any(add_dirichlet):
                keep_zero = probses[add_dirichlet] == 0
                dirichlets = np.random.dirichlet(np.full(self.game.num_actions, dirichlet_alpha),
                                                 add_dirichlet.size)
                dirichlets[keep_zero] = 0.0
                rescale(dirichlets)
                probses[add_dirichlet] = dirichlets

            nodes[non_existing] = [ InternalMCTSNode(state, probs) for state, probs in zip(states[non_existing], probses) ]

        return nodes

class MCTSPlayer(GamePlayer[BoardState]):
    def __init__(self,
                 alg: Algorithm[BoardState],
                 players: Set[Player],
                 num_games: int) -> None:
        super().__init__(players, num_games)
        self.alg = alg
        self.nodes: np.ndarray = np.full(num_games, None, dtype=np.object)
        self.hists: np.ndarray = np.empty(num_games, dtype=np.object)
        self.hists[:] = [[] for _ in range(num_games)]

    def _take_turns(self, states: np.ndarray, game_idxs: np.ndarray) -> np.ndarray:
        self.nodes[game_idxs] = self.alg._setup_nodes(states, self.nodes[game_idxs])

        probses = self.alg._do_searches(self.nodes[game_idxs])
        for hist, state, probs in zip(self.hists[game_idxs], states, probses):
            hist.append((state, probs))
        return sample(probses)

    def _watch_turns(self, states: np.ndarray, game_idxs: np.ndarray):
        for state, game_idx in zip(states, game_idxs):
            self.nodes[game_idx], self.hists[game_idx] = (
                self._watch_turn_stateless(self.nodes[game_idx], self.hists[game_idx], state))


    def _watch_turn_stateless(self,
                              node: InternalMCTSNode[BoardState],
                              hist: List[Tuple[State[BoardState], np.ndarray]],
                              state: State[BoardState]):
        if node == None:
            return None, hist

        node = cast(InternalMCTSNode[BoardState], node.children[state.prev_action])
        return node, hist + [(state, node.probs(self.alg.temp))]


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
    hist = players[0].hists[0]
    rewards = rewards_from_result(result)

    return [(state, probs, rewards) for state, probs in hist]

def play_selfs(num_games: int,
               game: Game[BoardState],
               evaluator: Evaluator[BoardState],
               search_size: int,
               temp: float = 0.0) -> List[List[Tuple[State[BoardState], np.ndarray, np.ndarray]]]:
    turn_summaries: List[List[Tuple[State[BoardState], np.ndarray, np.ndarray]]] = []
    _, results, players = play_games(num_games, game, [(cast(Set[Player], {0, 1}), Algorithm(game, evaluator, search_size, temp))])

    for result, hist in zip(results, players[0].hists):
        rewards = rewards_from_result(result)
        turn_summaries.append([(state, probs, rewards) for state, probs in hist])

    return turn_summaries
