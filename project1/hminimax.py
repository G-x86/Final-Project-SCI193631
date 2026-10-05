import sys
from collections import defaultdict

from pacman_module.game import Agent, Directions

# Defensive guard, same reasoning as in minimax.py. H-Minimax is
# depth-limited so this is less likely to be hit, but costs nothing.
sys.setrecursionlimit(10000)

# Used to penalize immediately reversing direction (a 180-degree
# turn), which otherwise causes Pacman to oscillate forever between
# two equally good looking positions.
OPPOSITE_DIRECTION = {
    Directions.NORTH: Directions.SOUTH,
    Directions.SOUTH: Directions.NORTH,
    Directions.EAST: Directions.WEST,
    Directions.WEST: Directions.EAST,
    Directions.STOP: Directions.STOP,
}


class PacmanAgent(Agent):
    """Pacman agent based on the H-Minimax algorithm.

    H-Minimax is a depth-limited version of Minimax: once a fixed
    depth is reached (counted in full rounds, i.e. one Pacman move
    plus one move per ghost), the state is scored with a heuristic
    instead of being expanded further.
    """

    def __init__(self, depth=3):
        super().__init__()

        # Number of full rounds (Pacman + all ghosts) explored before
        # falling back to the heuristic. Tune this to trade off
        # score quality against the number of expanded nodes.
        self.depth = depth

        # Counts how many times Pacman has actually occupied each
        # position during the real game (not simulated search nodes).
        # Helps discourage repeatedly camping on the same spot.
        self.visit_count = defaultdict(int)

        # The action actually taken on the previous real move. Used
        # to break ties that would otherwise cause Pacman to reverse
        # direction forever (e.g. walking north then south, repeat).
        self.last_action = Directions.STOP

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        This top-level decision is handled separately from the rest
        of the recursion so that a small anti-reversal tie-break can
        be applied to the real move about to be taken, without
        affecting the deeper, purely adversarial search.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        self.visit_count[state.getPacmanPosition()] += 1

        num_agents = state.getNumAgents()
        next_index = 1 % num_agents
        next_depth = self.depth if next_index != 0 else self.depth - 1

        successors = state.generatePacmanSuccessors()
        successors = sorted(
            successors,
            key=lambda pair: self.evaluate(pair[0]),
            reverse=True)

        best_value, best_action = float("-inf"), Directions.STOP
        reversal = OPPOSITE_DIRECTION[self.last_action]
        alpha = float("-inf")
        beta = float("inf")

        for successor, action in successors:
            value, _ = self.hminimax(
                successor, next_index, next_depth, alpha, beta)

            # Only breaks exact ties (difference of 0): never strong
            # enough to override a real, meaningful preference.
            if action == reversal:
                value -= 0.5

            if value > best_value:
                best_value, best_action = value, action
            alpha = max(alpha, best_value)

        self.last_action = best_action
        return best_action

    def hminimax(self, state, agent_index, depth,
                 alpha=float("-inf"), beta=float("inf")):
        """Recursively computes the H-Minimax value of a state.

        Alpha-beta pruning skips branches that cannot change the
        result, so the returned value is exactly the plain
        H-Minimax value with fewer expanded nodes. Successors are
        ordered by the heuristic so pruning happens early.

        Arguments:
            state: the current game state.
            agent_index: index of the agent to play in `state`
                (0 for Pacman, > 0 for a ghost).
            depth: remaining number of full rounds to explore before
                using the heuristic.
            alpha: best value the maximizer can guarantee so far.
            beta: best value the minimizer can guarantee so far.

        Returns:
            A tuple `(value, action)` where `value` is the H-Minimax
            value of `state` for the agent playing, and `action` is
            the move leading to that value.
        """

        if state.isWin() or state.isLose():
            return state.getScore(), Directions.STOP

        if depth == 0:
            return self.evaluate(state), Directions.STOP

        num_agents = state.getNumAgents()
        next_index = (agent_index + 1) % num_agents
        # A full round has been completed once we get back to Pacman.
        next_depth = depth - 1 if next_index == 0 else depth

        if agent_index == 0:
            successors = state.generatePacmanSuccessors()
        else:
            successors = state.generateGhostSuccessors(agent_index)

        if not successors:
            return self.evaluate(state), Directions.STOP

        if agent_index == 0:
            successors = sorted(
                successors,
                key=lambda pair: self.evaluate(pair[0]),
                reverse=True)
            best_value = float("-inf")
            best_action = Directions.STOP
            for successor, action in successors:
                value, _ = self.hminimax(
                    successor, next_index, next_depth, alpha, beta)
                if value > best_value:
                    best_value, best_action = value, action
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
            return best_value, best_action
        else:
            successors = sorted(
                successors,
                key=lambda pair: self.evaluate(pair[0]))
            best_value = float("inf")
            best_action = Directions.STOP
            for successor, action in successors:
                value, _ = self.hminimax(
                    successor, next_index, next_depth, alpha, beta)
                if value < best_value:
                    best_value, best_action = value, action
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            return best_value, best_action

    def evaluate(self, state):
        """Heuristic evaluation of a non-terminal state.

        Combines the current game score with:
        - a penalty based on the distance to the closest food dot,
          so that Pacman is guided towards food even before
          actually eating it;
        - a small penalty proportional to how often Pacman has
          actually been at this exact position already, discouraging
          camping on or repeatedly circling back to the same spot.

        Arguments:
            state: the game state to evaluate.

        Returns:
            A numeric estimate of how good `state` is for Pacman.
        """

        score = state.getScore()
        pacman_pos = state.getPacmanPosition()
        pacman_x, pacman_y = pacman_pos
        food = state.getFood()

        closest_food_distance = min(
            (
                abs(pacman_x - x) + abs(pacman_y - y)
                for x in range(food.width)
                for y in range(food.height)
                if food[x][y]
            ),
            default=0,
        )

        visit_penalty = self.visit_count[pacman_pos] * 5

        return score - closest_food_distance - visit_penalty
