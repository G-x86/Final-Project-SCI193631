import sys

from pacman_module.game import Agent, Directions

# Defensive guard against deep recursion on longer games.
sys.setrecursionlimit(10000)

FOOD_WEIGHT = 2
FOOD_REMAINING_PENALTY = 20
GHOST_DANGER_THRESHOLD = 3
GHOST_CLOSE_PENALTY = 500


def _state_key(state, agent_index, depth):
    """Return a hashable key that uniquely identifies a search node.

    Two nodes are considered identical when Pacman, all ghosts, the
    remaining food, the agent whose turn it is, and the remaining depth
    are the same. Caching on this key avoids re-expanding identical sub-trees.
    """
    return (
        state.getPacmanPosition(),
        tuple(state.getGhostPositions()),
        hash(state.getFood()),
        agent_index,
        depth,
    )


class PacmanAgent(Agent):
    """Pacman agent based on the Minimax algorithm."""

    def __init__(self, depth=None):
        super().__init__()
        self.depth = depth
        # Transposition table: maps state key -> (value, action).
        # Reset at the start of every real decision.
        self._cache = {}

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        self._cache = {}
        if self.depth is not None:
            max_depth = self.depth
        else:
            w = state.data.layout.width
            h = state.data.layout.height
            if w <= 7 and h <= 6:
                max_depth = 6
            else:
                max_depth = 3

        _, action = self.minimax(state, 0, max_depth)
        if action == Directions.STOP:
            legal = [
                a for a in state.getLegalPacmanActions()
                if a != Directions.STOP
            ]
            if legal:
                return legal[0]
        return action

    def minimax(self, state, agent_index, depth=3):
        """Recursively computes the minimax value of a state.

        Pacman (agent_index == 0) is the maximizing player, and every
        ghost (agent_index > 0) is a minimizing player. The recursion
        stops on terminal states (win/lose), when depth is 0, or when
        an agent has no legal move left.

        Arguments:
            state: the current game state.
            agent_index: index of the agent to play in `state`
                (0 for Pacman, > 0 for a ghost).
            depth: remaining search depth (in full rounds).

        Returns:
            A tuple `(value, action)` where `value` is the minimax
            value of `state` for the agent playing, and `action` is
            the move leading to that value.
        """
        if state.isWin():
            return state.getScore() + 1000 + depth, Directions.STOP
        if state.isLose():
            return -1000 - depth, Directions.STOP

        if depth == 0:
            return self.evaluate(state), Directions.STOP

        key = _state_key(state, agent_index, depth)
        if key in self._cache:
            return self._cache[key]

        num_agents = state.getNumAgents()
        next_index = (agent_index + 1) % num_agents
        next_depth = depth - 1 if next_index == 0 else depth

        if agent_index == 0:
            successors = state.generatePacmanSuccessors()
        else:
            successors = state.generateGhostSuccessors(agent_index)

        if not successors:
            result = self.evaluate(state), Directions.STOP
            self._cache[key] = result
            return result

        values = [
            (self.minimax(successor, next_index, next_depth)[0], action)
            for successor, action in successors
        ]

        if agent_index == 0:
            result = max(values, key=lambda pair: pair[0])
        else:
            result = min(values, key=lambda pair: pair[0])

        self._cache[key] = result
        return result

    def evaluate(self, state):
        """Heuristic evaluation for non-terminal leaf states."""
        score = state.getScore()
        pacman_x, pacman_y = state.getPacmanPosition()
        food = state.getFood()

        closest_food = min(
            (
                abs(pacman_x - x) + abs(pacman_y - y)
                for x in range(food.width)
                for y in range(food.height)
                if food[x][y]
            ),
            default=0,
        )

        ghost_penalty = 0
        ghost_positions = state.getGhostPositions()
        if ghost_positions:
            closest_ghost = min(
                abs(pacman_x - gx) + abs(pacman_y - gy)
                for gx, gy in ghost_positions
            )
            if closest_ghost <= GHOST_DANGER_THRESHOLD:
                ghost_penalty = GHOST_CLOSE_PENALTY

        food_remaining = state.getNumFood()
        return (
            score
            - FOOD_WEIGHT * closest_food
            - FOOD_REMAINING_PENALTY * food_remaining
            - ghost_penalty
        )
