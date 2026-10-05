import sys

from pacman_module.game import Agent, Directions

# Defensive guard against deep recursion on longer games.
sys.setrecursionlimit(10000)


class PacmanAgent(Agent):
    """Pacman agent based on the Minimax algorithm."""

    def __init__(self, depth=3):
        super().__init__()
        self.depth = depth

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        max_depth = self.depth
        _, action = self.minimax(
            state, 0, max_depth,
            float("-inf"), float("inf"))
        if action == Directions.STOP:
            legal = [
                a for a in state.getLegalPacmanActions()
                if a != Directions.STOP
            ]
            if legal:
                return legal[0]
        return action

    def minimax(self, state, agent_index, depth=3,
                alpha=float("-inf"), beta=float("inf")):
        """Recursively computes the minimax value of a state.

        Pacman (agent_index == 0) is the maximizing player, and every
        ghost (agent_index > 0) is a minimizing player. The recursion
        stops on terminal states (win/lose), when depth is 0, or when
        an agent has no legal move left. Alpha-beta pruning skips
        branches that cannot change the result, so the returned
        value is exactly the plain minimax value with fewer nodes.

        Arguments:
            state: the current game state.
            agent_index: index of the agent to play in `state`
                (0 for Pacman, > 0 for a ghost).
            depth: remaining search depth (in full rounds).
            alpha: best value the maximizer can guarantee so far.
            beta: best value the minimizer can guarantee so far.

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

        num_agents = state.getNumAgents()
        next_index = (agent_index + 1) % num_agents
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
                value, _ = self.minimax(
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
                value, _ = self.minimax(
                    successor, next_index, next_depth, alpha, beta)
                if value < best_value:
                    best_value, best_action = value, action
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            return best_value, best_action

    def evaluate(self, state):
        """Heuristic evaluation for non-terminal leaf states.

        Guides Pacman towards the closest food dot on top of the
        current game score. Ghost avoidance is left to the search
        itself, which already models ghosts as minimizing players.
        """
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

        return score - closest_food
