from pacman_module.game import Agent, Directions


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

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        _, action = self.hminimax(state, 0, self.depth)
        return action

    def hminimax(self, state, agent_index, depth):
        """Recursively computes the H-Minimax value of a state.

        Arguments:
            state: the current game state.
            agent_index: index of the agent to play in `state`
                (0 for Pacman, > 0 for a ghost).
            depth: remaining number of full rounds to explore before
                using the heuristic.

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

        values = [
            (self.hminimax(successor, next_index, next_depth)[0], action)
            for successor, action in successors
        ]

        if agent_index == 0:
            return max(values, key=lambda pair: pair[0])
        else:
            return min(values, key=lambda pair: pair[0])

    def evaluate(self, state):
        """Heuristic evaluation of a non-terminal state.

        Combines the current game score with a penalty based on the
        distance to the closest food dot, so that Pacman is guided
        towards food even before actually eating it.

        Arguments:
            state: the game state to evaluate.

        Returns:
            A numeric estimate of how good `state` is for Pacman.
        """

        score = state.getScore()
        pacman_x, pacman_y = state.getPacmanPosition()
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

        return score - closest_food_distance
