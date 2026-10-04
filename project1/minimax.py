import sys

from pacman_module.game import Agent, Directions

# Full Minimax recurses until the game ends, which can exceed Python's
# default recursion limit on longer games. Raise it defensively.
sys.setrecursionlimit(10000)


class PacmanAgent(Agent):
    """Pacman agent based on the Minimax algorithm."""

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        _, action = self.minimax(state, 0)
        return action

    def minimax(self, state, agent_index):
        """Recursively computes the minimax value of a state.

        Pacman (agent_index == 0) is the maximizing player, and every
        ghost (agent_index > 0) is a minimizing player. The recursion
        stops on terminal states (win/lose) or when an agent has no
        legal move left.

        Arguments:
            state: the current game state.
            agent_index: index of the agent to play in `state`
                (0 for Pacman, > 0 for a ghost).

        Returns:
            A tuple `(value, action)` where `value` is the minimax
            value of `state` for the agent playing, and `action` is
            the move leading to that value.
        """

        if state.isWin() or state.isLose():
            return state.getScore(), Directions.STOP

        num_agents = state.getNumAgents()
        next_index = (agent_index + 1) % num_agents

        if agent_index == 0:
            successors = state.generatePacmanSuccessors()
        else:
            successors = state.generateGhostSuccessors(agent_index)

        if not successors:
            return state.getScore(), Directions.STOP

        values = [
            (self.minimax(successor, next_index)[0], action)
            for successor, action in successors
        ]

        if agent_index == 0:
            return max(values, key=lambda pair: pair[0])
        else:
            return min(values, key=lambda pair: pair[0])