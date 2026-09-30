from collections import deque
from pacman_module.game import Agent
from pacman_module.pacman import Directions


def key(state):
    """
    Returns a key that uniquely identifies a Pacman game state.

    Arguments:
    ----------
    - `state`: the current game state. See FAQ and class
               `pacman.GameState`.

    Return:
    -------
    - A hashable key object that uniquely identifies a Pacman game state.
    """
    return (
        state.getPacmanPosition(),
        tuple(tuple(row) for row in state.getFood())
    )


class PacmanAgent(Agent):
    """
    A Pacman agent based on Breadth-First Search (BFS).
    """

    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        super().__init__()
        self.args = args
        self.moves = []

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """
        if not self.moves:
            self.moves = self.bfs(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    def bfs(self, state):
        """
        Given a pacman game state,
        returns a list of legal moves to solve the search layout using BFS.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A list of legal moves as defined in `game.Directions`.
        """
        frontier = deque([(state, [])])
        visited = {key(state)}

        while frontier:
            current_state, actions = frontier.popleft()

            if current_state.isWin():
                return actions

            for next_state, action in current_state.generatePacmanSuccessors():
                next_key = key(next_state)
                if next_key not in visited:
                    visited.add(next_key)
                    frontier.append((next_state, actions + [action]))

        return []
