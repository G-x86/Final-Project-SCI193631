import heapq
from itertools import count
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


def heuristic(state):
    """
    Computes an admissible heuristic for A* based on Manhattan distance
    to the furthest remaining food dot.

    Arguments:
    ----------
    - `state`: the current game state.

    Return:
    -------
    - An estimated cost (integer) to eat all remaining food dots.
    """
    pos = state.getPacmanPosition()
    food_grid = state.getFood()
    food_positions = food_grid.asList()

    if not food_positions:
        return 0

    return max(
        abs(pos[0] - fx) + abs(pos[1] - fy)
        for fx, fy in food_positions
    )


class PacmanAgent(Agent):
    """
    A Pacman agent based on A* Search.
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
            self.moves = self.astar(state)

        try:
            return self.moves.pop(0)
        except IndexError:
            return Directions.STOP

    def astar(self, state):
        """
        Given a pacman game state,
        returns a list of legal moves to solve the search layout using A*.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A list of legal moves as defined in `game.Directions`.
        """
        counter = count()
        start_key = key(state)

        frontier = [(heuristic(state), next(counter), 0, state, [])]
        best_g = {start_key: 0}

        while frontier:
            f, _, g, current_state, actions = heapq.heappop(frontier)
            curr_key = key(current_state)

            if current_state.isWin():
                return actions

            if g > best_g.get(curr_key, float("inf")):
                continue

            for next_state, action in current_state.generatePacmanSuccessors():
                next_key = key(next_state)
                next_g = g + 1

                if next_g < best_g.get(next_key, float("inf")):
                    best_g[next_key] = next_g
                    next_h = heuristic(next_state)
                    next_f = next_g + next_h
                    heapq.heappush(
                        frontier,
                        (
                            next_f,
                            next(counter),
                            next_g,
                            next_state,
                            actions + [action]
                        )
                    )

        return []
