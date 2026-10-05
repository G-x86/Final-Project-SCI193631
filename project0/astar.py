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
        tuple(tuple(row) for row in state.getFood()),
        tuple(sorted(state.getCapsules()))
    )


def heuristic(state):
    """
    Computes an admissible and tight heuristic for A*.
    Combines the distance to the nearest target with an exact minimum
    tour (for small target sets) or MST lower bound across targets.

    Arguments:
    ----------
    - `state`: the current game state.

    Return:
    -------
    - An estimated cost (integer) to collect all remaining targets.
    """
    pos = state.getPacmanPosition()
    targets = state.getFood().asList() + list(state.getCapsules())

    if not targets:
        return 0

    min_dist = min(abs(pos[0] - tx) + abs(pos[1] - ty) for tx, ty in targets)

    min_x = min(tx for tx, ty in targets)
    max_x = max(tx for tx, ty in targets)
    min_y = min(ty for tx, ty in targets)
    max_y = max(ty for tx, ty in targets)
    span = (max_x - min_x) + (max_y - min_y)

    if len(targets) <= 4:
        def get_min_tour(curr, remaining):
            if not remaining:
                return 0
            best = float("inf")
            for nxt in remaining:
                d = abs(curr[0] - nxt[0]) + abs(curr[1] - nxt[1])
                rem = [p for p in remaining if p != nxt]
                cost = d + get_min_tour(nxt, rem)
                if cost < best:
                    best = cost
            return best

        return max(span, get_min_tour(pos, targets))

    unvisited = set(targets)
    current = unvisited.pop()
    mst_cost = 0
    closest_dist = {
        t: abs(current[0] - t[0]) + abs(current[1] - t[1])
        for t in unvisited
    }

    while unvisited:
        next_target = min(unvisited, key=lambda t: closest_dist[t])
        mst_cost += closest_dist[next_target]
        unvisited.remove(next_target)
        for t in unvisited:
            d = abs(next_target[0] - t[0]) + abs(next_target[1] - t[1])
            if d < closest_dist[t]:
                closest_dist[t] = d

    return max(span, min_dist + mst_cost)


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

                step_cost = 1
                if len(next_state.getCapsules()) < len(
                    current_state.getCapsules()
                ):
                    step_cost += 5

                next_g = g + step_cost

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
