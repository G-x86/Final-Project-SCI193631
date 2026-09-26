# Complete this class for all parts of the project

import random
from collections import deque

import numpy as np

from pacman_module.game import Agent, Actions
from pacman_module.pacman import Directions


class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

    def get_action(self, state, belief_state):
        """
        Given a pacman game state and a belief state,
                returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.
        - `belief_state`: a list of probability matrices.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        # XXX: Your code here to obtain bonus
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal and len(legal) > 1:
            legal.remove(Directions.STOP)
        if not legal:
            return Directions.STOP
        target = self._choose_target(state, belief_state)
        if target is None:
            return random.choice(legal)
        action = self._bfs_first_action(state, target, legal)
        if action is None:
            action = self._greedy_action(state, target, legal)
        # XXX: End of your code here to obtain bonus

        return action

    def _choose_target(self, state, belief_state):
        """
        Choose the most attractive ghost cell to chase.

        Arguments:
        ----------
        - `state`: the current game state.
        - `belief_state`: a list of probability matrices, one per ghost.

        Return:
        -------
        - The (x, y) MAP cell of the closest probably-alive ghost,
          or None when no usable belief is available.
        """
        if not isinstance(belief_state, (list, tuple)):
            return None
        pacman_position = state.getPacmanPosition()
        pacman_cell = (int(pacman_position[0]), int(pacman_position[1]))
        best_target = None
        best_distance = None
        for belief in belief_state:
            try:
                mass = float(belief.sum())
            except (AttributeError, TypeError, ValueError):
                continue
            if mass <= 0.0:
                continue
            peak = np.unravel_index(
                int(np.argmax(belief)), belief.shape)
            target = (int(peak[0]), int(peak[1]))
            distance = abs(target[0] - pacman_cell[0]) + \
                abs(target[1] - pacman_cell[1])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_target = target
        return best_target

    def _bfs_first_action(self, state, target, legal):
        """
        Find the first step of a shortest path to the target.

        Arguments:
        ----------
        - `state`: the current game state.
        - `target`: the (x, y) cell to reach.
        - `legal`: the list of legal pacman actions.

        Return:
        -------
        - The first action of a shortest path to `target`,
          or None when the target cannot be reached.
        """
        walls = state.getWalls()
        width = walls.width
        height = walls.height
        pacman_position = state.getPacmanPosition()
        start = (int(pacman_position[0]), int(pacman_position[1]))
        if start == target:
            return None
        queue = deque([start])
        visited = {start: None}
        first_step = {start: None}
        while queue:
            cell = queue.popleft()
            for action in (Directions.NORTH, Directions.SOUTH,
                           Directions.EAST, Directions.WEST):
                vector = Actions.directionToVector(action)
                neighbour = (cell[0] + int(vector[0]),
                             cell[1] + int(vector[1]))
                if not (0 <= neighbour[0] < width
                        and 0 <= neighbour[1] < height):
                    continue
                if walls[neighbour[0]][neighbour[1]]:
                    continue
                if neighbour in visited:
                    continue
                visited[neighbour] = cell
                if cell == start:
                    first_step[neighbour] = action
                else:
                    first_step[neighbour] = first_step[cell]
                if neighbour == target:
                    action = first_step[neighbour]
                    if action in legal:
                        return action
                    return None
                queue.append(neighbour)
        return None

    def _greedy_action(self, state, target, legal):
        """
        Choose the legal action minimizing distance to the target.

        Arguments:
        ----------
        - `state`: the current game state.
        - `target`: the (x, y) cell to approach.
        - `legal`: the list of legal pacman actions.

        Return:
        -------
        - The legal move getting closest to `target`.
        """
        pacman_position = state.getPacmanPosition()
        start = (int(pacman_position[0]), int(pacman_position[1]))
        best_actions = []
        best_distance = None
        for action in legal:
            vector = Actions.directionToVector(action)
            neighbour = (start[0] + int(vector[0]),
                         start[1] + int(vector[1]))
            distance = abs(neighbour[0] - target[0]) + \
                abs(neighbour[1] - target[1])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_actions = [action]
            elif distance == best_distance:
                best_actions.append(action)
        return random.choice(best_actions)
