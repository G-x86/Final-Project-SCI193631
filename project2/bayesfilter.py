# Complete this class for all parts of the project

from pacman_module.game import Agent
import numpy as np
from pacman_module import util
from scipy.stats import binom


class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

        """
            Variables to use in 'update_belief_state' method.
            Initialization occurs in 'get_action' method.

            XXX: DO NOT MODIFY THE DEFINITION OF THESE VARIABLES
            # Doing so will result in a 0 grade.
        """

        # Current list of belief states over ghost positions
        self.beliefGhostStates = None

        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None

        # Hyper-parameters
        self.ghost_type = self.args.ghostagent
        self.sensor_variance = self.args.sensorvariance

        self.p = 0.5
        self.n = int(self.sensor_variance/(self.p*(1-self.p)))

        # XXX: Your code here
        # Weight given to ghost moves that do not approach Pacman.
        # 'confused' -> 1.0, 'afraid' -> 2.0, 'scared' -> 8.0.
        self._ghost_param = {"confused": 1.0, "afraid": 2.0,
                             "scared": 8.0}.get(self.ghost_type, 1.0)
        # Cached numpy wall mask, built lazily once walls are known.
        self._walls_array = None
        # Metric histories: one list per time step, one value per ghost.
        self.metric_entropy = []
        self.metric_map_error = []
        self.metric_true_prob = []
        # XXX: End of your code

    def _get_walls_array(self):
        """
        Return the walls as a cached numpy boolean array.

        Return:
        -------
        - A 2D numpy boolean array of size [width, height] where
          the element at position (x, y) is True for a wall.
        """
        width = self.walls.width
        height = self.walls.height
        if self._walls_array is None or self._walls_array.shape != (
                width, height):
            walls_array = np.zeros((width, height), dtype=bool)
            for x in range(width):
                for y in range(height):
                    walls_array[x, y] = bool(self.walls[x][y])
            self._walls_array = walls_array
        return self._walls_array

    def _get_uniform_belief(self):
        """
        Return a uniform belief over all non-wall cells.

        Return:
        -------
        - A 2D numpy array of size [width, height] summing to 1,
          with zeros on walls.
        """
        walls_array = self._get_walls_array()
        uniform = (~walls_array).astype(float)
        uniform /= float(np.sum(uniform))
        return uniform

    def _get_sensor_model(self, pacman_position, evidence):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        The sensor model represented as a 2D numpy array of
        size [width, height].
        The element at position (w, h) is the probability
        P(E_t=evidence | X_t=(w, h))
        """
        width = self.walls.width
        height = self.walls.height
        pacman_x = int(pacman_position[0])
        pacman_y = int(pacman_position[1])
        columns = np.arange(width).reshape(width, 1)
        rows = np.arange(height).reshape(1, height)
        distances = (np.abs(columns - pacman_x) +
                     np.abs(rows - pacman_y)).astype(float)
        if self.n <= 0:
            likelihood = np.where(
                np.abs(distances - evidence) < 1e-9, 1.0, 0.0)
        else:
            shifts = evidence - distances + self.n * self.p
            rounded = np.round(shifts)
            is_integer = np.abs(shifts - rounded) < 1e-6
            likelihood = np.where(
                is_integer, binom.pmf(rounded, self.n, self.p), 0.0)
        likelihood[self._get_walls_array()] = 0.0
        return likelihood

    def _get_transition_model(self, pacman_position):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        The transition model represented as a 4D numpy array of
        size [width, height, width, height].
        The element at position (w1, h1, w2, h2) is the probability
        P(X_t+1=(w1, h1) | X_t=(w2, h2))
        """
        width = self.walls.width
        height = self.walls.height
        pacman_x = int(pacman_position[0])
        pacman_y = int(pacman_position[1])
        walls_array = self._get_walls_array()
        model = np.zeros((width, height, width, height))
        for source_x in range(width):
            for source_y in range(height):
                if walls_array[source_x, source_y]:
                    continue
                neighbours = []
                for delta_x, delta_y in ((0, 1), (0, -1),
                                         (1, 0), (-1, 0)):
                    target_x = source_x + delta_x
                    target_y = source_y + delta_y
                    if (0 <= target_x < width
                            and 0 <= target_y < height
                            and not walls_array[target_x, target_y]):
                        neighbours.append((target_x, target_y))
                if not neighbours:
                    model[source_x, source_y, source_x, source_y] = 1.0
                    continue
                current = abs(source_x - pacman_x) + \
                    abs(source_y - pacman_y)
                weights = []
                for target_x, target_y in neighbours:
                    successor = abs(target_x - pacman_x) + \
                        abs(target_y - pacman_y)
                    if successor >= current:
                        weights.append(self._ghost_param)
                    else:
                        weights.append(1.0)
                total = float(sum(weights))
                for (target_x, target_y), weight in zip(neighbours,
                                                        weights):
                    model[target_x, target_y,
                          source_x, source_y] = weight / total
        return model

    def _get_updated_belief(self, belief, evidences, pacman_position,
                            ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        and the previous belief states before receiving the evidences,
        returns the updated list of belief states about ghosts positions

        Arguments:
        ----------
        - `belief`: A list of Z belief states at state x_{t-1}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze.
               Matrices filled with zeros must be returned for eaten ghosts.
        """

        # XXX: Your code here
        width = self.walls.width
        height = self.walls.height
        transition = self._get_transition_model(pacman_position)
        updated = []
        for index, prior in enumerate(belief):
            if ghosts_eaten[index]:
                updated.append(np.zeros((width, height)))
                continue
            predicted = np.tensordot(transition, prior,
                                     axes=([2, 3], [0, 1]))
            likelihood = self._get_sensor_model(pacman_position,
                                                evidences[index])
            posterior = predicted * likelihood
            total = float(np.sum(posterior))
            if total > 0.0 and np.isfinite(total):
                posterior = posterior / total
            else:
                predicted_total = float(np.sum(predicted))
                if predicted_total > 0.0 and np.isfinite(predicted_total):
                    posterior = predicted / predicted_total
                else:
                    posterior = self._get_uniform_belief()
            updated.append(posterior)
        # XXX: End of your code

        return updated

    def update_belief_state(self, evidences, pacman_position, ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        belief = self._get_updated_belief(self.beliefGhostStates, evidences,
                                          pacman_position, ghosts_eaten)
        self.beliefGhostStates = belief
        return belief

    def _get_evidence(self, state):
        """
        Computes noisy distances between pacman and ghosts.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.


        Return:
        -------
        - A list of Z noised distances in real numbers
          where Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        pacman_position = state.getPacmanPosition()
        noisy_distances = []

        for pos in positions:
            true_distance = util.manhattanDistance(pos, pacman_position)
            noise = binom.rvs(self.n, self.p) - self.n*self.p
            noisy_distances.append(true_distance + noise)

        return noisy_distances

    def _record_metrics(self, belief_states, state):
        """
        Use this function to record your metrics
        related to true and belief states.
        Won't be part of specification grading.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.
        - `belief_states`: A list of Z
           N*M numpy matrices of probabilities
           where N and M are respectively width and height
           of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """
        width = self.walls.width
        height = self.walls.height
        true_positions = state.getGhostPositions()
        try:
            eaten = list(state.data._eaten[1:1 + len(belief_states)])
        except (AttributeError, TypeError):
            eaten = [False] * len(belief_states)
        entropies = []
        map_errors = []
        true_probs = []
        for belief, true_position, is_eaten in zip(
                belief_states, true_positions, eaten):
            true_x = int(true_position[0])
            true_y = int(true_position[1])
            alive = (not is_eaten and 0 <= true_x < width
                     and 0 <= true_y < height)
            if not alive or float(np.sum(belief)) <= 0.0:
                entropies.append(float("nan"))
                map_errors.append(float("nan"))
                true_probs.append(float("nan"))
                continue
            flat = belief.ravel()
            positive = flat[flat > 0.0]
            entropy = float(-np.sum(positive * np.log(positive)))
            entropies.append(entropy)
            best = np.unravel_index(int(np.argmax(belief)), belief.shape)
            error = abs(best[0] - true_x) + abs(best[1] - true_y)
            map_errors.append(float(error))
            true_probs.append(float(belief[true_x, true_y]))
        self.metric_entropy.append(entropies)
        self.metric_map_error.append(map_errors)
        self.metric_true_prob.append(true_probs)

    def get_action(self, state):
        """
        Given a pacman game state, returns a belief state.

        Arguments:
        ----------
        - `state`: the current game state.
                   See FAQ and class `pacman.GameState`.

        Return:
        -------
        - A belief state.
        """

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """
        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()

        evidence = self._get_evidence(state)
        newBeliefStates = self.update_belief_state(evidence,
                                                   state.getPacmanPosition(),
                                                   state.data._eaten[1:])
        self._record_metrics(self.beliefGhostStates, state)

        return newBeliefStates, evidence
