"""Deterministic wandering Pacman used to drive project 2 belief runs.

Copied into the submission directory by grade.py. It walks a fixed
pseudo-random path (so every student is measured on the same trajectory) and
stops the game after GRADER_STEPS moves, since a project 2 layout has no food
and the game would otherwise never end.

It reads the true ghost positions to keep its distance: the ghosts are edible,
so bumping into the last one wins the game and cuts the measurement short.
That is fine here - this is the harness, not a submitted agent.
"""

import os
import random
import sys

from pacman_module.game import Agent, Directions


class PacmanAgent(Agent):
    def __init__(self, args=None):
        self.args = args
        self.rng = random.Random(int(os.environ.get("GRADER_WALK_SEED", 7)))
        self.steps = 0
        self.limit = int(os.environ.get("GRADER_STEPS", 60))

    def get_action(self, state, belief_state=None):
        # Project 2 hands the belief state to Pacman as a second argument.
        if self.steps >= self.limit:
            sys.stdout.flush()
            sys.exit(0)
        self.steps += 1
        legal = sorted(a for a in state.getLegalActions(0)
                       if a != Directions.STOP)
        if not legal:
            return Directions.STOP
        ghosts = [tuple(map(int, p)) for p in state.getGhostPositions()]
        if not ghosts:
            return self.rng.choice(legal)
        gaps = {a: self.gap(state, a, ghosts) for a in legal}
        safe = [a for a in legal if gaps[a] >= 4]
        if safe:
            return self.rng.choice(safe)
        widest = max(gaps.values())
        return self.rng.choice([a for a in legal if gaps[a] == widest])

    def gap(self, state, action, ghosts):
        """Manhattan distance to the nearest ghost after taking `action`."""
        x, y = state.getPacmanPosition()
        dx, dy = {Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1),
                  Directions.EAST: (1, 0), Directions.WEST: (-1, 0)}[action]
        x, y = x + dx, y + dy
        return min(abs(x - gx) + abs(y - gy) for gx, gy in ghosts)
