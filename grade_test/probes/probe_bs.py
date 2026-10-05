"""Instrumented wrapper around the student's BeliefStateAgent (project 2).

Copied into the submission directory by grade.py. It subclasses the student's
class and only overrides `_record_metrics`, which the handout explicitly
reserves for measurements, so nothing about the filter itself changes.

One JSON line per belief update is appended to $GRADER_METRICS.
"""

import json
import os

import numpy as np

from bayesfilter import BeliefStateAgent as Student

OUT = os.environ.get("GRADER_METRICS", "grader_metrics.jsonl")


class BeliefStateAgent(Student):
    def _record_metrics(self, belief_states, state):
        walls = state.getWalls()
        free = sum(not walls[x][y]
                   for x in range(walls.width)
                   for y in range(walls.height))
        row = {
            "pac": list(map(int, state.getPacmanPosition())),
            "free": free,
            "ghosts": [],
            "eaten": list(map(bool, state.data._eaten[1:])),
            "sum": [],
            "wall_mass": [],
            "min": [],
            "p_true": [],
            "entropy": [],
        }
        positions = state.getGhostPositions()
        for index, belief in enumerate(belief_states):
            b = np.asarray(belief, dtype=float)
            gx, gy = (int(round(v)) for v in positions[index])
            row["ghosts"].append([gx, gy])
            wall_mass = sum(
                float(b[x][y])
                for x in range(min(b.shape[0], walls.width))
                for y in range(min(b.shape[1], walls.height))
                if walls[x][y]
            )
            row["sum"].append(float(np.nansum(b)))
            row["wall_mass"].append(wall_mass)
            row["min"].append(float(np.nanmin(b)) if b.size else 0.0)
            inside = 0 <= gx < b.shape[0] and 0 <= gy < b.shape[1]
            row["p_true"].append(float(b[gx][gy]) if inside else 0.0)
            pos = b[b > 0]
            row["entropy"].append(
                float(-(pos * np.log(pos)).sum()) if pos.size else 0.0)

        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")
