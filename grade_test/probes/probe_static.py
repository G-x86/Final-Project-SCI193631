"""Check the student's transition and sensor models against the exact ones.

Copied into the submission directory by grade.py and run there:

    python3 probe_static.py <layout.lay> <sensor_variance>

Both references come straight from the handout code:

* transition - `ghostAgents.py` weights a legal move by `2**w` when it does not
  decrease the Manhattan distance to Pacman and by `1` otherwise, then
  normalises; `w` is 0 (confused), 1 (afraid), 3 (scared). In project 2 the
  ghost may reverse, so the model only depends on the ghost's cell.
* sensor - `_get_evidence` adds `binom(n, p) - n*p` to the true distance, hence
  `P(E=e | X=x) = binom.pmf(e - d(x) + n*p, n, p)`.

Prints one line: `GRADER_JSON {...}`.
"""

import json
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import binom

from pacman_module import layout as layout_module

FEAR = {"confused": 0, "afraid": 1, "scared": 3}
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
TOL = 1e-6


def free_cells(walls):
    return [(x, y) for x in range(walls.width) for y in range(walls.height)
            if not walls[x][y]]


def ref_transition(walls, pac, fear):
    w, h = walls.width, walls.height
    T = np.zeros((w, h, w, h))
    for (x2, y2) in free_cells(walls):
        d2 = abs(x2 - pac[0]) + abs(y2 - pac[1])
        moves = []
        for dx, dy in NEIGHBOURS:
            x1, y1 = x2 + dx, y2 + dy
            if 0 <= x1 < w and 0 <= y1 < h and not walls[x1][y1]:
                d1 = abs(x1 - pac[0]) + abs(y1 - pac[1])
                moves.append(((x1, y1), 2.0 ** fear if d1 >= d2 else 1.0))
        total = sum(weight for _, weight in moves)
        for (x1, y1), weight in moves:
            T[x1][y1][x2][y2] = weight / total
    return T


def ref_sensor(walls, pac, evidence, n, p):
    w, h = walls.width, walls.height
    S = np.zeros((w, h))
    for (x, y) in free_cells(walls):
        d = abs(x - pac[0]) + abs(y - pac[1])
        S[x][y] = binom.pmf(evidence - d + n * p, n, p)
    return S


def proportional(student, reference, cells):
    """True if student == c * reference on `cells` for one constant c > 0."""
    ratios = []
    for (x, y) in cells:
        r = reference[x][y]
        if r > 1e-12:
            ratios.append(student[x][y] / r)
    if not ratios:
        return False
    ratios = np.array(ratios)
    return bool(ratios[0] > 0 and np.allclose(ratios, ratios[0], atol=1e-6))


def check_transition(agent, walls, pac, fear):
    out = {"status": "error"}
    try:
        T = np.asarray(agent._get_transition_model(pac), dtype=float)
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    w, h = walls.width, walls.height
    if T.shape != (w, h, w, h):
        out["error"] = "shape %s, expected %s" % (T.shape, (w, h, w, h))
        return out
    R = ref_transition(walls, pac, fear)
    cells = free_cells(walls)
    out["max_err"] = float(max(abs(T[:, :, x, y] - R[:, :, x, y]).max()
                               for (x, y) in cells))
    out["max_sum_err"] = float(max(abs(T[:, :, x, y].sum() - 1.0)
                                   for (x, y) in cells))
    illegal = 0.0
    for (x, y) in cells:
        mask = R[:, :, x, y] <= 0
        illegal = max(illegal, float(T[:, :, x, y][mask].sum()))
    out["illegal_mass"] = illegal
    if out["max_err"] < TOL:
        out["status"] = "exact"
    elif out["max_sum_err"] < 1e-4 and illegal < 1e-9:
        out["status"] = "support"      # valid distribution, wrong weights
    else:
        out["status"] = "invalid"
    return out


def check_sensor(agent, walls, pac, n, p, evidences):
    out = {"status": "error", "per_evidence": {}}
    cells = free_cells(walls)
    verdicts = []
    for evidence in evidences:
        try:
            S = np.asarray(agent._get_sensor_model(pac, evidence),
                           dtype=float)
        except Exception as exc:                              # noqa: BLE001
            out["error"] = "%s: %s" % (type(exc).__name__, exc)
            return out
        if S.shape != (walls.width, walls.height):
            out["error"] = "shape %s, expected %s" % (
                S.shape, (walls.width, walls.height))
            return out
        R = ref_sensor(walls, pac, evidence, n, p)
        err = float(max(abs(S[x][y] - R[x][y]) for (x, y) in cells))
        if err < TOL:
            verdict = "exact"
        elif proportional(S, R, cells):
            verdict = "proportional"
        else:
            verdict = "wrong"
        out["per_evidence"][str(evidence)] = {
            "max_err": err, "verdict": verdict}
        verdicts.append(verdict)
    if all(v == "exact" for v in verdicts):
        out["status"] = "exact"
    elif all(v in ("exact", "proportional") for v in verdicts):
        out["status"] = "proportional"
    else:
        out["status"] = "wrong"
    return out


def main():
    lay = layout_module.getLayout(sys.argv[1])
    variance = float(sys.argv[2])
    walls = lay.walls
    pac = lay.getPacmanPosition()
    p = 0.5
    n = int(variance / (p * (1 - p)))
    evidences = [0, 3, 7, 12]

    from bayesfilter import BeliefStateAgent

    result = {"transition": {}, "sensor": {}}
    for ghost, fear in sorted(FEAR.items()):
        args = SimpleNamespace(
            ghostagent=ghost, sensorvariance=variance, layout=sys.argv[1],
            nghosts=1, seed=1, w=1, p=p, silentdisplay=True,
            hiddenghosts=False, edibleghosts=True,
        )
        try:
            agent = BeliefStateAgent(args)
        except Exception as exc:                              # noqa: BLE001
            result["transition"][ghost] = {
                "status": "error",
                "error": "constructor: %s: %s" % (type(exc).__name__, exc)}
            result["sensor"][ghost] = dict(result["transition"][ghost])
            continue
        agent.walls = walls
        agent.beliefGhostStates = None
        result["transition"][ghost] = check_transition(agent, walls, pac, fear)
        result["sensor"][ghost] = check_sensor(agent, walls, pac, n, p,
                                               evidences)
    print("GRADER_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
