"""Build `tests.json` from `maps.py` + the exact reference solver.

Run this once, and again whenever you add or change a map:

    python3 bootstrap_tests.py

It calls `reference.py` in a subprocess per map (the solver chdirs into an
engine, so one process per call), fills in the score ceiling and the reference
state count, and derives the default node-count thresholds from it. Everything
it writes is meant to be hand-tuned afterwards: `tests.json` is the rubric.
"""

import json
import os
import subprocess
import sys

import maps

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUTS = os.path.join(HERE, "layouts")

# Maps where BFS (which minimises steps, not score) cannot match the ceiling.
ASTAR_ONLY = {"p0_capsule_detour"}
# The exam demo subset, picked to stay under a minute while still showing
# one map per failure mode.
FAST_P0 = {"p0_adjacent", "p0_snake", "p0_star", "p0_dense",
           "p0_capsule_detour"}
FAST_P1 = {("p1_tight", "dumby"), ("p1_ring", "greedy"),
           ("p1_stub", "smarty"), ("p1_longring", "dumby"),
           ("p1_pillar", "greedy")}
FAST_P2 = {("p2_pockets", 1.0)}
# Layouts too large for plain Minimax without a cutoff.
NO_MINIMAX = {"p1_pillar", "p1_longring"}
P1_GHOSTS = ["dumby", "greedy", "smarty"]
# Provisional H-Minimax node budget: wide enough that a sane
# depth-limited search passes, tight enough to catch a search
# with no cutoff or no pruning. Replace with --calibrate.
DEFAULT_NODES = [1500, 15000]


def solve(engine, layout, ghost=None):
    cmd = [sys.executable, os.path.join(HERE, "reference.py"),
           "--engine", os.path.join(HERE, engine),
           "--layout", os.path.join(LAYOUTS, layout + ".lay")]
    if ghost:
        cmd += ["--ghost", ghost]
    done = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(done.stdout)
    except ValueError:
        return {"error": (done.stderr or done.stdout).strip()[-300:]}


def project0():
    tests = []
    for name in sorted(maps.P0):
        ref = solve("../project0", name)
        if "score" not in ref:
            print("  !! %s: %s" % (name, ref.get("error")))
            continue
        states = ref["states"]
        agents = ["astar"] if name in ASTAR_ONLY else ["bfs", "astar"]
        tests.append({
            "layout": name,
            "note": maps.NOTES[name],
            "fast": name in FAST_P0,
            "agents": agents,
            "optimal": ref["score"],
            "ref_states": states,
            "nodes": {
                "bfs": [round(1.5 * states + 10), round(6 * states + 60)],
                "astar": [round(0.8 * states + 8), round(3 * states + 40)],
            },
        })
        print("  p0 %-20s optimal=%-5d ref_states=%d" %
              (name, ref["score"], states))
    return {
        "weights": {"bfs": 20, "astar": 75, "pep8": 5},
        "score_part": 0.7,
        "nodes_part": 0.3,
        "timeout_sec": 60,
        "fast_timeout_sec": 30,
        "tests": tests,
    }


def project1():
    tests = []
    for name in sorted(maps.P1):
        exact = {}
        for ghost in ("dumby", "greedy"):
            ref = solve("../project1", name, ghost)
            if "score" in ref:
                exact[ghost] = ref
            else:
                print("  !! %s/%s: %s" % (name, ghost, ref.get("error")))
        for ghost in P1_GHOSTS:
            ref = exact.get(ghost)
            # smarty keeps state between calls, so it has no exact ceiling.
            # Fall back to the greedy ceiling and a "must survive" floor.
            ceiling = ref["score"] if ref else (
                exact["greedy"]["score"] if "greedy" in exact else None)
            tests.append({
                "layout": name,
                "ghost": ghost,
                "note": maps.NOTES[name],
                "exact": ref is not None,
                "fast": (name, ghost) in FAST_P1,
                "ceiling": ceiling,
                "minimax": ref is not None and name not in NO_MINIMAX,
                "hminimax": True,
                "score_low": (ceiling - 60) if ref else 0,
                "score_high": ceiling,
                "nodes_low": DEFAULT_NODES[0],
                "nodes_high": DEFAULT_NODES[1],
                "seed": 1,
            })
            if ref:
                print("  p1 %-14s %-7s ceiling=%d" %
                      (name, ghost, ref["score"]))
    return {
        "weights": {"minimax": 45, "hminimax": 50, "pep8": 5},
        "minimax_margin": 0,
        "timeout_sec": 180,
        "fast_timeout_sec": 60,
        "tests": tests,
    }


def project2():
    tests = []
    for name in sorted(maps.P2):
        for ghost in ("scared", "afraid", "confused"):
            tests.append({
                "layout": name,
                "ghost": ghost,
                "note": maps.NOTES[name],
                "steps": 60,
                "fast_steps": 30,
                "fast": (name, 1.0) in FAST_P2,
                "sensor_variance": 1.0,
                "seed": 1,
                "quality_pass": 0.25,
                "quality_target": 0.50,
            })
    # One high-variance run per ghost: the filter must still beat uniform.
    for ghost in ("scared", "afraid", "confused"):
        tests.append({
            "layout": "p2_pockets",
            "ghost": ghost,
            "note": "noisy sensor (variance 4): likelihood must stay correct",
            "steps": 60,
            "fast_steps": 30,
            "fast": False,
            "sensor_variance": 4.0,
            "seed": 2,
            "quality_pass": 0.15,
            "quality_target": 0.35,
        })
    return {
        "weights": {"models": 50, "belief": 50},
        "timeout_sec": 180,
        "fast_timeout_sec": 60,
        "static_layout": "p2_pockets",
        "tests": tests,
    }


if __name__ == "__main__":
    print("project 0")
    p0 = project0()
    print("project 1")
    p1 = project1()
    print("project 2")
    p2 = project2()
    out = {
        "version": 1,
        "generated_by": "bootstrap_tests.py",
        "project0": p0,
        "project1": p1,
        "project2": p2,
    }
    path = os.path.join(HERE, "tests.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print("wrote %s (%d + %d + %d tests)" % (
        path, len(p0["tests"]), len(p1["tests"]), len(p2["tests"])))
