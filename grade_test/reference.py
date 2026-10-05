"""Exact optimal-score reference for Pacman projects 0 and 1.

Why this works without an instructor solution: when Pacman wins, the score is

    score = 500 + 10 * F_total - steps - 5 * capsules_eaten

F_total is fixed by the layout, so maximising the score is exactly minimising
`cost = steps + 5 * capsules_eaten`. Every Pacman move costs at least 1, so a
Dijkstra over real engine states (`generateSuccessor`) returns the true
optimum. Deterministic ghosts (`dumby`, `greedy`) collapse the adversary into
a plain state transition, which is why project 1 can be graded exactly too.
`smarty` keeps A* scores between calls, so it is not a function of the state
and is deliberately refused here: grade it against calibrated thresholds.

Usage:
    python reference.py --engine ../project0 --layout p0_star
    python reference.py --engine ../project1 --layout p1_ring --ghost dumby
    python reference.py --engine ../project1 --all-p1        # JSON on stdout
"""

import argparse
import heapq
import itertools
import json
import os
import sys


class NonDeterministicGhost(Exception):
    pass


def load_engine(engine_dir):
    """Import `pacman_module` from `engine_dir` and chdir there.

    The chdir is required: `layout.getLayout` resolves layouts relative to the
    working directory.
    """
    engine_dir = os.path.abspath(engine_dir)
    sys.path.insert(0, engine_dir)
    os.chdir(engine_dir)
    from pacman_module import pacman, layout, ghostAgents
    return pacman, layout, ghostAgents


def ghost_action(agent, state):
    """Return the single action a deterministic ghost takes in `state`."""
    dist = agent.getDistribution(state)
    support = [a for a, p in dist.items() if p > 1e-9]
    if len(support) > 1:
        raise NonDeterministicGhost(
            "ghost has %d possible actions" % len(support))
    return support[0] if support else None


def state_key(state):
    """Hashable summary of everything that drives the dynamics."""
    data = state.data
    ghosts = tuple(
        (tuple(a.configuration.pos), a.configuration.direction)
        for a in data.agentStates[1:]
    )
    return (
        tuple(data.agentStates[0].configuration.pos),
        tuple(sorted(state.getFood().asList())),
        tuple(sorted(data.capsules)),
        ghosts,
    )


def solve(engine_dir, layout_name, ghost=None, max_states=500000):
    """Return the best score reachable by any play, or why there is none.

    Every state key fixes how much food and how many capsules are gone, so its
    score is `const(key) - steps`: minimising cost also maximises the score at
    that state. The search therefore visits the whole reachable space and keeps
    the best *terminal* score (a win, or the least bad death on a layout that
    cannot be won).

    Returns a dict with `score`, `win`, `cost` (steps + 5 * capsules),
    `states` (states expanded before the best terminal was found, a fair scale
    for "roughly the same number of nodes"), `states_total`, or `error`.
    """
    pacman, layout, ghostAgents = load_engine(engine_dir)
    names = {
        "dumby": ghostAgents.DumbyGhost,
        "greedy": ghostAgents.GreedyGhost,
        "smarty": ghostAgents.SmartyGhost,
        "eastrandy": ghostAgents.EastRandyGhost,
    }
    if ghost in ("smarty", "eastrandy"):
        return {"error": "%s is not a function of the state" % ghost}

    lay = layout.getLayout(layout_name)
    if lay is None:
        return {"error": "layout %s not found" % layout_name}
    nghosts = lay.getNumGhosts()
    if nghosts and ghost is None:
        return {"error": "layout has %d ghost(s): --ghost required" % nghosts}

    start = pacman.GameState()
    start.initialize(lay, nghosts)
    agents = [names[ghost](i + 1) for i in range(nghosts)] if nghosts else []

    tie = itertools.count()
    best = {state_key(start): 0}
    heap = [(0, next(tie), start)]
    expanded = 0
    found = None

    while heap:
        cost, _, state = heapq.heappop(heap)
        key = state_key(state)
        if cost > best.get(key, float("inf")):
            continue
        expanded += 1
        if expanded > max_states:
            return {"error": "state limit %d exceeded" % max_states}
        if expanded % 2000 == 0:
            pacman.GameState.explored.clear()

        for action in state.getLegalActions(0):
            if action == "Stop":
                # `generatePacmanSuccessors` drops Stop, so a student agent
                # cannot reach a plan that waits. The ceiling must not either.
                continue
            nxt = state.generateSuccessor(0, action)
            step = 1 + 5 * (len(state.getCapsules()) - len(nxt.getCapsules()))
            if not (nxt.isWin() or nxt.isLose()):
                for index, agent in enumerate(agents, start=1):
                    move = ghost_action(agent, nxt)
                    if move is None:
                        break
                    nxt = nxt.generateSuccessor(index, move)
                    if nxt.isLose() or nxt.isWin():
                        break
            ncost = cost + step
            if nxt.isWin() or nxt.isLose():
                # Terminal: the game is over here, so this score is final.
                candidate = (int(nxt.getScore()), bool(nxt.isWin()))
                if found is None or candidate[0] > found[0]:
                    found = (candidate[0], candidate[1], ncost, expanded)
                continue
            nkey = state_key(nxt)
            if ncost < best.get(nkey, float("inf")):
                best[nkey] = ncost
                heapq.heappush(heap, (ncost, next(tie), nxt))

    if found is None:
        return {"error": "no terminal state is reachable"}
    return {
        "score": found[0],
        "win": found[1],
        "cost": found[2],
        "states": found[3],
        "states_total": expanded,
        "food": lay.totalFood,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True,
                        help="a project directory containing pacman_module")
    parser.add_argument("--layout")
    parser.add_argument("--ghost", choices=["dumby", "greedy", "smarty"])
    parser.add_argument("--max-states", type=int, default=500000)
    args = parser.parse_args()

    out = solve(args.engine, args.layout, args.ghost, args.max_states)
    print(json.dumps(out, indent=2))
    return 0 if "score" in out else 1


if __name__ == "__main__":
    sys.exit(main())
