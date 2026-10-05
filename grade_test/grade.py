"""Run the grading layouts against a submission and produce a grade.

    python3 grade.py --submission ~/subs/group07 --project 0
    python3 grade.py --submission ~/subs/group07 --project all
    python3 grade.py --submission ../project0 --project 0 --calibrate

`--reference DIR` grades inside a throwaway copy of your own project folder
with only the student's deliverables dropped in, so an edited `pacman_module`
cannot change the result.

`--calibrate` does not grade: it runs the submission and writes the measured
scores and node counts back into tests.json as thresholds. Use it once with
your own reference agents, mainly for project 1 (`smarty` and all node
budgets, which no solver can predict).
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUTS = os.path.join(HERE, "layouts")
PROBES = os.path.join(HERE, "probes")

DELIVERABLES = {
    0: ["bfs.py", "astar.py"],
    1: ["minimax.py", "hminimax.py"],
    2: ["bayesfilter.py"],
}
SCORE_RE = re.compile(r"score\s*:?\s*(-?\d+(?:\.\d+)?)", re.I)
NODES_RE = re.compile(r"expanded nodes\s*:?\s*(\d+)", re.I)


# ----------------------------------------------------------------- utilities

def find_project(root, number):
    """Directory holding a runnable project (contains run.py)."""
    for candidate in (os.path.join(root, "project%d" % number), root):
        if os.path.isfile(os.path.join(candidate, "run.py")):
            return candidate
    return None


def find_deliverables(root, number):
    """Directory holding the student's files; they need not ship run.py."""
    candidates = [os.path.join(root, "project%d" % number), root]
    for candidate in candidates:
        if any(os.path.isfile(os.path.join(candidate, name))
               for name in DELIVERABLES[number]):
            return candidate
    return next((c for c in candidates if os.path.isdir(c)), None)


def stage(submission, reference, number):
    """Return the directory to grade in (a copy if `reference` is given)."""
    if not reference:
        return find_project(submission, number), None
    source = find_deliverables(submission, number)
    if source is None:
        return None, None
    base = find_project(reference, number)
    if base is None:
        raise SystemExit("no project%d/run.py under %s" % (number, reference))
    tmp = tempfile.mkdtemp(prefix="grade_p%d_" % number)
    work = os.path.join(tmp, "project")
    shutil.copytree(base, work, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "temp"))
    for name in DELIVERABLES[number] + ["pacmanagent.py"]:
        src = os.path.join(source, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(work, name))
    return work, tmp


def install_layouts(workdir, number):
    target = os.path.join(workdir, "pacman_module", "layouts")
    os.makedirs(target, exist_ok=True)
    count = 0
    for name in sorted(os.listdir(LAYOUTS)):
        if name.startswith("p%d_" % number) and name.endswith(".lay"):
            shutil.copy(os.path.join(LAYOUTS, name),
                        os.path.join(target, name))
            count += 1
    return count


def run(cmd, cwd, timeout, env=None):
    """Run a game, return (status, score, nodes, seconds, output tail)."""
    full_env = dict(os.environ)
    full_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        full_env.update(env)
    start = time.time()
    try:
        done = subprocess.run(cmd, cwd=cwd, env=full_env, timeout=timeout,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return "timeout", None, None, timeout, ""
    seconds = round(time.time() - start, 2)
    out = (done.stdout or "") + (done.stderr or "")
    scores = SCORE_RE.findall(done.stdout or "")
    nodes = NODES_RE.findall(done.stdout or "")
    if done.returncode != 0 or not scores:
        return "crash", None, None, seconds, out.strip()[-400:]
    illegal = "Illegal move" in out
    return ("illegal" if illegal else "ok",
            int(float(scores[-1])), int(nodes[-1]) if nodes else None,
            seconds, "")


def ramp_down(value, full, zero):
    """1 at or below `full`, 0 at or above `zero`, linear in between."""
    if value is None or full is None or zero is None or zero <= full:
        return 0.0
    return max(0.0, min(1.0, (zero - value) / float(zero - full)))


def ramp_up(value, low, high):
    if value is None or low is None or high is None or high <= low:
        return 1.0 if (value is not None and low is not None
                       and value >= low) else 0.0
    return max(0.0, min(1.0, (value - low) / float(high - low)))


def pep8(workdir, files):
    present = [f for f in files if os.path.isfile(os.path.join(workdir, f))]
    if not present:
        return {"grade": 0.0, "detail": "no files to check"}
    done = subprocess.run([sys.executable, "-m", "pycodestyle"] + present,
                          cwd=workdir, capture_output=True, text=True)
    if done.returncode and "No module named" in (done.stderr or ""):
        return {"grade": None, "detail": "pycodestyle not installed"}
    errors = [ln for ln in (done.stdout or "").splitlines() if ln.strip()]
    return {"grade": 1.0 if not errors else 0.0,
            "detail": "%d error(s)" % len(errors),
            "errors": errors[:20]}


# ------------------------------------------------------------------ project 0

def grade_p0(work, spec, calibrate):
    results = []
    for test in spec["tests"]:
        for agent in test["agents"]:
            agent_file = agent + ".py"
            row = {"layout": test["layout"], "agent": agent,
                   "note": test["note"], "optimal": test["optimal"]}
            if not os.path.isfile(os.path.join(work, agent_file)):
                row.update(status="missing", grade=0.0)
                results.append(row)
                continue
            status, score, nodes, secs, tail = run(
                [sys.executable, "run.py", "--agentfile", agent_file,
                 "--layout", test["layout"], "--silentdisplay"],
                work, spec["timeout_sec"])
            full, zero = test["nodes"][agent]
            optimal = 1.0 if (status == "ok"
                              and score == test["optimal"]) else 0.0
            node_grade = (ramp_down(nodes, full, zero)
                          if status == "ok" else 0.0)
            row.update(status=status, score=score, nodes=nodes, sec=secs,
                       nodes_full=full, nodes_zero=zero,
                       optimal_hit=bool(optimal),
                       grade=(spec["score_part"] * optimal
                              + spec["nodes_part"] * node_grade),
                       output=tail)
            if calibrate and status == "ok" and nodes:
                test["nodes"][agent] = [round(nodes * 1.2) + 5,
                                        round(nodes * 4.0) + 40]
                row["calibrated"] = test["nodes"][agent]
            results.append(row)
    parts = {}
    for agent in ("bfs", "astar"):
        rows = [r for r in results if r["agent"] == agent]
        parts[agent] = (sum(r["grade"] for r in rows) / len(rows)
                        if rows else 0.0)
    return results, parts


# ------------------------------------------------------------------ project 1

def grade_p1(work, spec, calibrate):
    results = []
    for test in spec["tests"]:
        for agent in ("minimax", "hminimax"):
            if not test.get(agent):
                continue
            row = {"layout": test["layout"], "ghost": test["ghost"],
                   "agent": agent, "note": test["note"],
                   "ceiling": test["ceiling"], "exact": test["exact"]}
            if not os.path.isfile(os.path.join(work, agent + ".py")):
                row.update(status="missing", grade=0.0)
                results.append(row)
                continue
            status, score, nodes, secs, tail = run(
                [sys.executable, "run.py", "--agent", agent,
                 "--layout", test["layout"], "--ghost", test["ghost"],
                 "--nographics", "--seed", str(test.get("seed", 1))],
                work, spec["timeout_sec"])
            row.update(status=status, score=score, nodes=nodes, sec=secs,
                       output=tail)
            if status != "ok":
                row["grade"] = 0.0
            elif agent == "minimax":
                if test["ceiling"] is None:
                    row.update(grade=0.0, status="no ceiling")
                    results.append(row)
                    continue
                target = test["ceiling"] - spec.get("minimax_margin", 0)
                row["target"] = target
                row["grade"] = 1.0 if score >= target else 0.0
            else:
                low, high = test["score_low"], test["score_high"]
                n_low, n_high = test["nodes_low"], test["nodes_high"]
                if score < low or (n_high is not None and nodes is not None
                                   and nodes > n_high):
                    row["grade"] = 0.0
                    row["failed_threshold"] = True
                else:
                    s = ramp_up(score, low, high)
                    if n_low is None or n_high is None:
                        row["grade"] = s
                        row["nodes_graded"] = False
                    else:
                        row["grade"] = 0.5 * s + 0.5 * ramp_down(
                            nodes, n_low, n_high)
                        row["nodes_graded"] = True
            if calibrate and status == "ok" and agent == "hminimax":
                # A reference must never lower a ceiling that was computed
                # exactly. On a `smarty` row there is no exact ceiling, only
                # a proxy borrowed from `greedy`, so the reference replaces
                # it: it is the only evidence of what is reachable there.
                if not test["exact"]:
                    test["ceiling"] = score
                test["score_high"] = test["ceiling"]
                test["score_low"] = min(test["score_high"] - 1, score - 50)
                if nodes:
                    # Node counts swing a lot with the cutoff depth, and
                    # exceeding nodes_high is a hard fail, so stay generous.
                    test["ref_nodes"] = nodes
                    test["nodes_low"] = round(nodes * 1.5)
                    test["nodes_high"] = round(nodes * 10.0)
                row["calibrated"] = [test["score_low"], test["score_high"],
                                     test["nodes_low"], test["nodes_high"]]
            results.append(row)
    parts = {}
    for agent in ("minimax", "hminimax"):
        rows = [r for r in results if r["agent"] == agent]
        parts[agent] = (sum(r["grade"] for r in rows) / len(rows)
                        if rows else 0.0)
    return results, parts


# ------------------------------------------------------------------ project 2

def ghost_count(layout_name):
    with open(os.path.join(LAYOUTS, layout_name + ".lay")) as f:
        return f.read().count("G")


def read_metrics(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def score_metrics(rows, quality_pass, quality_target):
    """Validity fraction and informativeness of the recorded beliefs.

    `quality` is how far the mean negative log-likelihood of the ghost's true
    cell gets from a uniform guess over the free cells, as a fraction. A
    correct filter cannot reach 1.0 - the ghost's position stays genuinely
    uncertain - so `quality_target` is what counts as full marks.
    """
    if not rows:
        return {"steps": 0, "validity": 0.0, "quality": 0.0, "nll": None,
                "baseline": None, "passed": False, "graded_quality": 0.0}
    valid_steps = 0
    for row in rows:
        ok = True
        for i, eaten in enumerate(row["eaten"]):
            total, wall = row["sum"][i], row["wall_mass"][i]
            if eaten:
                ok = ok and abs(total) < 1e-9
            else:
                ok = ok and abs(total - 1.0) < 1e-6
                ok = ok and wall < 1e-9
                ok = ok and row["min"][i] > -1e-12
        valid_steps += bool(ok)
    tail = rows[len(rows) // 2:]
    lls, baseline = [], math.log(rows[0]["free"])
    for row in tail:
        for i, eaten in enumerate(row["eaten"]):
            if not eaten:
                lls.append(-math.log(max(row["p_true"][i], 1e-12)))
    nll = sum(lls) / len(lls) if lls else None
    quality = 0.0 if nll is None else max(
        0.0, min(1.0, (baseline - nll) / baseline))
    return {"steps": len(rows), "validity": valid_steps / len(rows),
            "quality": quality, "nll": nll, "baseline": baseline,
            "passed": quality >= quality_pass,
            "graded_quality": min(1.0, quality / quality_target)}


def grade_p2(work, spec, calibrate):
    for name in ("probe_pacman.py", "probe_bs.py", "probe_static.py"):
        shutil.copy(os.path.join(PROBES, name), os.path.join(work, name))
    tmp = tempfile.mkdtemp(prefix="grade_p2_metrics_")
    results, models = [], {}
    try:
        static = run_static(work, spec)
        models = static["parts"]
        for test in spec["tests"]:
            # Ghost start positions are random in project 2, so an unlucky
            # seed can let Pacman eat the last ghost at once and end the run.
            # Retry with another seed when that happens.
            for attempt in range(3):
                seed = test.get("seed", 1) + 100 * attempt
                metrics_path = os.path.join(tmp, "%s_%s_%s_%d.jsonl" % (
                    test["layout"], test["ghost"], test["sensor_variance"],
                    seed))
                status, _, _, secs, tail = run(
                    [sys.executable, "run.py",
                     "--agentfile", "probe_pacman.py",
                     "--bsagentfile", "probe_bs.py",
                     "--ghostagent", test["ghost"],
                     "--nghosts", str(ghost_count(test["layout"])),
                     "--layout", test["layout"],
                     "--seed", str(seed),
                     "--sensorvariance", str(test["sensor_variance"]),
                     "--silentdisplay"],
                    work, spec["timeout_sec"],
                    env={"GRADER_METRICS": metrics_path,
                         "GRADER_STEPS": str(test["steps"])})
                # The probe stops the game with sys.exit, so "crash" (no
                # score printed) is normal here: the metrics file decides.
                recorded = read_metrics(metrics_path)
                if len(recorded) >= 0.8 * test["steps"]:
                    break
            metrics = score_metrics(recorded, test["quality_pass"],
                                    test["quality_target"])
            row = {"layout": test["layout"], "ghost": test["ghost"],
                   "variance": test["sensor_variance"], "note": test["note"],
                   "sec": secs, "seed": seed,
                   "short": len(recorded) < 0.8 * test["steps"],
                   "status": "ok" if metrics["steps"] else status}
            row.update(metrics)
            row["grade"] = (0.4 * metrics["validity"]
                            + 0.6 * (metrics["graded_quality"]
                                     if metrics["passed"] else 0.0))
            if not metrics["steps"]:
                row["output"] = tail
            if calibrate and metrics["steps"]:
                test["quality_target"] = round(
                    max(0.10, metrics["quality"] * 0.85), 3)
                test["quality_pass"] = round(metrics["quality"] * 0.4, 3)
                row["calibrated"] = [test["quality_pass"],
                                     test["quality_target"]]
            results.append(row)
    finally:
        for name in ("probe_pacman.py", "probe_bs.py", "probe_static.py"):
            path = os.path.join(work, name)
            if os.path.isfile(path):
                os.remove(path)
    belief = (sum(r["grade"] for r in results) / len(results)
              if results else 0.0)
    return results, {"models": models.get("models", 0.0), "belief": belief,
                     "static": models}


def run_static(work, spec):
    layout = spec["static_layout"]
    done = subprocess.run(
        [sys.executable, "probe_static.py",
         os.path.join("pacman_module", "layouts", layout + ".lay"), "1.0"],
        cwd=work, capture_output=True, text=True, timeout=spec["timeout_sec"])
    line = next((ln for ln in (done.stdout or "").splitlines()
                 if ln.startswith("GRADER_JSON ")), None)
    if line is None:
        return {"parts": {"models": 0.0},
                "error": ((done.stdout or "") + (done.stderr or ""))
                .strip()[-400:]}
    detail = json.loads(line[len("GRADER_JSON "):])
    points = {"exact": 1.0, "support": 0.5, "proportional": 0.5}
    per_ghost = {}
    for ghost in sorted(detail["transition"]):
        t = points.get(detail["transition"][ghost].get("status"), 0.0)
        s = points.get(detail["sensor"][ghost].get("status"), 0.0)
        per_ghost[ghost] = {"transition": t, "sensor": s}
    grade = (sum(0.5 * v["transition"] + 0.5 * v["sensor"]
                 for v in per_ghost.values()) / len(per_ghost)
             if per_ghost else 0.0)
    return {"parts": {"models": grade}, "per_ghost": per_ghost,
            "detail": detail}


# -------------------------------------------------------------------- driver

def select_fast(spec):
    """Shrink a spec to the exam demo: tests marked `fast`, short timeouts."""
    chosen = [t for t in spec["tests"] if t.get("fast")] or spec["tests"]
    tests = []
    for test in chosen:
        test = dict(test)
        if "fast_steps" in test:
            test["steps"] = test["fast_steps"]
        tests.append(test)
    out = dict(spec)
    out["tests"] = tests
    out["timeout_sec"] = spec.get("fast_timeout_sec", 30)
    return out


def grade_project(number, submission, reference, spec, calibrate):
    work, tmp = stage(submission, reference, number)
    if work is None:
        return {"project": number,
                "error": "no project %d files found under %s" % (
                    number, submission)}
    try:
        install_layouts(work, number)
        if number == 0:
            rows, parts = grade_p0(work, spec, calibrate)
        elif number == 1:
            rows, parts = grade_p1(work, spec, calibrate)
        else:
            rows, parts = grade_p2(work, spec, calibrate)
        style = pep8(work, DELIVERABLES[number])
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    weights = spec["weights"]
    total, missing_weight = 0.0, 0.0
    breakdown = {}
    for key, weight in weights.items():
        if key == "pep8":
            value = style["grade"]
        else:
            value = parts.get(key)
        if value is None:
            missing_weight += weight
            breakdown[key] = {"weight": weight, "grade": None,
                              "points": None}
            continue
        breakdown[key] = {"weight": weight, "grade": round(value, 4),
                          "points": round(weight * value, 2)}
        total += weight * value
    graded_weight = sum(w for k, w in weights.items()) - missing_weight
    return {"project": number, "tests": rows, "style": style,
            "breakdown": breakdown, "points": round(total, 2),
            "out_of": graded_weight,
            "percent": round(100.0 * total / graded_weight, 2)
            if graded_weight else None}


def report(result):
    number = result["project"]
    print("\n=== project %d ===" % number)
    if "error" in result:
        print("  !! %s" % result["error"])
        return
    for row in result["tests"]:
        if number == 2:
            print("  %-13s %-9s v=%-4s %-8s steps=%-3d valid=%4.0f%% "
                  "quality=%4.0f%%/%2.0f%% nll=%s -> %.2f" % (
                      row["layout"], row["ghost"], row["variance"],
                      row["status"], row["steps"], 100 * row["validity"],
                      100 * row["quality"], 100 * row["graded_quality"],
                      "n/a" if row["nll"] is None else "%.2f" % row["nll"],
                      row["grade"]))
        elif number == 0:
            print("  %-20s %-6s %-8s score=%-5s (opt %-5s) nodes=%-7s "
                  "[%s..%s] -> %.2f" % (
                      row["layout"], row["agent"], row["status"],
                      row.get("score"), row["optimal"], row.get("nodes"),
                      row.get("nodes_full"), row.get("nodes_zero"),
                      row["grade"]))
        else:
            print("  %-13s %-7s %-9s %-8s score=%-6s ceiling=%-6s nodes=%-7s "
                  "-> %.2f%s" % (
                      row["layout"], row["ghost"], row["agent"],
                      row["status"], row.get("score"), row["ceiling"],
                      row.get("nodes"), row["grade"],
                      "" if row.get("exact") else "  (uncalibrated)"))
        if row.get("output"):
            print("        %s" % row["output"].replace("\n", "\n        "))
    print("  style: %s" % result["style"]["detail"])
    for key, item in sorted(result["breakdown"].items()):
        print("  %-10s weight %-4s grade %-7s points %s" % (
            key, item["weight"],
            "n/a" if item["grade"] is None else "%.3f" % item["grade"],
            "n/a" if item["points"] is None else item["points"]))
    print("  TOTAL %.2f / %s  (%s%%)" % (
        result["points"], result["out_of"], result["percent"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default=os.path.dirname(HERE),
                        help="student directory (a project folder, or a "
                             "folder containing project0/1/2). Defaults to "
                             "the folder holding this grading kit.")
    parser.add_argument("--project", default="all",
                        choices=["0", "1", "2", "all"])
    parser.add_argument("--reference",
                        help="grade inside a copy of this clean project tree")
    parser.add_argument("--tests", default=os.path.join(HERE, "tests.json"))
    parser.add_argument("--out", help="write the full results as JSON here")
    parser.add_argument("--calibrate", action="store_true",
                        help="write measured thresholds back into tests.json")
    parser.add_argument("--fast", action="store_true",
                        help="exam demo: run only the tests marked `fast` "
                             "with a short timeout")
    parser.add_argument("--install-only", action="store_true",
                        help="only copy the layouts into the project, so you "
                             "can run run.py by hand")
    args = parser.parse_args()
    if args.fast and args.calibrate:
        raise SystemExit("--fast runs a subset, so it cannot calibrate")

    if args.install_only:
        for number in ([0, 1, 2] if args.project == "all"
                       else [int(args.project)]):
            target = find_project(os.path.abspath(args.submission), number)
            if target is None:
                print("project %d: no run.py found" % number)
                continue
            print("project %d: installed %d layouts into %s" % (
                number, install_layouts(target, number), target))
        return

    with open(args.tests) as f:
        spec = json.load(f)

    numbers = [0, 1, 2] if args.project == "all" else [int(args.project)]
    results = []
    started = time.time()
    for number in numbers:
        part = spec["project%d" % number]
        results.append(grade_project(
            number, os.path.abspath(args.submission), args.reference,
            select_fast(part) if args.fast else part, args.calibrate))
        report(results[-1])
    if args.fast:
        print("\ndemo subset only, %d s elapsed. Full run: drop --fast"
              % round(time.time() - started))

    if args.calibrate:
        with open(args.tests, "w") as f:
            json.dump(spec, f, indent=2)
            f.write("\n")
        print("\ncalibrated thresholds written to %s" % args.tests)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"submission": args.submission, "results": results},
                      f, indent=2)
            f.write("\n")
        print("\nfull results written to %s" % args.out)


if __name__ == "__main__":
    main()
