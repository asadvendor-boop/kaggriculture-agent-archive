"""gcov report of the verifier's coverage build: which lines / branches of the rules code were never executed.

usage (server): python cov_report.py <gcda_root> <out.json> [--listing out.txt]
  <gcda_root>: a GCOV_PREFIX directory (the .gcda sits under it at the object's absolute path)
Runs `gcov --json-format` on a copy of (gcno, gcda) and reports, for sim_trace.hpp / pyrandom.hpp /
kagsim_trace.cpp, per function: line coverage, branch coverage, and every never-executed line and never-taken branch
(the trace-receipt code, which only runs with trace=True, and the KAG_ABLATE seam are listed separately).
"""
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

BUILD = "/Users/asad.ali/kaggriculture/kagjax/verify/cov_build/cov"
SRC = "/Users/asad.ali/kaggriculture/engine/src_v2"
STEM = "kagsim_trace.cpython-312-x86_64-linux-gnu.so-kagsim_cov"


def run_gcov(gcda_root):
    gcda = glob.glob(os.path.join(gcda_root, "**", STEM + ".gcda"), recursive=True)
    assert len(gcda) == 1, gcda
    tmp = tempfile.mkdtemp()
    shutil.copy(os.path.join(BUILD, STEM + ".gcno"), tmp)
    shutil.copy(gcda[0], tmp)
    subprocess.run(["gcov", "--json-format", "-b", "-c", "-o", tmp, os.path.join(tmp, STEM + ".gcda")], cwd=tmp,
                   check=True, capture_output=True)
    out = glob.glob(os.path.join(tmp, "*.gcov.json.gz"))
    assert out, os.listdir(tmp)
    data = json.load(gzip.open(out[0]))
    shutil.rmtree(tmp)
    return data


def trace_lines():
    """Line numbers of sim_trace.hpp that are trace-only (TRACE PATCH blocks guarded by trace / cur) or the ablation
    seam; they never run in a normal (trace=False) game."""
    src = open(os.path.join(SRC, "sim_trace.hpp")).read().split("\n")
    skip = set()
    # KAG_ABLATE seam
    for i, l in enumerate(src, 1):
        if "#ifdef KAG_ABLATE" in l:
            j = i
            while "#else" not in src[j - 1]:
                skip.add(j); j += 1
    # blocks opened by `if (trace)`, `if (cur)`, `if (cur && ...)`, `if (!trace) {...; continue; }` tail
    i = 0
    while i < len(src):
        l = src[i]
        s = l.strip()
        opener = (s.startswith("if (cur") or s.startswith("if (trace)") or s.startswith("if (trace &&")
                  or "TRACE PATCH: exact per-unit receipt" in s or s.startswith("if (cur &&"))
        if opener and "{" in l:
            depth = 0
            j = i
            while True:
                depth += src[j].count("{") - src[j].count("}")
                skip.add(j + 1)
                if depth <= 0 and j > i:
                    break
                if depth <= 0 and j == i and "}" in src[j]:
                    break
                j += 1
            i = j + 1
            continue
        if opener:                                    # single-statement if (next line)
            skip.add(i + 1); skip.add(i + 2)
        i += 1
    return skip


def analyse(data):
    rep = {}
    for f in data["files"]:
        name = os.path.basename(f["file"])
        if name not in ("sim_trace.hpp", "pyrandom.hpp", "kagsim_trace.cpp", "kagsim_cov.cpp"):
            continue
        src = open(f["file"]).read().split("\n")
        fr = rep.setdefault(name, {"lines": {}, "functions": []})
        for fn in f.get("functions", []):
            fr["functions"].append({"name": fn.get("demangled_name", fn["name"]), "start": fn["start_line"],
                                    "end": fn["end_line"], "count": fn["execution_count"],
                                    "blocks": fn["blocks"], "blocks_executed": fn["blocks_executed"]})
        for ln in f["lines"]:
            n = ln["line_number"]
            e = fr["lines"].setdefault(n, {"count": 0, "branches": None, "unexec_block": False,
                                           "src": src[n - 1] if n - 1 < len(src) else ""})
            e["count"] += ln["count"]
            e["unexec_block"] = e["unexec_block"] or ln.get("unexecuted_block", False)
            br = [(b["count"], b.get("throw", False)) for b in ln.get("branches", [])]
            if br:
                if e["branches"] is None:
                    e["branches"] = [list(x) for x in br]
                else:                                  # template instantiations: merge per index
                    for k, (c, t) in enumerate(br):
                        if k < len(e["branches"]):
                            e["branches"][k][0] += c
                        else:
                            e["branches"].append([c, t])
    return rep


def main():
    root, out = sys.argv[1], sys.argv[2]
    listing = sys.argv[sys.argv.index("--listing") + 1] if "--listing" in sys.argv else None
    rep = analyse(run_gcov(root))
    skip = trace_lines()
    summary = {}
    lines_out = []
    for name, fr in rep.items():
        L = fr["lines"]
        rules = {n: e for n, e in L.items() if not (name == "sim_trace.hpp" and n in skip)}
        nl = len(rules)
        nl_hit = sum(e["count"] > 0 for e in rules.values())
        nb = sum(len([b for b in e["branches"] if not b[1]]) for e in rules.values() if e["branches"])
        nb_hit = sum(sum(1 for b in e["branches"] if not b[1] and b[0] > 0) for e in rules.values() if e["branches"])
        summary[name] = {"lines": nl, "lines_hit": nl_hit, "branches": nb, "branches_taken": nb_hit,
                         "line_pct": round(100 * nl_hit / max(nl, 1), 2), "branch_pct": round(100 * nb_hit / max(nb, 1), 2)}
        for n in sorted(rules):
            e = rules[n]
            nt = [k for k, b in enumerate(e["branches"] or []) if not b[1] and b[0] == 0]
            if e["count"] == 0 or nt:
                lines_out.append({"file": name, "line": n, "count": e["count"], "never_taken_branches": nt,
                                  "branch_counts": [b[0] for b in (e["branches"] or []) if not b[1]],
                                  "src": e["src"].rstrip()})
    json.dump({"summary": summary, "uncovered": lines_out,
               "functions": {k: v["functions"] for k, v in rep.items()}}, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if listing:
        with open(listing, "w") as fh:
            for x in lines_out:
                fh.write(f"{x['file']}:{x['line']:5d} count={x['count']:<10} nt={x['never_taken_branches']} "
                         f"br={x['branch_counts']} | {x['src']}\n")
    print(len(lines_out), "lines with an unexecuted statement or a never-taken branch")


if __name__ == "__main__":
    main()
