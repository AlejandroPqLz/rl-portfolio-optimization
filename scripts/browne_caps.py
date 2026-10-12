"""A55 (D89): the Browne leverage variant at caps 3, 10 and 30, with artefacts and the restricted set.

Outside the package and outside the freeze. It changes nothing under ``src/`` or ``config/``: the
validation configuration is parsed once, deep-copied, and only ``leverage.max_equity_weight`` is
set in memory. Everything else, ``grid_step`` and every tolerance included, is the file's. The
loaders and keyword arguments are those of ``run_browne_validation`` (``dp_solver.py``), which is
not called because it would also rerun the no-leverage variant and rewrite
``results/checks/browne_validation.json``; that file is never written here.

On top of the document ``run_leverage_variant`` returns, each cap gets a ``restricted_set`` key:
the value and policy errors on the states with at least six months remaining and a closed-form
fraction ``w* <= 3``, the set on which the prediction of A50 was written. It is scored per
refinement level with the package's own ``solve_leverage``, ``leverage_comparison_set`` and
``score_browne_detail``; the only addition is the filter.

Writes ``results/checks/browne_cap<cap>/browne_leverage.json`` and ``provenance.json`` per cap and
refuses to overwrite either.

    uv run python scripts/browne_caps.py
"""

import copy
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from portfolio_manager.browne import optimal_fraction
from portfolio_manager.browne import value as browne_value
from portfolio_manager.dp_solver import (
    browne_setup,
    leverage_comparison_set,
    load_browne_config,
    load_solver_config,
    run_leverage_variant,
    score_browne_detail,
    solve_leverage,
)
from portfolio_manager.env import load_problem
from portfolio_manager.generator import git_commit, load_gbm_params, load_matrix

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "validation_browne.yml"
CHECKS = ROOT / "results" / "checks"
CAPS = (3, 10, 30)

# The restricted set of the A50 prediction, written on 11 October before the first run.
RESTRICTED_MIN_MONTHS = 6.0
RESTRICTED_MAX_REFERENCE_FRACTION = 3.0


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def restricted_level(browne, *, config, problem, closed_form, equity_column, riskless_column, level) -> dict:
    """Errors on the restricted set at one refinement level."""
    steps, nodes = int(level["steps_per_month"]), int(level["n_wealth"])
    solution = solve_leverage(browne, config=config, problem=problem, steps_per_month=steps, n_wealth=nodes)
    node, step, months = leverage_comparison_set(solution, closed_form, steps)
    wealth = solution.wealth_grid[node]
    reference = optimal_fraction(wealth, months, closed_form)
    keep = (months >= RESTRICTED_MIN_MONTHS) & (reference <= RESTRICTED_MAX_REFERENCE_FRACTION)
    node, step, months, wealth, reference = node[keep], step[keep], months[keep], wealth[keep], reference[keep]

    detail = score_browne_detail(
        solution,
        closed_form,
        node,
        step,
        months,
        equity_column,
        riskless_column,
        max_equity_weight=float(browne["leverage"]["max_equity_weight"]),
    )
    # Percentiles are not in score_browne_detail; the errors are recomputed with the same formulas.
    value_error = np.abs(solution.value[node, step] - browne_value(wealth, months, closed_form))
    policy_error = np.abs(solution.policy[node, step, equity_column] - reference)
    return {
        "steps_per_month": steps,
        "n_wealth": nodes,
        "value_p90_error": float(np.quantile(value_error, 0.9)),
        "policy_p90_error": float(np.quantile(policy_error, 0.9)),
        "ruin_reached": bool(solution.ruin_reached),
        **detail,
    }


def main() -> int:
    browne = load_browne_config()
    config, problem = load_solver_config(), load_problem()
    columns, _ = load_matrix()
    _, _, closed_form = browne_setup(browne, columns, load_gbm_params())
    closed_form.require_paper_assumption()
    equity_column = columns.index(str(browne["assets"]["risky"]))
    riskless_column = columns.index(str(browne["assets"]["riskless"]))
    yaml_sha256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    shared = {
        "config": config,
        "problem": problem,
        "closed_form": closed_form,
        "equity_column": equity_column,
        "riskless_column": riskless_column,
    }

    rows = {}
    for cap in CAPS:
        directory = CHECKS / f"browne_cap{cap}"
        targets = (directory / "browne_leverage.json", directory / "provenance.json")
        if any(path.exists() for path in targets):
            raise FileExistsError(f"{directory} already holds results; refusing to overwrite")

        settings = copy.deepcopy(browne)
        settings["leverage"]["max_equity_weight"] = float(cap)
        print(f"[{now()}] cap {cap}: start", flush=True)
        start = time.perf_counter()
        variant = run_leverage_variant(settings, **shared)
        for level in variant["levels"]:
            print(
                f"[{now()}] cap {cap}: level {level['steps_per_month']}/month, {level['n_wealth']} nodes, "
                f"{level['n_actions']} actions, {level['seconds_elapsed']:.1f} s",
                flush=True,
            )
        restricted = []
        for level in settings["refinement"]:
            restricted.append(restricted_level(settings, level=level, **shared))
            print(f"[{now()}] cap {cap}: restricted set at {level['steps_per_month']}/month", flush=True)
        elapsed = time.perf_counter() - start
        variant["restricted_set"] = {
            "rule": (
                f"0 < z < 1, at least {RESTRICTED_MIN_MONTHS:g} months remaining and closed-form fraction "
                f"w* <= {RESTRICTED_MAX_REFERENCE_FRACTION:g} (A50 prediction, written 2026-10-11)"
            ),
            "min_months_remaining": RESTRICTED_MIN_MONTHS,
            "max_reference_fraction": RESTRICTED_MAX_REFERENCE_FRACTION,
            "source": "scripts/browne_caps.py, outside the package (A55, D89)",
            "levels": restricted,
            "finest": restricted[-1],
        }
        print(f"[{now()}] cap {cap}: done in {elapsed:.1f} s", flush=True)

        directory.mkdir(parents=True, exist_ok=True)
        targets[0].write_text(json.dumps(variant, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
        provenance = {
            "cap": cap,
            "grid_step": float(settings["leverage"]["grid_step"]),
            "n_quad": int(settings["n_quad"]),
            "refinement": settings["refinement"],
            "closed_form_parameters": {
                "r": closed_form.r,
                "mu": closed_form.mu,
                "sigma": closed_form.sigma,
                "theta": closed_form.theta,
                "premium": closed_form.premium,
            },
            "validation_browne_yml_sha256": yaml_sha256,
            "override": {"leverage.max_equity_weight": float(cap)},
            "seconds_elapsed": elapsed,
            "git_commit": git_commit(),
            "command_line": sys.argv,
            "written_at_utc": now(),
        }
        targets[1].write_text(json.dumps(provenance, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
        rows[cap] = (variant, elapsed)

    print("\nfinest pair: restricted set (>= 6 months, w* <= 3) | whole set | floor binding")
    for cap, (variant, elapsed) in rows.items():
        r = variant["restricted_set"]["finest"]
        whole = variant["levels"][-1]
        print(
            f"cap {cap:>2}: value mean {r['value_mean_error']:.4f} p90 {r['value_p90_error']:.4f} "
            f"max {r['value_max_error']:.4f} | policy mean {r['policy_mean_error']:.4f} "
            f"p90 {r['policy_p90_error']:.4f} max {r['policy_max_error']:.4f} | n {r['n_nodes']} | "
            f"whole value mean {whole['value_mean_error']:.4f} | floor {variant['bankruptcy_floor']['binding']} | "
            f"{elapsed:.0f} s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
