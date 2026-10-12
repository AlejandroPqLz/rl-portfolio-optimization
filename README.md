# rl-portfolio-optimization

Goal-based dynamic asset allocation with deep reinforcement learning (MSc thesis, *Optimización de
Carteras con Aprendizaje por Refuerzo Profundo*). An investor contributes 400 USD a month for 120
months and wants to reach 60,000 USD. Each month a PPO agent with a Dirichlet policy splits wealth
across four assets: US equity, 10-year Treasury, 2-year Treasury and gold. The agent's policy is
compared with a dynamic-programming (DP) table, which is the optimal policy of the same problem on
a discretised grid. The DP solver is itself validated against Browne's (1999) closed form, and
everything is also scored against five heuristic portfolios. Scenarios come from a parametric
(GBM) generator or from a stationary block bootstrap of 1990–2024 monthly returns.

## Code freeze

Tag `freeze-2026-10-11` (11 October 2026, decisions D64/D81). No functional change is allowed to
the generator, environment, baselines, DP solver, training, evaluation, ETL or any config file.
Still allowed: `plots.py` (D73), this README, docstrings, comments and formatting.

Every allowed change is checked with these commands:

| Command | Expected outcome |
|---|---|
| `uv run pytest -q -m "not slow"` | 388 passed, 18 deselected |
| `uv run pytest -q -m slow` | 18 collected: 17 pass, 1 strict `xfail` (DP policy-stability criterion, 0.903 vs 0.95, D78) |
| `uv run ruff check src/ tests/` | All checks passed |
| `uv run python -m portfolio_manager.generator --test-set` | `match` for the three tensors (digests below) |
| `uv run python -m portfolio_manager.env --checks` | `PASS` on all five lines (`contributions_only`, `turnover_cost`, `bootstrap_bounds`, `interface`, `toy_path`) |
| `uv run python -m portfolio_manager.evaluate --reference-run` | `MATCH`, digest `887863fa1f0e2cd008687f614ebc67143bb9cd228a908a2a19483ccca9ee4509` |

Frozen test-set digests (`config/test_set.yml`, SHA-256 of the float64 C-order tensor bytes):

| Configuration | Seed | Digest |
|---|---|---|
| `A_gbm` | 20260809 | `9dd8165be1235dfb2e5752a3ad7581360a72084e12f51c617f9a4297065fea99` |
| `B_L12` | 20260810 | `aba55a058906a37b81716834119f73f7f8008b84d2c8460e2ea6c1ab16e5838b` |
| `B_L24` | 20260811 | `7aaee8856894f46ab5579e755ba42f4a7d46ee78400858a57b558324bd1f7267` |

These digests were obtained on the author's machine (macOS arm64, Python 3.12.13, NumPy 2.5.1,
Accelerate BLAS; full list in `results/environment/environment_fingerprint.json`). Other hardware or
library builds may differ at floating-point level. `generator --test-set` compares digests only
when the frozen environment descriptors match the running ones; otherwise it prints the
descriptors that differ and skips the comparison (D39).

The verification commands rewrite the `git_commit` and `written_at_utc` fields of their side-cars
under `results/`. To discard those changes, run `git checkout -- results/`.

## Requirements and setup

- Python 3.12 (`requires-python = "==3.12.*"` in `pyproject.toml`).
- [uv](https://docs.astral.sh/uv/). The lock file `uv.lock` pins every dependency.

```bash
uv sync
```

Every executable is a module: `uv run python -m portfolio_manager.<module> --help`.

## Reproduction map

The source for every number is `docs/09_REGISTRO_DE_RESULTADOS.md`. The commands below are the
ones recorded in each result's side-car (`command_line`). "Validation band" means seed 1000, the
default when `--seed` is omitted. The DP solver is deterministic and takes no seed.

| Thesis result | Results file | Command | Seed(s) |
|---|---|---|---|
| Historical matrix 420 × 4 | `data/returns_matrix.csv`, `.json` | `uv run python -m portfolio_manager.etl.download --refresh` then `uv run python -m portfolio_manager.etl.dataset.consolidate --build` | — |
| Mode A parameters | `config/gbm_params.yml` | `uv run python -m portfolio_manager.generator --fit-gbm` | — |
| Frozen test set | `results/test_set/<config>.npy`, `.json` | `uv run python -m portfolio_manager.generator --test-set` | 20260809 / 10 / 11 |
| Stylized facts, mode A | `results/checks/stylized_facts_A.json` | `uv run python -m portfolio_manager.generator --stylized-facts --mode A` | 1000 |
| Stylized facts, mode B, L = 12 | `results/checks/stylized_facts_B_12.json` | `uv run python -m portfolio_manager.generator --stylized-facts --mode B --block-length 12` | 1000 |
| Stylized facts, mode B, L = 24 | `results/checks/stylized_facts_B_24.json` | `uv run python -m portfolio_manager.generator --stylized-facts --mode B --block-length 24` | 1000 |
| Fixed-income report, φ = 0.656452 (D10 = 8.1336, D2 = 1.9232) | `config/baselines.yml` | `uv run python -m portfolio_manager.etl.dataset.consolidate --fixed-income-report` (needs `data/raw/`) | — |
| Environment checks (four thesis checks + toy path) | `results/checks/env_checks.json` | `uv run python -m portfolio_manager.env --checks` | 1000 (declared in code) |
| Heuristics table, B_L12, 10 bp | `results/heuristics_B_L12_10bp.csv`, `.json` | `uv run python -m portfolio_manager.evaluate --policies baselines --test-set B_L12 --cost-bp 10` | test 20260810; bootstrap 1000 |
| Reference-run guard | `results/reference_run.json` | `uv run python -m portfolio_manager.evaluate --reference-run` | 1000 |
| DP table 801 / 7, A_gbm | `results/dp/A_gbm.npz`, `.json` | `uv run python -m portfolio_manager.dp_solver --config A_gbm --stability` | — |
| DP stability, 801/7 vs 1601/9 | `results/checks/dp_stability.json` | same command as the DP table | — |
| Agent training, A_gbm, 0 bp | `results/runs/A_gbm_0bp/seed_<n>/` | `uv run python -m portfolio_manager.train --config A_gbm --seed <n> --cost-bp 0 --total-timesteps 10000000` | 0–4 |
| **H1**: A_gbm, trained 0 bp, evaluated 0 bp (agents vs DP table, McNemar) | `results/all_A_gbm_0bp.csv`, `.json` | `uv run python -m portfolio_manager.evaluate --policies all --test-set A_gbm --cost-bp 0` | test 20260809; bootstrap 1000 |
| Agent training, B_L12, 10 bp | `results/runs/B_L12_10bp/seed_<n>/` | `uv run python -m portfolio_manager.train --config B_L12 --seed <n> --total-timesteps 10000000` | 0–4 |
| **H2**: B_L12, 10 / 10 bp, McNemar vs glide path | `results/all_B_L12_10bp.csv`, `.json` | `uv run python -m portfolio_manager.evaluate --policies all --test-set B_L12 --cost-bp 10` | test 20260810; bootstrap 1000 |
| **H3**: terciles, B_L12, 10 / 10 bp | `results/terciles_B_L12_10bp.json` | `uv run python -m portfolio_manager.evaluate --terciles --policies all --test-set B_L12 --cost-bp 10` | test 20260810 |
| Terciles, A_gbm (0 bp and 10 bp) | `results/terciles_A_gbm_0bp.json`, `terciles_A_gbm_10bp.json` | `uv run python -m portfolio_manager.evaluate --terciles --policies all --test-set A_gbm --cost-bp 0` (or `10`) | test 20260809 |
| Agent training, B_L24, 10 bp | `results/runs/B_L24_10bp/seed_<n>/` | `uv run python -m portfolio_manager.train --config B_L24 --seed <n> --total-timesteps 10000000` | 0–4 |
| Block-length sweep: B_L24, 10 / 10 bp | `results/all_B_L24_10bp.csv`, `.json` | `uv run python -m portfolio_manager.evaluate --policies all --test-set B_L24 --cost-bp 10` | test 20260811; bootstrap 1000 |
| λ sweep: B_L12 agents trained at 10 bp, evaluated at 0 / 10 / 25 bp | `results/all_B_L12_train10bp_eval0bp.csv`, `all_B_L12_10bp.csv`, `all_B_L12_train10bp_eval25bp.csv` | `uv run python -m portfolio_manager.evaluate --policies all --test-set B_L12 --train-cost-bp 10 --cost-bp 0` (and `--cost-bp 25`) | test 20260810; bootstrap 1000 |
| Agent training, A_gbm, 10 bp (appendix variant) | `results/runs/A_gbm_10bp/seed_<n>/` | `uv run python -m portfolio_manager.train --config A_gbm --seed <n> --total-timesteps 10000000` | 0–4 (see note) |
| A_gbm, 10 / 10 bp (appendix variant) | `results/all_A_gbm_10bp.csv` | `uv run python -m portfolio_manager.evaluate --policies all --test-set A_gbm --cost-bp 10` | test 20260809; bootstrap 1000 |
| Browne validation, both variants (`no_leverage`, `leverage`) | `results/checks/browne_validation.json`, `results/dp/browne_{1,2,4}_per_month.npz` | `uv run python -m portfolio_manager.dp_solver --browne` | — |
| Figures | `results/figures/*.png` | `uv run python -m portfolio_manager.plots --all --config A_gbm --train-cost-bp 0` and `uv run python -m portfolio_manager.plots --all --config B_L12 --train-cost-bp 10 --table results/all_B_L12_10bp.csv` | — |

Notes:

- `results/agent_A_gbm_10bp.*`: 10 Oct artefact, 1,000 bootstrap resamples, written before the D85
  fix; superseded by `all_A_gbm_10bp.*` and kept unused.
- The `train` CLI accepts `--total-timesteps 10000000`. SB3 rounds this up to whole rollouts, so
  each run logs 10,010,624 steps.
- The side-car of `results/runs/A_gbm_10bp/seed_0/metrics.json` records `"shaping": true`. The
  diagnostic shaping run (`--shaping`, D74) was written to the same directory as the unshaped
  seed 0, so the 10 bp A_gbm results read that shaped policy for seed 0. H1 uses the 0 bp runs and
  is not affected.
- `--fixed-income-report` writes φ into `config/baselines.yml` only when the computed value differs
  from the stored one. After the freeze it must reproduce the stored value.
- Figures are pending A48 (see `CLAUDE.md`). `plots.py` is outside the freeze and its output is not
  covered by the digests above. Pass `--table` explicitly: without it the wealth figure reads the
  first CSV under `results/` in alphabetical order. The training-curve figure currently reads the
  base-case (10 bp) run directory whatever `--train-cost-bp` says.

## Single-command reproduction of the heuristics table

From the frozen B_L12 test set (seed 20260810, regenerated if missing and checked against its
digest), at 10 bp, with bootstrap seed 1000:

```bash
uv run python -m portfolio_manager.evaluate --policies baselines --test-set B_L12 --cost-bp 10 --seed 1000
```

This writes `results/heuristics_B_L12_10bp.csv` and its side-car. `--seed 1000` is the default and
is written out here only for explicitness. `--reference-run` scores the same five portfolios and
compares their digest against the recorded one.

## Repository layout

```
pyproject.toml, uv.lock       dependencies (src layout, hatchling build)
config/                       every problem, generator, comparator and agent number (no literals in code)
data/returns_matrix.csv|json  consolidated 420 x 4 monthly simple returns, 1990-01 to 2024-12, with metadata
data/raw/, data/processed/    downloaded series (git-ignored)
src/portfolio_manager/
  etl/download.py             raw series download and provenance manifest
  etl/dataset/                equity, fixed income (par-bond reconstruction), gold, consolidation
  generator.py                mode A (GBM) and mode B (stationary block bootstrap); test set; stylized facts
  env.py                      GoalPortfolioEnv (Gymnasium) and the shared transition step_wealth
  baselines.py                constant mix, glide path, Merton, equal weight, DP-table adapter
  dp_solver.py                backward induction on the wealth grid; stability; Browne validation
  browne.py                   Browne (1999) closed-form reference functions
  train.py                    PPO (SB3) with a Dirichlet policy
  evaluate.py                 metrics, bootstrap intervals, McNemar, terciles, reference run
  plots.py                    figures for section 4.4; reads results/, computes nothing
  environment_fingerprint.py  environment descriptors written into every side-car
tests/                        pytest; `slow` marks anything that trains or re-solves
scripts/browne_caps.py        A55 driver (Browne caps 3/10/30, restricted set); outside the package and the freeze
results/                      outputs with JSON side-cars (layout below)
docs/                         project documents (Spanish); not versioned
docs/memoria/                 thesis LaTeX sources
```

## Configuration files

| File | Content |
|---|---|
| `config/dataset.yml` | ETL only: window 1990-01 to 2024-12, duration model, raw filenames, bond maturities |
| `config/problem.yml` | Base case: goal 60,000 USD, 120 months, 400 USD/month, W₀ = 0, cost 10 bp, sweep 0/10/25 bp, objective LPM2 |
| `config/test_set.yml` | Frozen test-set recipe (frozen 2026-08-09): seeds, modes, block lengths, digests |
| `config/gbm_params.yml` | Mode A mean and covariance of monthly log gross returns, estimated once by `--fit-gbm` |
| `config/baselines.yml` | Comparator parameters (fixed-income split φ, glide path, Merton γ, equal weight). Externally sourced, `calibrated: false`, and never modified after a result was seen |
| `config/dp_solver.yml` | DP grid: 801 wealth nodes on [0, 4]·G, action step 0.05 (1,771 portfolios), 7 quadrature nodes; stability grid 1601/9 |
| `config/ppo.yml` | SB3 2.9.0 PPO defaults written out (D57), Dirichlet `alpha_min`, `n_envs`, evaluation cadence |
| `config/validation_browne.yml` | Separate configuration for validating the DP solver against Browne's closed form (D44) |

## `results/` layout

```
results/
  test_set/<config>.npy, .json        frozen tensors (npy git-ignored) and side-cars with digest
  checks/*.json                        stylized facts, env checks, DP stability, Browne validation
  dp/<name>.npz, .json                 V, policy, wealth grid and action grid, with side-car
  runs/<config>_<train_bp>bp/seed_<n>/ policy.zip (git-ignored), metrics.json, tensorboard/ (git-ignored)
  heuristics_<test_set>_<bp>bp.csv     comparators only
  all_<test_set>_<bp>bp.csv            every policy, when the training and evaluation costs are equal
  all_<test_set>_train<a>bp_eval<b>bp.csv   every policy, when the two costs differ
  terciles_<test_set>_<bp>bp.json      H3 split by progress at mid-horizon
  reference_run.json                   digest guarded by --reference-run
  environment/environment_fingerprint.json
  figures/*.png
  batch_logs/*.log                     logs of the unattended training batches
```

Each CSV, NPY and NPZ file has a JSON side-car that records the command line, seed, git commit,
environment fingerprint, the SHA-256 of the input data and the config files read. For evaluation
tables the side-car also records `training_cost_bp` and `evaluation_cost_bp`.
