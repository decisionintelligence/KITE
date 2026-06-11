# Repository Guidelines

## Project Structure & Module Organization
`ts_benchmark/` contains the core benchmark framework: data loading, evaluation, reporting, utilities, and model adapters. Baseline implementations, including DAG, live under `ts_benchmark/baselines/`; DAG-specific code is in `ts_benchmark/baselines/dag/`. Entry-point scripts are in `scripts/`, with reproducible forecasting runs in `scripts/covariate_forecasting/`. Config files live in `config/`, datasets in `dataset/forecasting/`, generated outputs in `result/`, and paper figures in `docs/figures/`. The `test_10.24/` tree is experiment scaffolding and logs, not a formal unit test suite.

## Build, Test, and Development Commands
Use Python 3.8 for local work.

```bash
pip install -r requirements.txt
```
Installs pinned dependencies.

```bash
python scripts/run_benchmark.py --config-path rolling_forecast_config.json --data-name-list NP.csv --model-name dag.DAG --gpus 0 --num-workers 1 --save-path NP/DAG
```
Runs a single benchmark job.

```bash
sh scripts/covariate_forecasting/DAG.sh
```
Launches the repository's scripted DAG experiments.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for functions/modules/variables, PascalCase for classes, and JSON keys that match current config names such as `model_name` and `strategy_args`. Keep new code inside the existing benchmark layers instead of adding parallel entry points. This checkout does not include configured formatters or linters, so match surrounding code and keep imports, CLI flags, and config structures consistent.

## Testing Guidelines
There is no dedicated `tests/` package or pytest configuration in this snapshot. Validate changes by running a small benchmark job and checking the generated CSV report under `result/<save-path>/`. For script additions, keep names descriptive and aligned with existing patterns such as `DAG.sh` or `TimeXer.sh`.

## Commit & Pull Request Guidelines
Git history is not available in this checkout, so use short imperative commit subjects, for example: `Add DAG benchmark contributor guide`. Keep commits scoped to one change. Pull requests should include the benchmark setting affected, config or script changes, sample output paths under `result/`, and screenshots only when UI or report rendering changes are involved. Link the related issue or experiment request when applicable.

## Configuration & Data Tips
Keep datasets under `dataset/forecasting/` and do not commit large generated artifacts unless the repository already tracks that output intentionally. Prefer editing JSON configs in `config/` or shell wrappers in `scripts/covariate_forecasting/` instead of hardcoding paths in library modules.
