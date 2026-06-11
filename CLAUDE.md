# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is the official PyTorch implementation of **DAG: A Dual Causal Network for Time Series Forecasting with Exogenous Variables** ([paper](https://arxiv.org/pdf/2509.14933)). DAG is one model among many baselines; the surrounding code is a fork of the **TFB (Time Series Forecasting Benchmark)** evaluation framework. Most of `ts_benchmark/` is benchmark plumbing, not DAG-specific code.

## Environment

- Use the **`TFB` conda env** on this machine: `/opt/conda/envs/TFB/bin/python` (Python 3.8.20, torch 2.4.1+cu121, all deps already installed). Do not use the system `/opt/conda/bin/python` (3.11) — its `dash` import is broken and `requirements.txt` is pinned for 3.8 anyway.
  - Quick activate: `conda activate TFB`, or just call `/opt/conda/envs/TFB/bin/python ./scripts/run_benchmark.py ...` directly.
- Python **3.8** (the project is "fully tested" only on 3.8 — newer versions are not validated).
- Install deps with `pip install -r requirements.txt`. Pinned key versions: `torch==2.4.1`, `numpy==1.24.4`, `pandas==1.5.3`, `darts==0.25.0`, `salesforce_merlion==2.0.2`, `ray==2.10.0`.
- Datasets are not in the repo. Download the pre-processed bundle from the Google Drive link in `README.md` and unpack into `./dataset/forecasting/` (must include `FORECAST_META.csv` plus per-dataset CSVs like `NP.csv`, `ETTh1.csv`, etc.). `ts_benchmark/common/constant.py` resolves this path relative to the repo root.

## Common commands

Everything goes through `scripts/run_benchmark.py`. There is no Makefile, no test suite, and no linter configured — verification means running the benchmark on a small dataset/horizon and inspecting `result/<save-path>/`.

Reproduce the DAG paper results (uncomment the dataset/horizon combo you want inside the file):

```shell
sh ./scripts/covariate_forecasting/DAG.sh
```

Run a single experiment directly (DAG on NP with horizon 24):

```shell
python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "NP.csv" \
    --strategy-args '{"horizon": 24, "target_channel": [-1]}' \
    --deterministic "full" \
    --model-name "dag.DAG" \
    --model-hyper-params '{"alpha": 0.9, "batch_size": 64, "d_ff": 128, "d_model": 64, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 168, "stride": 48, "use_c": 1, "use_c_exog": 1, "use_t": 1, "use_t_exog": 1}' \
    --gpus 1 --num-workers 1 --timeout 60000 --save-path "NP/DAG"
```

Other baselines have their own scripts under `scripts/covariate_forecasting/` (`Amplifier.sh`, `CrossLinear.sh`, `DUET.sh`, `PatchTST.sh`, `TimeXer.sh`, `xPatch.sh`, etc.) — same CLI, different `--model-name`.

Key CLI arguments (see `scripts/run_benchmark.py` for the full list):

- `--config-path` — JSON file under `config/` (e.g. `rolling_forecast_config.json`, `fixed_forecast_config_*.json`). Provides defaults for data/model/evaluation/report sections.
- `--data-name-list` — one or more CSV filenames from `dataset/forecasting/`.
- `--model-name` — dotted path inside `ts_benchmark/baselines/` (e.g. `dag.DAG`, `timexer.TimeXer`).
- `--model-hyper-params` — JSON string merged onto the model's `MODEL_HYPER_PARAMS` defaults.
- `--strategy-args` — JSON overriding `evaluation_config.strategy_args` (notably `horizon` and `target_channel`, where `[-1]` means "predict only the last column, treat the rest as exogenous").
- `--gpus` — single GPU index (passed to `ParallelBackend`).
- `--save-path` — output subfolder under `result/`.
- `--deterministic` — `full` or `efficient`; `full` reseeds aggressively.

`script.sh` and `1generate.sh` / `1generate_wide.sh` are user-side scaffolding for hyperparameter sweeps (they generate `test_10.24/<dataset>/e<i>.sh` shards). They are not required to run a single experiment.

## Architecture

### Pipeline flow

`scripts/run_benchmark.py` is the single entry point. It:

1. Loads a base JSON config from `config/`, then layers CLI overrides on top via `build_data_config` / `build_model_config` / `build_evaluation_config` / `build_report_config`.
2. Initializes `ParallelBackend` (`ts_benchmark/utils/parallel/`) — either `ray_backend` or `sequential_backend`. Workers get `init_worker` to add `THIRD_PARTY_PATH` to `sys.path` and pin `torch.set_num_threads(1)`.
3. Calls `ts_benchmark.pipeline.pipeline(...)`, which:
   - Resolves the dataset via `PREDEFINED_DATASETS` (`large_forecast`, `small_forecast`, `user_forecast`) → a `LocalForecastingDataSource` reading `dataset/forecasting/FORECAST_META.csv`.
   - Spawns a `GlobalStorageDataServer` so workers can fetch series without re-reading disk.
   - Builds models with `get_models(model_config)` (resolves `--model-name` like `dag.DAG` to `ts_benchmark.baselines.dag.dag.DAG` via `ts_benchmark/models/model_loader.py`).
   - Runs `eval_model(...)` (in `ts_benchmark/evaluation/evaluate_model.py`) per (model, dataset) pair under the configured strategy (`rolling_forecast` or `fixed_forecast` in `ts_benchmark/evaluation/strategy/`).
   - Persists per-run logs via `recording.save_log` and writes a leaderboard via `ts_benchmark/report`.

### Adding / modifying a model

All deep-learning baselines extend `ts_benchmark.baselines.deep_forecasting_model_base.DeepForecastingModelBase`. The base class owns the data pipeline, optimizer, AMP, early stopping (`patience`), LR schedules (`lradj` types `type1` / `type2` / `type3` / `constant` defined in `baselines/utils.py`), and checkpointing. A subclass typically only implements:

- `_init_model()` — return the `nn.Module`.
- `_process(input, target, input_mark, target_mark, exog_future=None)` — forward pass, returning `{"output": ..., "additional_loss": ...}` (the `additional_loss` is added to the main loss during training).
- `_init_criterion()` — optional, picks `MSE` / `MAE` / `DBLoss` / Huber from `self.config.loss`.
- A class-level `MODEL_HYPER_PARAMS` dict for defaults; the `Config` object in the base merges defaults → `MODEL_HYPER_PARAMS` → user overrides, then aliases `horizon` → `pred_len` (with a deprecation warning — new code should set `pred_len`).

### DAG model layout

DAG-specific code lives in `ts_benchmark/baselines/dag/`:

- `dag.py` — the `DAG(DeepForecastingModelBase)` adapter and its `MODEL_HYPER_PARAMS`.
- `models/dag_model.py` — the top-level `DAGModel` that combines a `TemporalCausalityEncoder` and a `CovCausalityEncoder`, mixes their outputs as `alpha * t_output + (1 - alpha) * c_output`, and returns `causality_loss = beta * (temporal + cov)` during training.
- `layers/TC_EncDec.py` — Temporal Causal Module (causal discovery over historical exogenous → future exogenous, then injects into endogenous forecast).
- `layers/CC_EncDec.py` — Channel Causal Module (causal discovery between historical exogenous and historical endogenous).
- `layers/Embed.py`, `layers/SelfAttention_Family.py`, `layers/Transformer_EncDec.py` — supporting building blocks (largely lifted from the time-series-library style).
- `utils/tools.py` — DAG-only helpers.

Toggles `use_t`, `use_c`, `use_t_exog`, `use_c_exog` ablate the two modules and their use of exogenous inputs. At inference time, `infer_use_future=True` lets the temporal encoder consume the *known* future exogenous variables; setting it to `False` falls back to predicting them and using the predictions.

### Configs

`config/rolling_forecast_config.json` is the workhorse for covariate forecasting. Notable fields:

- `evaluation_config.strategy_args.target_channel: [-1]` — only the last column is the endogenous target; everything else is exogenous. CLI passes this per run.
- `train_ratio_in_tv` — per-dataset train/val split inside the train+val portion. ETT*, PEMS*, AQ*, Solar use 0.75; everything else falls back to `__default__: 0.875`.
- `report_metrics` — normalized variants (`mse_norm`, `mae_norm`, `wape_norm`, `msmape_norm`, …) computed against the StandardScaler-normalized space.

`config/fixed_forecast_config_*.json` are for fixed-horizon (non-rolling) protocols by frequency.

### Result layout

Each run writes a per-(model, dataset, horizon) row into `result/<save-path>/...`. `report` then aggregates rows into a `test_report*.csv` leaderboard (`--report-method csv`). `1result.csv` at the repo root and the `result/12.x` / `result/1.5` subfolders are author scratch outputs from prior runs — safe to ignore for a fresh experiment.

## Things to watch out for

- Some baselines duplicate code as zip archives next to their packages (e.g. `ts_benchmark/baselines/dag.zip`, `timexer.zip`). The live code is the unzipped directory; treat the zips as historical snapshots.
- `nohup.out` (~190 MB) is committed and is a leftover training log — don't read it through Read; grep what you need.
- The `horizon` model hyperparameter is deprecated in favor of `pred_len`, but every script in `scripts/covariate_forecasting/` still passes `horizon`. The `Config` class converts it transparently.
- `--num-workers 1` plus `--gpus N` (a single index, not a count) is the path used by the maintained scripts. Multi-GPU through `ParallelBackend`'s ray mode exists but isn't exercised by the DAG scripts.
- `THIRD_PARTY_PATH` (`ts_benchmark/baselines/third_party`) is added to `sys.path` for both the main process and workers; vendored dependencies live there if present.
