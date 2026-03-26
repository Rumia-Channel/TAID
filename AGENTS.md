# Repository Guidelines

## Project Structure & Module Organization
`train.py` is the main training entry point. Core implementation lives in `src/`: argument parsing in `src/arguments.py`, data loading in `src/data.py`, the Lightning model in `src/model.py`, shared helpers in `src/utils.py`, and distillation losses in `src/distil_losses/`. Dataset preparation utilities such as `prepare_ultrachat.py` and checkpoint conversion helpers such as `convert_l_to_hf.py` live at the repository root. Experiment launchers are organized under `scripts/<teacher>/` with one shell script per method, for example `scripts/phi-3/taid.sh`.

## Build, Test, and Development Commands
Install dependencies with `uv`:
```bash
uv sync --extra cpu
uv sync --extra cuda
uv sync --extra rocm
uv sync --extra xpu
```
`rocm` is Linux-only, and `deepspeed` should be added only for Linux multi-GPU CUDA training: `uv sync --extra cuda --extra deepspeed`. Prepare data with `uv run python prepare_ultrachat.py --model_type phi-3 --output_dir data`. Run training with a provided launcher such as `uv run bash scripts/phi-3/taid.sh`, or call `uv run python train.py` directly when iterating on flags.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions, variables, and CLI flags, and `PascalCase` for classes such as `KDForLM`. Keep modules small and task-focused. Match the current import style and prefer explicit argument names over positional-only training configuration. There is no formatter config in the repo, so keep changes consistent with surrounding code and avoid unrelated style churn.

## Testing Guidelines
There is no committed automated test suite yet. Treat validation as script-driven smoke testing: run `uv lock`, then a relevant `uv sync --extra ...`, confirm argument parsing works, and verify that training or validation starts cleanly. For data-path changes, run `uv run python prepare_ultrachat.py` on a small sample first. Document the exact command you used in the PR when changing training, sampling, or loss behavior.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `Update README.md` and `fix blog links`. Keep commit subjects concise and specific to one change. PRs should include a short description, impacted scripts or modules, any required environment assumptions, and representative logs or screenshots when behavior changes. Link the related issue or experiment note when available.

## Configuration Tips
Do not commit Hugging Face, Weights & Biases, or dataset credentials. Keep generated datasets under a local `data/` directory and training outputs under `logs/`, and leave large artifacts out of version control.
