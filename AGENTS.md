# Repository Guidelines

## Project Structure & Module Organization

- Entry points: `demo.py` (Flask GUI), `inference.py` (CLI inference), `train.py` (training).
- Configuration: `configs/*.yaml` (OmegaConf configs used by training/inference).
- Core code: `pipeline/` (training/inference pipelines), `model/` (model variants), `trainer/` (trainer loops), `utils/` (helpers/wrappers).
- Research tooling: `experiments/` (analysis scripts + pytest), `notebooks/` (Jupyter).
- Third-party code: `wan/` (modified from Wan2.1, Apache-2.0) — keep changes minimal and well-scoped.
- Assets/outputs: `templates/` (GUI HTML), `images/`, `videos/` (generated outputs; ignored by git).

## Build, Test, and Development Commands

- Install deps (recommended): `uv sync`
- Install FlashAttention (GPU/CUDA toolchain required): `uv pip install flash-attn --no-build-isolation`
- Run GUI demo: `python demo.py`
- Run CLI inference (example): `python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path checkpoints/self_forcing_dmd.pt --data_path prompts/MovieGenVideoBench_extended.txt --output_folder videos/run`
- Distributed training (example): `torchrun --nnodes=... --nproc_per_node=... train.py --config_path configs/self_forcing_dmd.yaml --logdir logs/self_forcing_dmd`

## Coding Style & Naming Conventions

- Python 3.10; use 4-space indentation and keep functions/modules small and composable.
- Formatting/imports: `black .` and `isort .` (both are project dependencies).
- Naming: `snake_case` for files/functions, `CamelCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Configs: keep names descriptive and stable (e.g., `configs/self_forcing_dmd.yaml`); avoid breaking existing paths used in docs/scripts.

## Testing Guidelines

- Test runner: `pytest` (tests currently live under `experiments/`).
- Run the main test file: `PYTHONPATH=. pytest experiments/test_attention_extraction.py -v -s`
- Some tests require CUDA; to skip GPU-dependent cases: `PYTHONPATH=. pytest experiments/test_attention_extraction.py -v -s -k "not cuda"`

## Commit & Pull Request Guidelines

- Commits follow a lightweight conventional style seen in history: `feat: ...`, `fix: ...`, `docs: ...`, `chores: ...`, `test: ...`.
- Keep messages imperative and specific; include the config/script you touched when relevant.
- PRs should include: a short problem/solution summary, commands/configs used to reproduce (e.g., `configs/*.yaml`), and any output artifacts/metrics (VRAM, speed, sample video path). Call out any edits under `wan/` explicitly.

## Security & Large Files

- Do not commit checkpoints/models or generated media: `checkpoints/`, `wan_models/`, `videos/`, `logs/`, `wandb/`, `cache/`, `figures/`, and `*.pt/*.pth/*.safetensors` are intentionally ignored via `.gitignore`.
