# Agent Guidelines

> **Keep this file concise.** Be specific but brief. Update as you learn.

**Required reading:** [Contributing.md](Contributing.md) for code structure, workflow, and standards.

## Code Rules

- **Match existing style** - follow patterns in surrounding code
- **Single Responsibility, DRY, KISS, Fail Fast**
- **Type hints** on all functions, **docstrings** on public APIs
- **Never hardcode** paths/hyperparameters - use `config.yaml`
- **Use existing utilities** from `utils/` before writing new code

## Testing

- **Use small datasets** (`Month_1.csv`, `cv_folds: 3`)
- **Use `run_evaluation()`** not `run_grid_search_experiment()` for quick tests
- **Use GPU** (`device: "cuda"` or `"mps"`) when available
- Embeddings are cached - first run is slow, subsequent runs are fast

## Editing config.yaml

- Change **only data values** - preserve structure and formatting
- Use `uv run python` to execute scripts

## Learned Lessons

Update this section when you encounter issues or discover better practices.

- Use `uv run python` instead of `python` (respects venv)
- CPU embedding generation is slow (~2 samples/sec) - use GPU or small datasets
- `/dev/null` discards data - doesn't consume storage

### Code Style Patterns
<!-- Add patterns observed in this codebase as you work -->
