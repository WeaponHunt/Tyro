# AGENTS.md

## Project Notes

This is a Python project for the Tyro/TalkRobot assistant. Prefer small, focused changes that follow the existing module layout under `talkrobot/`.

## Environment

Use the Conda environment named `robot_sys` for local checks and test runs:

```bash
conda activate robot_sys
```

When running non-interactively, prefer:

```bash
conda run -n robot_sys <command>
```

## Common Checks

Run syntax/import compilation for touched Python files:

```bash
conda run -n robot_sys python -m compileall <files>
```

Run project tests from the repository root when needed:

```bash
conda run -n robot_sys python -m pytest talkrobot/tests
```

If unrelated globally installed pytest plugins interfere, disable plugin autoload:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n robot_sys python -m pytest talkrobot/tests
```

Some tests may require audio devices, model files, API keys, or local services. If a check depends on those external resources, note the missing requirement instead of hiding the failure.

## Logging

Runtime logs are written under `talkrobot/logs/`. Per-interaction JSONL logs are written under `talkrobot/logs/interactions/` and are split by day.
