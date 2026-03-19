# Contributing

Thanks for your interest in improving the public COMET model repository.

## Scope

This repository is intended to host the training-ready model architecture and minimal usage examples. Please keep contributions focused on:

- architecture improvements
- reproducibility improvements
- input validation
- documentation
- training utilities that do not depend on private datasets

## Before opening a pull request

1. Keep changes small and well-scoped.
2. Make sure the minimal training example still runs.
3. Run the unit tests:

```bash
python -m unittest discover -s tests
```

4. If you change the public API, update `README.md`.

## Coding guidelines

- Use Python 3.10+.
- Keep modules independent from project-specific paths.
- Do not commit trained weights, raw data, or private clinical metadata.
- Prefer explicit tensor-shape documentation in public interfaces.

