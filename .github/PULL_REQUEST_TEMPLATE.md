## What this changes and why

<!-- One or two paragraphs. Link the issue it closes, if any. -->

## Checklist

- [ ] `cmake --build build/dev --target check-pulse` passes (lit and pytest).
- [ ] `pre-commit run --all-files` and `mypy -p cudaq_pulse` pass.
- [ ] New behavior has tests; GPU-only changes say how they were tested on a GPU.
- [ ] User-visible changes have an entry under `## [Unreleased]` in `CHANGELOG.md`.
- [ ] Breaking changes follow the versioning policy in `CONTRIBUTING.md`.
