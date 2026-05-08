#!/usr/bin/env bash
# Run once after cloning to activate the pre-commit hooks.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
git -C "$root" config --unset core.hooksPath 2>/dev/null || true
pre-commit install
echo "Git hooks installed via pre-commit."
