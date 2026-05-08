#!/usr/bin/env bash
set -euo pipefail

if ! command -v clang-tidy &>/dev/null; then
    echo "WARNING: clang-tidy not found, skipping."
    exit 0
fi

if [[ ! -f build/compile_commands.json ]]; then
    echo "WARNING: build/compile_commands.json not found."
    echo "  Run: cmake -S . -B build -DBUILD_TESTS=ON && cmake --build build"
    exit 0
fi

exec clang-tidy -p build --quiet "$@"
