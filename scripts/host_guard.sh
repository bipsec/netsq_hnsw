#!/usr/bin/env bash
# Refuse to start a timed campaign unless the host is quiet (vLLM stopped, RAM free, P-cores at
# max frequency, low load). Usage: scripts/host_guard.sh && <timed command>
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m nestbench.host_guard --require "$@"
