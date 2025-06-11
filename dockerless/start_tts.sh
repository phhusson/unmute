#!/bin/bash
set -ex
cd "$(dirname "$0")/.."

cd ../moshi-rs
cd moshi-server
uv sync
export PATH=$(pwd)/.venv/bin:$PATH
export LD_LIBRARY_PATH=$(python -c "from distutils.sysconfig import get_config_var as s; print(s('LIBDIR'))")
cd ..
cargo run --features=cuda --bin=moshi-server -r -- worker --config=moshi-server/config-py-swarm-small.toml --port 8089
