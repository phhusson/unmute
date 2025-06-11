#!/bin/bash
set -ex
cd "$(dirname "$0")/.."

cd ../moshi-rs
cargo run --features=cuda --bin=moshi-server -r -- worker --config=moshi-server/config-basr-swarm-small.toml --port 8090
