#!/bin/bash
set -ex

export LD_LIBRARY_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

CARGO_TARGET_DIR=/app/target cargo install --features cuda moshi-server@0.6.0

# We use the absolute path because we might get moshi-server from python otherwise
/root/.cargo/bin/moshi-server $@
