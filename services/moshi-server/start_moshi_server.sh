#!/bin/bash
set -ex

export LD_LIBRARY_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

cargo run --features=cuda --bin=moshi-server -r -- $@
