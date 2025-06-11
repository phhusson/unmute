#!/bin/bash
set -ex
cd "$(dirname "$0")/.."

cd frontend
pnpm env use --global lts
pnpm dev
