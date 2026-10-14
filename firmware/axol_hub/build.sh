#!/usr/bin/env bash
# Build the Axol hub firmware and install the image the Python package ships
# (almond_axol/hub/axol_hub.uf2). Needs the Pico SDK 2.2.0 and an
# arm-none-eabi toolchain; see README.md.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
build="${BUILD_DIR:-$here/build}"

: "${PICO_SDK_PATH:=$HOME/.pico-sdk/sdk/2.2.0}"
export PICO_SDK_PATH

cmake -S "$here" -B "$build" ${CMAKE_ARGS:-}
cmake --build "$build" -j "$(nproc)"
cp "$build/axol_hub.uf2" "$repo/almond_axol/hub/axol_hub.uf2"
echo "installed $repo/almond_axol/hub/axol_hub.uf2"
