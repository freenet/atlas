#!/usr/bin/env bash
# Reproducibly build the Atlas index-contract WASM and refresh the committed
# artifact that `cli/build.rs` embeds.
#
# The contract's on-network address is `hash(compiled_wasm, params)`, so the
# committed WASM must be byte-identical no matter who rebuilds it — otherwise a
# rebuild silently re-keys the live index and orphans its curated entries.
#
# Two things make the build deterministic:
#   1. rust-toolchain.toml pins the exact compiler (toolchain drift is the
#      biggest reproducibility threat).
#   2. --remap-path-prefix strips machine-specific absolute paths (cargo
#      registry, rustup sysroot, the checkout dir) that otherwise leak into
#      panic-location strings baked into the binary. `trim-paths` would do this
#      but is still unstable on the pinned stable toolchain, so we use the
#      stable RUSTFLAGS form.
#
# Run from anywhere; paths are resolved relative to this script.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$here/../.." && pwd)"
cargo_home="${CARGO_HOME:-$HOME/.cargo}"
rustup_home="${RUSTUP_HOME:-$HOME/.rustup}"

# Order matters: rustc applies the first matching prefix, so remap the most
# specific (the checkout) before the broader home subdirs.
export RUSTFLAGS="\
--remap-path-prefix=$workspace=/atlas \
--remap-path-prefix=$cargo_home=/cargo \
--remap-path-prefix=$rustup_home=/rustup \
${RUSTFLAGS:-}"

cd "$workspace"
# --locked: this artifact's hash IS the contract's address, and freenet-stdlib is a
# caret requirement, so letting cargo re-resolve would silently change the address.
# Fail loudly on a stale lockfile instead.
cargo build --locked --release -p atlas-index-contract --target wasm32-unknown-unknown

built="$workspace/target/wasm32-unknown-unknown/release/atlas_index_contract.wasm"
dest="$here/atlas_index_contract.wasm"
cp "$built" "$dest"

echo "refreshed $dest"
if command -v b3sum >/dev/null 2>&1; then
  echo -n "blake3 code hash: "; b3sum "$dest" | cut -d' ' -f1
fi
