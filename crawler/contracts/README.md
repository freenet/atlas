# Atlas crawler contracts

## room_contract.wasm

A copy of River's room-contract WASM, used to derive the CURRENT room-contract
key for River-room ingestion (Atlas issue #2) — see the doc comment on
`ROOM_CONTRACT_WASM` in `../src/main.rs` for why this must be a bundled WASM
rather than a hand-copied hash constant.

**Must be kept in sync with River's own copy.** When River re-keys the room
contract (any change under `river/main/contracts/room-contract/`), update
this file:

```bash
cp ../../../river/main/cli/contracts/room_contract.wasm contracts/room_contract.wasm
```

Then `cargo test -p atlas-crawler the_bundled_wasm_is_not_yet_legacy` — if the
file was already stale, that test was failing before the copy and passes
after. If it does NOT fail before the copy, the file did not need updating.
