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

Then two steps, both required:

1. `cargo test -p atlas-crawler the_bundled_wasm_is_not_yet_legacy` — if the
   file was already stale, that test was failing before the copy and passes
   after. If it does NOT fail before the copy, the file did not need updating.
2. Refresh `OFFICIAL_CURRENT_KEY` in `src/main.rs` from
   `riverctl debug contract-key 4uNUKFzZQCnzo4K2ecZ16cMsYEEfoaRS35z6exEsbvm4`
   (or the room's owner VK, if it ever changes), then run
   `room_key_derivation_matches_the_live_network`. Sourcing that value from
   `riverctl` rather than from this crate's own computation is the whole
   point — it is the one check that can catch a wrong-file `cp` or a broken
   hash computation, neither of which step 1 alone can see.
