# FREENET.md

This file enumerates the Freenet contracts published from this repository — what each one is for, where its source lives, and how to depend on it — for anyone integrating with Atlas rather than building it. It's a convention (see [freenet-core#5194](https://github.com/freenet/freenet-core/issues/5194)), not a protocol requirement: a fixed, predictable place to look before reading source.

## Contracts

### atlas-index-contract
- **Purpose:** The discovery index itself — signed metadata entries (descriptions, search terms, ratings) that the UI and crawler read and write. This is the thing a competing curator, analyzer, or alternative UI would actually want to read from or publish into.
- **Source:** [`contracts/index-contract/`](contracts/index-contract/)
- **Shared types crate:** [`atlas-common`](common/) ("Shared types for the Atlas discovery layer on Freenet") — not yet published to crates.io; depend via a git or path dependency for now.
- **Deployed key:** re-keys on any change under `common/` or `contracts/` (content-addressed: `hash(compiled_wasm, params)`), so **there is no single fixed address to cite here** — see the README's "Publishing" section for the current one and `cli/src/migration.rs`'s `LEGACY_INDEX_CODE_HASHES` for the full history of prior generations, which a client re-derives and probes to recover across an upgrade.

### web-container
- **Purpose:** Serves the compiled Atlas UI.
- **Source:** not vendored — reuses the generic, reusable `web-container-contract` WASM (the same one River publishes; see [`contracts/web-container/`](contracts/web-container/), which holds only the compiled artifact, not source).
- **Deployed key:** currently `771DvtPMwt2PumPyrFvsz7fpvU1gogcmb5qtS1yYEEH9` (see README for the live, current value — this moves on republish).

## Other components

- **[`cli/`](cli/) — `atlasctl`.** Command-line publishing/query client.
- **[`crawler/`](crawler/) — `atlas-crawler`.** Background service that discovers and describes resources (via an LLM) and publishes entries into the index. Someone building a competing curator would look here for the shape of a well-formed entry, not to reuse the crawler itself.

## Notes for integrators

- The index contract's frequent, silent re-keying (any `common/`/`contracts/` change) is exactly the failure class [freenet-core#5194](https://github.com/freenet/freenet-core/issues/5194) is meant to fix. **Atlas now publishes a pointer record — resolve it instead of pinning a key** (below).

## Stable identity: resolve the pointer, do not pin the key

Atlas's index contract re-keys on any `common/` or `contracts/` change. A build-time constant pointing at it goes stale silently, and the stale key still answers — it returns an older, frozen index rather than an error.

**That is not a hypothetical.** On 2026-08-18 this repository's own key documentation was found naming `4uasDWz1zGhT845Tkey9ai3Yo8GRXRvtVC7hX3o5tj6A` as the index id. That value derives from a *preserved legacy* WASM and was two re-keys behind. Both ids answered a GET: the stale one returned 17,209 bytes against the current 69,446.

### The author verifying key — your trust anchor

```
430b530d2089c82db23df0b6f01e3f7d04069714ad8ee48e3732280f6f31e9f8
```

Pin this 32-byte value as a constant. It is Atlas's root identity and the entire trust anchor: take it from anywhere else and you may resolve a validly-signed pointer belonging to somebody else. You can check it without trusting this file — it is the owner key from which Atlas's web-container id `771DvtPMwt2PumPyrFvsz7fpvU1gogcmb5qtS1yYEEH9` derives.

Atlas is a **single-writer** system, so this key is genuinely a root key rather than a release key: losing it makes the live Atlas permanently un-updatable.

### The pointer

| `app_id` | Points at | Pointer key (fixed, GET this) |
|---|---|---|
| `atlas.index-contract` | [`contracts/index-contract/`](contracts/index-contract/) | `BwsKx5iDhjBJGDNAPtZbbC9f6twDAUrnb2Yh1D6Wng2K` |

Derivable offline from the pointer contract's frozen code hash `8wnAPaSRY1oYZCz723fdwK6BgzL6q8ozP3buVovXnt6v` and `(author_vk ‖ app_id)`, so you need not trust the table.

The record names the index's current **code hash**; derive the contract key from that hash plus **your own params** (`root_vk ‖ slug`, `default` for the main index) — not the pointer's. That step is the one integrators get wrong.

Current record: [`pointer-records.toml`](pointer-records.toml), checked by CI on every PR (`scripts/check-pointer-freshness.sh`) — if the index WASM changes and no new record is signed, the build fails.

### How to resolve

```rust
use freenet_migrate::pointer::{resolve_app_pointer, PointerFloor, PointerOutcome};

let outcome = resolve_app_pointer(&mut io, &ATLAS_ROOT_VK, b"atlas.index-contract", floor).await?;
```

Handle **every** arm — a bare `if let Some(r) = outcome.resolved()` silently does nothing on the outcomes that carry no record, so a withdrawal, a rollback attempt and a plain timeout all become "no output". Only `NeverPublished` permits falling back to a baked-in key. Persist `outcome.next_floor()`, keyed by `(author_vk, app_id)`. Non-Rust implementers: wire format and hex test vectors are in the [pointer contract's README](https://github.com/freenet/freenet-migrate/tree/main/contracts/pointer-contract).

### What the pointer does NOT tell you

**It solves addressing only.** It tells you which code hash is current; it says nothing about whether the state under the previous key came with it. For Atlas the index is curator-published and re-published forward, so in practice it does — but that is a property of how Atlas operates, not something the pointer guarantees, and it is worth keeping the two apart.

- Depend on `atlas-common` for the wire types rather than re-deriving the index's state/summary/delta shapes by hand.
