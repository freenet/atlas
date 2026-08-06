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

- The index contract's frequent, silent re-keying (any `common/`/`contracts/` change) is exactly the failure class [freenet-core#5194](https://github.com/freenet/freenet-core/issues/5194) is meant to fix — there is no stable-identity pointer published yet.
- Depend on `atlas-common` for the wire types rather than re-deriving the index's state/summary/delta shapes by hand.
