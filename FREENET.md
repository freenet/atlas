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

## River rooms in the index

Public River rooms are indexed as ordinary entries, so reading them needs no new
contract or API: read the index state and filter. The Rust implementation is the
[`atlas-rooms`](rooms/) crate, whose `search::search` is the read path River's
Discover Rooms mirrors. **No contract or `common/` change was needed, so the
index address does not move.**

| `IndexEntry` field | Meaning for a room |
|---|---|
| `kind` | `"Room"` |
| `locator` | `AppResource { app: "river", resource: <room owner verifying key, base58, FULL key>, path: "" }`, URI `app:river/<owner key>` |
| `title` | The owner's room name, a sanitised snapshot taken from the owner-signed configuration. The contract requires a title; this is the one owner-sourced field. |
| `snippet` | Atlas's OWN characterisation (curator or LLM), or empty. Never the owner's description. |
| `ext.ss` | Source of `snippet`: `c` curator, `l` LLM. Absent when `snippet` is empty. |
| `ext.ts` | Source of `title`, always `o` (owner). |
| `ext.cv` | Owner-signed `configuration_version` the title was read at. A read at a lower version is ignored as stale. |
| `ext.rv` | Room-descriptor schema version, `1`. |
| `tags`, `class`, `featured` | Curator-asserted, as for any entry. Unclassified rooms are hidden by safe search. |
| `verified` | When Atlas last read the room's signed state: `live` (read and valid) or `unreachable` (a read proved nothing; not a takedown). |

**Identity.** The room's identity is the owner key. It survives room-WASM
re-keys because River derives the contract from the owner key plus the CURRENT
room-contract code hash, which it resolves from its own signed pointer record
(`river.room-contract`). Atlas stores no contract id and no room-WASM hash.
`atlasctl` resolves that pointer at run time and refuses to guess when it does
not resolve. Short handles are display-only.

**The owner's description is not in the index.** It is owner-signed in the
room's own state, so a client reads it from the room. The index carries only
what Atlas asserts, so the two are never merged and a stale copy of the owner's
words cannot outlive an edit.

**Who is indexed.** Only rooms whose owner-signed configuration verifies
(`verify_strict`) against the owner key in the locator, is `PrivacyMode::Public`
with a plaintext name, and that a curator has submitted (`atlasctl room-add`).
Owner self-announcement waits for river#974: the owner's opt-in is
`Configuration.public_admission.listed` (river branch `feat/public-admission`,
**pending until that merges**; no deployed configuration carries it yet). When it
exists, `listed == false` always delists and `listed == true` stands in for the
submission. A refresh delists a room that goes private, opts out, or whose
owner-signed configuration this reader cannot parse (fail closed), ignores a read
older than any listing for that owner (equal is not stale), and on an unreadable
room only marks the entry `unreachable`, and only once it has gone 24 hours
without a good read.

**Limits of the privacy gate, stated plainly.**
- **Freshness.** The gate proves the configuration was signed by the owner, not
  that it is current. `atlasctl` reads rooms with a subscribing GET and
  keeps a persisted anti-rollback floor for River's room-contract pointer
  (`river_room_pointer_floor.json` in the key directory), but a subscribing GET is not a
  network-fresh read: freenet-core serves a copy the node already hosts, is
  subscribed to, or holds while isolated, and only otherwise asks the network. A
  lagging or hostile node could still serve an old public configuration. The stale-read guard only
  protects rooms already listed, so a FIRST `room-add` has nothing to compare
  against. `room-add` prints the owner-signed configuration version and
  privacy mode it read, so compare them with the room before publishing.
- **Standing subscriptions.** Each room read leaves the local node subscribed to
  that room contract (freenet-stdlib 0.8 has no unsubscribe request), so a sweep
  subscribes the node to every listed room until the node drops them. A room the
  network cannot find is abandoned after 25 seconds.
- **A sweep delists at most a fifth of the rooms (minimum 3)** unless
  `--force-delist` is passed, because a River configuration change this reader
  cannot parse delists every room it touches (fail closed).
- **Opt-out depends on the field name.** Until river#974 merges, the reader looks
  for `public_admission` by name. If River ships a different name, an owner's
  `listed == false` would read as "no policy" and the room would stay eligible,
  because `room-add` and `room-refresh` always count as a submission. The golden
  fixtures from the #974 branch fail if the name changes.
- **Nothing schedules `room-refresh`** ([#88](https://github.com/freenet/atlas/issues/88)),
  so a room that goes private stays listed until someone runs it.

```bash
atlasctl room-add <owner-key> --landing general --tags rust,chat \
    --characterization "A busy channel for Rust learners."   # --dry-run to preview
atlasctl room-refresh [<owner-key>]    # renames, going private, re-verification
atlasctl rooms [query] [--tag t] [--json] [--only-live] [--all]
```

For links to resolve in the UI, the `river` app must be registered
(`atlasctl app-set --app river ...`; tracked in [#86](https://github.com/freenet/atlas/issues/86)).
Rooms stay hidden by safe search until classified ([#87](https://github.com/freenet/atlas/issues/87)).

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
