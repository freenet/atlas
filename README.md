# Atlas

Atlas is a **decentralized discovery layer for Freenet**: a way to describe,
index, search, recommend, review, and discover Freenet content and
applications without relying on a centralized search engine or recommendation
service.

Rather than imposing a single canonical index or ranking, Atlas provides a
framework for publishing signed metadata and building many competing,
pluralistic discovery systems on top of it. Users stay in control of which
analyzers, indexes, curators, and ranking policies they trust.

**A first version is live on Freenet.** With a Freenet node running locally,
open Atlas at:

<http://127.0.0.1:7509/v1/contract/web/771DvtPMwt2PumPyrFvsz7fpvU1gogcmb5qtS1yYEEH9/>

## UI

The Atlas browsing experience: open it, search, and open what you find, with
no setup or learning curve. The screenshot below is the live UI. Its entries
are gathered automatically by a crawler that describes each resource with an
LLM.

![The Atlas discovery UI](atlas-screenshot.png)

## Publishing (curator)

The index contract's address is `hash(compiled_wasm, params)`, so **any change
under `common/` or `contracts/` re-keys the live index**. The UI has to be told
the new address, and the curated entries have to be carried across. Getting the
order wrong has already cost us once: the stdlib-0.8.3 bump re-keyed the index
and the UI was never rebuilt, so from 2026-07-21 the published site quietly
served a frozen pre-re-key snapshot, six entries behind. Nothing errors in that
state; new listings just stop appearing.

Run these **in order**. Steps 1-3 leave the UI reading the still-populated old
address, so there is no window where the site shows an empty index.

```bash
# 0. Sanity: what moved, and what is the new address?
atlasctl keys                    # current + every registered legacy generation

# 1. Carry the entries forward FIRST. This also performs the initial PUT that
#    creates the new address, so `atlasctl add` would fail before it.
atlasctl migrate --dry-run       # survey; reports per-generation contents
atlasctl migrate

# 2. Confirm what the new address now serves.
atlasctl show

# 3. Register any app-hosted containers (see "app registry" below).
atlasctl apps

# 4. Rebuild the UI against the NEW index id. There is deliberately no default:
#    the build fails if this is unset, because a stale default is what froze the
#    site last time.
ATLAS_INDEX_ID=$(atlasctl key) dx build --release -p atlas-ui

# 5. Publish the web container (its address does NOT move; only its content).
atlasctl webapp-sign --archive webapp.tar.xz --version <n> --out-meta meta.cbor
atlasctl webapp-put --wasm <container.wasm> --archive webapp.tar.xz --metadata meta.cbor
```

When the rebuild changes the contract WASM, the **re-key ritual** is mandatory
and CI enforces it: preserve the outgoing generation's WASM under
`contracts/index-contract/legacy/` and prepend its code hash to
`LEGACY_INDEX_CODE_HASHES` in `cli/src/migration.rs`, in the same change. Skipping
it orphans the curated entries at an address nothing probes.

### App registry

Some Freenet content is not contract-addressed: every Delta site is served by the
*same* Delta web container, so it is identified by `(app, resource)` rather than
by a contract id. Those entries use an `app:<slug>/<resource>` locator and are
resolved through a root-signed registry in the index state:

```bash
atlasctl app-set --app delta --name Delta \
  --contract-id EqJ5YpEEV3XLqEvKWLQHFhGAac2qXzSUoE6k2zbdnXBr \
  --expect-version <current>     # ALWAYS required (use 0 for the first registry)
```

When an app republishes and its container address moves, re-point it with one
`app-set` and every entry for that app follows. Entries naming an app the
registry does not know are valid but render as "Unavailable" until it is
registered, so ordering between adding entries and registering an app does not
matter.

## Status

Atlas is early and evolving. The first version, a fast, populated, read-only
front door to Freenet, is live. The broader design (open descriptors and
reviews, publisher self-submission, and multiple competing indexes) is still
taking shape, so goals, architecture, schemas, and naming may still change.

See [PROPOSAL.md](PROPOSAL.md) for the full RFC.
