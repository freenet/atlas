//! atlasctl: the Atlas curator CLI. Manages the single-writer index contract on
//! a Freenet node. The root key authorizes an online signing key; entries are
//! signed by the online key and merged into the index by per-subject version.

mod api;
mod migration;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use atlas_common::{
    generate_key, sign, AppRecord, AppRegistry, AppRegistryBody, IndexDelta, IndexEntry,
    IndexParams, IndexState, KeyAuth, KeyAuthBody, Kind, Locator, RecordBody, SignedRecord,
    SubjectId, Tombstone,
};
use clap::{Parser, Subcommand};
use ed25519_dalek::{Signature, Signer, SigningKey};
use serde::Serialize;

use api::NodeClient;
use freenet_stdlib::prelude::ContractInstanceId;

const CONTRACT_WASM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/atlas_index_contract.wasm"));
const DEFAULT_URL: &str = "ws://127.0.0.1:7509/v1/contract/command?encodingProtocol=native";

#[derive(Parser)]
#[command(name = "atlasctl", version, about = "Atlas curator CLI")]
struct Cli {
    /// Node WebSocket URL.
    #[arg(long, default_value = DEFAULT_URL, global = true)]
    node: String,
    /// Key directory holding root.key and online.key (default: ~/.config/atlas).
    #[arg(long, global = true)]
    key_dir: Option<PathBuf>,
    /// Index slug (discriminates contract instances under the same root key).
    #[arg(long, default_value = "default", global = true)]
    slug: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate the root and online signing keys.
    Keygen,
    /// PUT a fresh, empty index (root authorizes the online key).
    Init,
    /// Add a new subject (mints a subject id at version 1).
    Add {
        #[arg(long)]
        kind: String,
        #[arg(long)]
        title: String,
        #[arg(long, default_value = "")]
        snippet: String,
        /// Comma-separated tags.
        #[arg(long, default_value = "")]
        tags: String,
        /// `freenet:<full-id><path>`, `app:<slug>/<resource>[<path>]`, or
        /// `https://...`
        #[arg(long)]
        locator: String,
        #[arg(long)]
        featured: bool,
        /// Add even if an entry with the same dedup key is already listed.
        #[arg(long)]
        allow_duplicate: bool,
    },
    /// Tombstone a subject by id (needs the current version to supersede it).
    Remove {
        #[arg(long)]
        subject: String,
        #[arg(long)]
        cur_version: u64,
    },
    /// Register or re-point an app in the root-signed app registry, so
    /// `app:<slug>/<resource>` locators resolve to a live address.
    ///
    /// Run this after an app republishes: it is the ONE edit that re-points
    /// every `AppResource` entry for that app, instead of rewriting each entry.
    /// Reads the current registry, applies the change, and PUTs it back at
    /// version+1 (root-signed).
    AppSet {
        /// App slug, `[a-z0-9-]` (e.g. `delta`).
        ///
        /// Named `--app`, NOT `--slug`: `--slug` is the global INDEX slug, and a
        /// subcommand flag of the same name shadowed it, so `app-set --slug delta`
        /// silently pointed the whole command at a different index and failed with
        /// a confusing "contract not found".
        #[arg(long)]
        app: String,
        /// The app's CURRENT web-container contract instance id.
        #[arg(long)]
        contract_id: String,
        /// Display name (e.g. `Delta`).
        #[arg(long)]
        name: String,
        /// Path template after the contract id; `{resource}` and `{path}` are
        /// substituted.
        #[arg(long, default_value = "/#{resource}{path}")]
        link_template: String,
        /// The registry version you believe you are superseding. ALWAYS required,
        /// including `--expect-version 0` for the first-ever registry: a node that
        /// has merely not seen the registry also reports 0, so accepting an omitted
        /// value there would let a stale read silently drop every registered app.
        #[arg(long)]
        expect_version: Option<u64>,
    },
    /// Remove an app from the registry. Entries naming it stay in the index but
    /// become unresolvable, so prefer re-pointing over removing.
    AppUnset {
        /// App slug (see the note on `app-set --app`).
        #[arg(long)]
        app: String,
        #[arg(long)]
        expect_version: Option<u64>,
    },
    /// Print the app registry. `--json` emits the machine-readable form the
    /// crawler consumes to recognise app-hosted links.
    Apps {
        #[arg(long)]
        json: bool,
    },
    /// GET and print the index's live entries.
    Show {
        /// Subscribe (and blocking-subscribe) so the connected node hosts the
        /// index, making it findable by cross-node GETs.
        #[arg(long)]
        subscribe: bool,
    },
    /// Print the index contract id (no network).
    Key,
    /// Print the current index id plus every legacy (pre-rebuild) id, so a
    /// curator can see exactly which addresses a migration spans (no network).
    Keys,
    /// Carry the curated index forward after a rebuild re-keyed it: GET EVERY
    /// legacy (pre-rebuild) generation and PUT-merge each non-empty state into
    /// the current address (subscribing so the node hosts it). The contract
    /// merges per-subject by version (tombstone-aware), so merging all
    /// generations is idempotent and order-independent — re-running, or a
    /// generation holding only tombstones, never resurrects a deleted subject. A
    /// legacy GET that errors (dead-end / timeout) aborts rather than being
    /// treated as an empty state.
    Migrate {
        /// Probe/GET only; report what WOULD be carried forward without PUTting.
        #[arg(long)]
        dry_run: bool,
        /// Accept a SPECIFIC generation reporting NOT FOUND, by instance id.
        /// Repeatable.
        ///
        /// Deliberately per-generation rather than a blanket flag: a registered
        /// generation that legitimately holds nothing reports NOT FOUND on every
        /// run forever, so a blanket flag would become habitual and would then also
        /// silence the generation that DOES hold entries — the exact failure this
        /// guard exists to prevent.
        #[arg(long, value_name = "INSTANCE_ID")]
        allow_missing: Vec<String>,
    },
    /// Write the web-container params (root vk) and print its contract id
    /// (needed for the UI base_path). Reuses the root key as the UI owner.
    WebappParams {
        /// Path to the (generic) web-container contract wasm.
        #[arg(long)]
        wasm: PathBuf,
        #[arg(long)]
        out_params: PathBuf,
    },
    /// Sign a compressed webapp archive into web-container metadata (CBOR).
    WebappSign {
        #[arg(long)]
        archive: PathBuf,
        #[arg(long)]
        version: u32,
        #[arg(long)]
        out_meta: PathBuf,
    },
    /// Publish (PUT) the web-container that serves the UI to the node.
    WebappPut {
        #[arg(long)]
        wasm: PathBuf,
        #[arg(long)]
        archive: PathBuf,
        #[arg(long)]
        metadata: PathBuf,
    },
    /// Raw GET of any contract instance id; writes its state bytes to --out.
    RawGet {
        instance: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Repair a stale replica: GET the index state from a source node and PUT it
    /// into a target node so that node serves the current state locally. Useful
    /// while cross-node subscribe/propagation is unreliable (the target may hold
    /// an old copy it can't refresh because its subscribe dead-ends).
    PushState {
        /// Source node holding the authoritative current state.
        #[arg(long, default_value = DEFAULT_URL)]
        from: String,
        /// Target node to push the state into (e.g. a tunnel to a stale node).
        #[arg(long)]
        to: String,
    },
}

/// Mirrors River's web-container metadata so the generic web-container contract
/// accepts our signed webapp. Signature covers `version.to_be_bytes() || webapp`.
#[derive(Serialize)]
struct WebContainerMetadata {
    version: u32,
    signature: Signature,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let dir = resolve_key_dir(&cli.key_dir);
    match &cli.cmd {
        Cmd::Keygen => keygen(&dir),
        Cmd::Init => init(&cli, &dir).await,
        Cmd::Add {
            kind,
            title,
            snippet,
            tags,
            locator,
            featured,
            allow_duplicate,
        } => {
            add(
                &cli,
                &dir,
                kind,
                title,
                snippet,
                tags,
                locator,
                *featured,
                *allow_duplicate,
            )
            .await
        }
        Cmd::Remove {
            subject,
            cur_version,
        } => remove(&cli, &dir, subject, *cur_version).await,
        Cmd::AppSet {
            app,
            contract_id,
            name,
            link_template,
            expect_version,
        } => {
            app_set(
                &cli,
                &dir,
                app,
                Some(AppRecord {
                    contract_id: contract_id.clone(),
                    name: name.clone(),
                    link_template: link_template.clone(),
                }),
                *expect_version,
            )
            .await
        }
        Cmd::AppUnset {
            app,
            expect_version,
        } => app_set(&cli, &dir, app, None, *expect_version).await,
        Cmd::Apps { json } => apps(&cli, &dir, *json).await,
        Cmd::Show { subscribe } => show(&cli, &dir, *subscribe).await,
        Cmd::Key => {
            let params = params_bytes(&dir, &cli.slug)?;
            let key = NodeClient::contract_key(CONTRACT_WASM, &params);
            println!("{}", key.id());
            Ok(())
        }
        Cmd::Keys => {
            let params = params_bytes(&dir, &cli.slug)?;
            let key = NodeClient::contract_key(CONTRACT_WASM, &params);
            println!("current: {}", key.id());
            for (i, k) in migration::legacy_index_keys(&params).iter().enumerate() {
                println!("legacy[{i}]: {}", k.id());
            }
            Ok(())
        }
        Cmd::Migrate {
            dry_run,
            allow_missing,
        } => migrate(&cli, &dir, *dry_run, allow_missing).await,
        Cmd::WebappParams { wasm, out_params } => webapp_params(&dir, wasm, out_params),
        Cmd::WebappSign {
            archive,
            version,
            out_meta,
        } => webapp_sign(&dir, archive, *version, out_meta),
        Cmd::WebappPut {
            wasm,
            archive,
            metadata,
        } => webapp_put(&cli, &dir, wasm, archive, metadata).await,
        Cmd::RawGet { instance, out } => raw_get(&cli, instance, out).await,
        Cmd::PushState { from, to } => push_state(&dir, &cli.slug, from, to).await,
    }
}

/// GET the index state from `from` and PUT it into `to`. The target merges it
/// (per-subject LWW) into whatever stale copy it holds, so it ends up serving
/// the current state locally. Records are already signed by the online key, so
/// no re-signing is needed; the target contract verifies them on update.
async fn push_state(dir: &Path, slug: &str, from: &str, to: &str) -> Result<()> {
    let params = params_bytes(dir, slug)?;
    let key = NodeClient::contract_key(CONTRACT_WASM, &params);
    println!("index {}", key.id());

    let mut src = NodeClient::connect(from).await?;
    let state = src.get(&key, false).await?;
    let live_src = count_live(&state);
    println!("source has {live_src} live entries ({} bytes)", state.len());

    // PUT-over-existing on the target is applied as a merging update; subscribe
    // is left off so we don't block on the target's dead-ending subscribe.
    let mut dst = NodeClient::connect(to).await?;
    dst.put(CONTRACT_WASM, params, state, false).await?;
    println!("pushed state to {to}");

    // Read the target back to confirm it now serves the current state.
    let after = dst.get(&key, false).await?;
    println!("target now has {} live entries", count_live(&after));
    Ok(())
}

/// Carry the curated index forward after a rebuild re-keyed the contract.
///
/// The index address is `hash(wasm, params)`; a stdlib/toolchain bump that
/// changes the WASM moves it. This GETs EVERY legacy (pre-rebuild) address and
/// PUT-merges each non-empty state into the current address, so no generation's
/// entries are stranded. The contract merges per-subject by version
/// (tombstone-aware), so merging all generations is idempotent and order-
/// independent — re-running is safe, and a tombstone-only generation still
/// contributes its takedowns without resurrecting anything.
///
/// A legacy GET that *errors* (a dead-ended / timed-out cross-node probe) is NOT
/// treated as an empty state: it aborts, so the operator never rebuilds the UI
/// onto an empty index while entries still live at the old address.
async fn migrate(cli: &Cli, dir: &Path, dry_run: bool, allow_missing: &[String]) -> Result<()> {
    let params = params_bytes(dir, &cli.slug)?;
    let current_key = NodeClient::contract_key(CONTRACT_WASM, &params);
    let legacy_keys = migration::legacy_index_keys(&params);
    if legacy_keys.is_empty() {
        bail!("no legacy code hashes registered — nothing to migrate from");
    }
    println!("current index: {}", current_key.id());

    let mut client = NodeClient::connect(&cli.node).await?;

    // Carry EVERY legacy generation forward, not just the "biggest" one. The
    // contract merges per-subject by version (tombstone-aware), so PUT-merging
    // all of them is idempotent and order-independent: a generation holding only
    // tombstones still contributes its takedowns, and a stale generation cannot
    // resurrect a subject a newer one deleted. No count-based selection or
    // early-return — every non-empty generation participates.
    let mut merged = 0usize;
    let mut probe_errors = 0usize;
    let mut missing: Vec<String> = Vec::new();
    for (i, key) in legacy_keys.iter().enumerate() {
        // M1: a GET *error* (dead-ended / timed-out cross-node probe) must NEVER
        // be silently treated as an empty state — otherwise a rebuild would land
        // on an empty index while the entries still live at the old address. On a
        // real run we abort immediately (the operator fixes connectivity and
        // re-runs; the merge is idempotent). In a dry-run we instead warn and keep
        // surveying the remaining generations, then exit non-zero at the end, so
        // the operator sees the full picture rather than only the first failure.
        // A definitive NotFound means the address genuinely holds nothing, which
        // can be a legitimate outcome (a registered generation that holds nothing),
        // but is NOT proof of it.
        // Treating it as an abort (which folding it into the error bucket did)
        // made the command unusable: probing an empty generation stopped the run
        // before it reached the generation holding the entries.
        //
        // It is reported LOUDLY rather than silently, because NotFound is not
        // absolute proof of absence — a contract that exists but is momentarily
        // unfindable answers the same way. The merge is idempotent, so re-running
        // later recovers such a generation; the operator just has to know to.
        let probe = client.get_optional(key, false).await;
        let id_str = key.id().to_string();
        if let Ok(None) = &probe {
            println!(
                "legacy[{i}] {} NOT FOUND — treated as absent and skipped.\n\
                 WARNING: if that generation was actually published, it is \
                 momentarily unfindable rather than empty; re-run migrate later \
                 (merging is idempotent) and confirm with `atlasctl show`.",
                key.id()
            );
            missing.push(id_str.clone());
            continue;
        }
        let plan = match plan_generation(probe.map(|o| o.unwrap_or_default())) {
            Ok(plan) => plan,
            Err(e) if dry_run => {
                eprintln!(
                    "legacy[{i}] {} GET FAILED (reported, NOT counted as empty): {e:#}",
                    key.id()
                );
                probe_errors += 1;
                continue;
            }
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "legacy[{i}] {} GET failed — aborting so its entries are \
                         not silently dropped as empty",
                        key.id()
                    )
                });
            }
        };
        match plan {
            None => println!(
                "legacy[{i}] {} is empty — nothing to carry forward",
                key.id()
            ),
            Some((state, live, tomb)) => {
                println!(
                    "legacy[{i}] {} holds {live} live / {tomb} tombstone record(s)",
                    key.id()
                );
                // Pre-flight BEFORE the PUT (and in a dry-run, where it is the
                // whole point): the node will reject the entire generation if any
                // one record fails, so find that out here with a per-record
                // report rather than from an opaque contract error mid-publish.
                let index_params = IndexParams::from_bytes(&params)
                    .ok_or_else(|| anyhow!("could not parse index params"))?;
                if let Err(e) = preflight_generation(&state, &index_params) {
                    if dry_run {
                        eprintln!("legacy[{i}] {} PRE-FLIGHT FAILED: {e:#}", key.id());
                        probe_errors += 1;
                        continue;
                    }
                    return Err(e).with_context(|| {
                        format!("legacy[{i}] {} would be rejected by the contract", key.id())
                    });
                }
                if dry_run {
                    println!(
                        "  [dry-run] would PUT-merge legacy[{i}] into {}",
                        current_key.id()
                    );
                } else {
                    // PUT-over the current contract; the node applies it as a
                    // merging update and, with subscribe, hosts it so cross-node
                    // GETs can find it.
                    client
                        .put(CONTRACT_WASM, params.clone(), state, true)
                        .await
                        .with_context(|| {
                            format!("PUT-merging legacy[{i}] into the current index")
                        })?;
                    merged += 1;
                    println!("  merged legacy[{i}] into {}", current_key.id());
                }
            }
        }
    }

    // ANY skipped generation fails the run unless the operator explicitly accepts
    // it. An earlier version only failed when NOTHING merged, on the reasoning
    // that reaching one address made a NotFound elsewhere credible. That
    // reasoning is wrong: each generation is a different contract key, so a
    // different ring location with independent placement and findability, and
    // reaching one says nothing about another.
    //
    // The concrete failure it allowed: the newest entries live at legacy[0], not
    // legacy[1]. If legacy[0] answered NotFound and legacy[1] merged, migrate
    // exited 0 having silently dropped exactly the entries it was run to recover.
    // Any skipped generation the operator has NOT explicitly acknowledged fails the
    // run. Per-generation, not blanket: see the note on `--allow-missing`.
    //
    // Checked AFTER the readback below on a real run, so the operator still gets the
    // "current index now holds N live entries" line — the PUTs already happened, so
    // bailing before it protects nothing and hides the most useful datum.
    let unacknowledged: Vec<&String> = missing
        .iter()
        .filter(|id| !allow_missing.iter().any(|a| a == *id))
        .collect();
    if !missing.is_empty() {
        println!(
            "\nNOTE: {} generation(s) reported NOT FOUND and were skipped:\n{}",
            missing.len(),
            missing
                .iter()
                .map(|id| format!("  {id}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        println!(
            "NotFound is NOT proof of absence: a node returns it when a GET exhausts \
             its retries, so an existing-but-unfindable contract looks identical. \
             Merging is idempotent, so re-running later is safe."
        );
    }

    if dry_run {
        if probe_errors > 0 {
            bail!(
                "[dry-run] {probe_errors} legacy generation(s) were unreachable — a \
                 real migrate would abort on these; fix connectivity and retry so \
                 their entries are not left stranded"
            );
        }
        if !unacknowledged.is_empty() {
            bail!(
                "[dry-run] {} skipped generation(s) unacknowledged — a real migrate \
                 would refuse. Confirm them, or pass --allow-missing <id> for each.",
                unacknowledged.len()
            );
        }
        println!("[dry-run] complete — no state written");
        return Ok(());
    }

    // Read the current index back to confirm what it now serves. This is
    // informational only (the PUTs above already succeeded), so a failed readback
    // is a warning, not an abort.
    match client.get(&current_key, false).await {
        Ok(after) => println!(
            "current index {} now holds {} live entries",
            current_key.id(),
            count_live(&after)
        ),
        Err(e) => eprintln!("warning: could not read the current index back: {e:#}"),
    }
    if merged == 0 {
        println!("no legacy generation held any state — nothing was carried forward");
    }
    if !unacknowledged.is_empty() {
        bail!(
            "{} skipped generation(s) were not acknowledged. Each generation is a \
             separate contract at a separate ring location, so reaching one says \
             nothing about another — and the generation holding the NEWEST entries \
             is not necessarily the one that answered. Confirm each with \
             `atlasctl raw-get <id> --out /tmp/x`, then re-run, or accept \
             explicitly with:\n{}\nDo NOT publish a UI against this index until you \
             have.",
            unacknowledged.len(),
            unacknowledged
                .iter()
                .map(|id| format!("  --allow-missing {id}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    Ok(())
}

/// Decide what to do with one legacy generation, given the *outcome* of its GET.
///
/// This is the M1 seam: a GET **error** stays an `Err` here — it is NEVER folded
/// into "empty" — so the caller can abort (real run) or warn-and-continue
/// (dry-run) instead of silently skipping. Only a *successful* GET is classified:
/// `Ok(None)` means the address is genuinely empty (skip); `Ok(Some((state, live,
/// tomb)))` means carry the state forward (any non-empty state, including a
/// tombstone-only one, so takedowns still propagate).
fn plan_generation(get_result: Result<Vec<u8>>) -> Result<Option<(Vec<u8>, usize, usize)>> {
    let state = get_result?;
    if state.is_empty() {
        return Ok(None);
    }
    let (live, tomb) = count_records(&state)?;
    Ok(Some((state, live, tomb)))
}

/// Decode an index state and count its (live, tombstone) records.
///
/// An undecodable state is an ERROR, not `(0, 0)`. It used to be the latter, which
/// printed "holds 0 live / 0 tombstone record(s)" — indistinguishable from a
/// genuinely empty generation — and then PUT the bytes anyway.
fn count_records(state: &[u8]) -> Result<(usize, usize)> {
    if state.is_empty() {
        return Ok((0, 0));
    }
    let st: IndexState = ciborium::de::from_reader(state)
        .context("legacy state did not decode with the current types")?;
    let mut live = 0;
    let mut tomb = 0;
    for rec in st.records.values() {
        match rec.body {
            RecordBody::Live(_) => live += 1,
            RecordBody::Tomb(_) => tomb += 1,
        }
    }
    Ok((live, tomb))
}

/// Run the validation the CONTRACT will run, and report every record that fails.
///
/// This is the pre-flight, and it has to live here rather than in a scratch
/// script. `validate_state` is all-or-nothing: the node loops every record and one
/// failure rejects the ENTIRE PUT, so a single record that a tightened rule has
/// retroactively invalidated makes a whole generation unmigratable — with no
/// indication of which record is at fault. Counting records (all the dry-run used
/// to do) cannot see that at all.
///
/// Uses `IndexState::verify`, not just per-record `check_structure`: `verify`
/// additionally requires every signer to be authorized by the CURRENT key_auth and
/// every record to be stored under its own subject id, and both are things a
/// migration can get wrong.
fn preflight_generation(state: &[u8], params: &IndexParams) -> Result<()> {
    let st: IndexState = ciborium::de::from_reader(state)
        .context("legacy state did not decode with the current types")?;
    if st.verify(params).is_ok() {
        return Ok(());
    }
    // Whole-state verify failed. Localise it: report every individual record that
    // the current rules reject, so the operator learns WHICH entries block the
    // migration instead of just that something does.
    let mut failures = Vec::new();
    for (sid, rec) in &st.records {
        if let Err(e) = rec.body.check_structure() {
            failures.push(format!("  {} — {e}", sid.as_str()));
        } else if let Err(e) = rec.verify_sig() {
            failures.push(format!("  {} — bad signature: {e}", sid.as_str()));
        } else if rec.body.subject_id() != sid {
            failures.push(format!(
                "  {} — stored under the wrong subject id",
                sid.as_str()
            ));
        }
    }
    let whole = st.verify(params).unwrap_err();
    if failures.is_empty() {
        bail!(
            "this generation would be REJECTED by the contract, but no individual \
             record is at fault — the problem is state-level: {whole}"
        );
    }
    bail!(
        "this generation would be REJECTED by the contract ({whole}). \
         {} record(s) fail the current rules:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Count live (non-tombstoned) entries only. Used by `push_state` and the
/// migrate readback.
fn count_live(state: &[u8]) -> usize {
    count_records(state).map(|(live, _)| live).unwrap_or(0)
}

async fn raw_get(cli: &Cli, instance: &str, out: &Path) -> Result<()> {
    let id: ContractInstanceId = instance
        .parse()
        .map_err(|e| anyhow!("bad instance id: {e}"))?;
    let mut client = NodeClient::connect(&cli.node).await?;
    let bytes = client.get_instance(id).await?;
    fs::write(out, &bytes).with_context(|| format!("writing {}", out.display()))?;
    println!("wrote {} bytes to {}", bytes.len(), out.display());
    Ok(())
}

async fn webapp_put(
    cli: &Cli,
    dir: &Path,
    wasm: &Path,
    archive: &Path,
    metadata: &Path,
) -> Result<()> {
    let root = load_key(&dir.join("root.key"))?;
    let params = root.verifying_key().as_bytes().to_vec();
    let code = fs::read(wasm).with_context(|| format!("reading {}", wasm.display()))?;
    let meta = fs::read(metadata).with_context(|| format!("reading {}", metadata.display()))?;
    let web = fs::read(archive).with_context(|| format!("reading {}", archive.display()))?;
    // Web-container state layout (River's generic contract):
    // [metadata_len: u64 BE][metadata][web_len: u64 BE][web]
    let mut state = Vec::with_capacity(16 + meta.len() + web.len());
    state.extend_from_slice(&(meta.len() as u64).to_be_bytes());
    state.extend_from_slice(&meta);
    state.extend_from_slice(&(web.len() as u64).to_be_bytes());
    state.extend_from_slice(&web);
    let mut client = NodeClient::connect(&cli.node).await?;
    let key = client.put(&code, params, state, true).await?;
    println!("web-container published: {}", key.id());
    Ok(())
}

fn webapp_params(dir: &Path, wasm: &Path, out_params: &Path) -> Result<()> {
    let root = load_key(&dir.join("root.key"))?;
    let vk = root.verifying_key();
    fs::write(out_params, vk.as_bytes())
        .with_context(|| format!("writing {}", out_params.display()))?;
    let code = fs::read(wasm).with_context(|| format!("reading {}", wasm.display()))?;
    let key = NodeClient::contract_key(&code, vk.as_bytes());
    println!("{}", key.id());
    Ok(())
}

fn webapp_sign(dir: &Path, archive: &Path, version: u32, out_meta: &Path) -> Result<()> {
    let root = load_key(&dir.join("root.key"))?;
    let webapp = fs::read(archive).with_context(|| format!("reading {}", archive.display()))?;
    let mut message = Vec::with_capacity(4 + webapp.len());
    message.extend_from_slice(&version.to_be_bytes());
    message.extend_from_slice(&webapp);
    let signature = root.sign(&message);
    let meta = WebContainerMetadata { version, signature };
    let bytes = encode(&meta)?;
    fs::write(out_meta, bytes).with_context(|| format!("writing {}", out_meta.display()))?;
    println!("signed webapp v{version} -> {}", out_meta.display());
    Ok(())
}

fn keygen(dir: &Path) -> Result<()> {
    let root = generate_key();
    let online = generate_key();
    save_key(&dir.join("root.key"), &root)?;
    save_key(&dir.join("online.key"), &online)?;
    println!("keys written to {}", dir.display());
    println!("root_vk:   {}", b58(root.verifying_key().as_bytes()));
    println!("online_vk: {}", b58(online.verifying_key().as_bytes()));
    println!(
        "\nKeep root.key offline once the index is initialized; the online key is the hot signer."
    );
    Ok(())
}

async fn init(cli: &Cli, dir: &Path) -> Result<()> {
    let root = load_key(&dir.join("root.key"))?;
    let online = load_key(&dir.join("online.key"))?;
    let body = KeyAuthBody {
        version: 1,
        authorized: vec![online.verifying_key()],
    };
    let key_auth = KeyAuth {
        sig: sign(&body, &root),
        body,
    };
    let state = encode(&IndexState::initialized(key_auth))?;
    let params = IndexParams {
        root_vk: root.verifying_key(),
        slug: cli.slug.clone(),
    }
    .to_bytes();
    let mut client = NodeClient::connect(&cli.node).await?;
    let key = client.put(CONTRACT_WASM, params, state, true).await?;
    println!("index initialized");
    println!("contract id: {}", key.id());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn add(
    cli: &Cli,
    dir: &Path,
    kind: &str,
    title: &str,
    snippet: &str,
    tags: &str,
    locator: &str,
    featured: bool,
    allow_duplicate: bool,
) -> Result<()> {
    let online = load_key(&dir.join("online.key"))?;
    let locator = parse_locator(locator)?;
    // Refuse a second listing for the same subject. Subject ids are random, so
    // nothing else stops `add` minting a fresh one for a locator already in the
    // index — and for an app-hosted locator `dedup_key` deliberately ignores the
    // path, so two links to different pages of ONE Delta site collapse here
    // rather than becoming two cards.
    if !allow_duplicate {
        let read = fetch_state(cli, dir, false).await?;
        if matches!(read, IndexRead::Unfindable) {
            eprintln!(
                "warning: the node could not find the index, so the duplicate check \
                 was SKIPPED — this may create a second listing for something already \
                 indexed"
            );
        }
        if let IndexRead::Present(state) = read {
            let key = locator.dedup_key();
            if let Some(existing) = state.live_entries().find(|e| e.locator.dedup_key() == key) {
                bail!(
                    "already listed as subject {} ({}), locator {}\n\
                     pass --allow-duplicate to add it anyway, or `atlasctl remove` the existing one",
                    existing.subject_id.as_str(),
                    existing.title,
                    existing.locator.to_uri()
                );
            }
        }
    }
    let entry = IndexEntry {
        subject_id: SubjectId::random(),
        version: 1,
        kind: parse_kind(kind)?,
        title: title.to_string(),
        snippet: snippet.to_string(),
        tags: split_tags(tags),
        locator,
        featured,
        added_at: now_secs(),
    };
    let subject = entry.subject_id.as_str().to_string();
    let body = RecordBody::Live(entry);
    // Validate locally BEFORE signing and sending. The contract enforces the same
    // rules, but a rejection there arrives as an opaque `InvalidUpdateWithInfo`
    // from the node; checking here names the offending field.
    body.check_structure()
        .map_err(|e| anyhow!("entry would be rejected by the contract: {e}"))?;
    let rec = SignedRecord {
        sig: sign(&body, &online),
        by: online.verifying_key(),
        body,
    };
    send_delta(cli, dir, vec![rec]).await?;
    println!("added subject {subject}");
    Ok(())
}

async fn remove(cli: &Cli, dir: &Path, subject: &str, cur_version: u64) -> Result<()> {
    let online = load_key(&dir.join("online.key"))?;
    let subject_id = SubjectId::parse(subject).ok_or_else(|| anyhow!("malformed subject id"))?;
    let body = RecordBody::Tomb(Tombstone {
        subject_id,
        version: cur_version + 1,
    });
    let rec = SignedRecord {
        sig: sign(&body, &online),
        by: online.verifying_key(),
        body,
    };
    send_delta(cli, dir, vec![rec]).await?;
    println!("removed subject {subject}");
    Ok(())
}

async fn show(cli: &Cli, dir: &Path, subscribe: bool) -> Result<()> {
    let params = params_bytes(dir, &cli.slug)?;
    let key = NodeClient::contract_key(CONTRACT_WASM, &params);
    let mut client = NodeClient::connect(&cli.node).await?;
    let bytes = client.get(&key, subscribe).await?;
    if bytes.is_empty() {
        println!("(index is empty / not initialized)");
        return Ok(());
    }
    let state: IndexState =
        ciborium::de::from_reader(&bytes[..]).context("decoding index state")?;
    let mut entries: Vec<_> = state.live_entries().collect();
    entries.sort_by_key(|e| std::cmp::Reverse(e.added_at));
    println!("{} live entries:", entries.len());
    let mut unresolvable = 0;
    for e in entries {
        let star = if e.featured { "★ " } else { "  " };
        // Flag an app-hosted entry the registry cannot resolve: the listing is
        // structurally valid but would not open, and that is a curator action
        // (register the app) rather than a bad entry.
        let note = match state.resolve_href(&e.locator) {
            Some(_) => String::new(),
            None => {
                unresolvable += 1;
                "  [UNRESOLVABLE: app not in registry]".to_string()
            }
        };
        println!(
            "{star}{}  [{:?}]  {}\n     {}\n     {}{note}",
            e.subject_id.as_str(),
            e.kind,
            e.title,
            e.snippet,
            e.locator.to_uri()
        );
    }
    if unresolvable > 0 {
        println!(
            "\n{unresolvable} entr{} unresolvable — register the app with `atlasctl app-set`",
            if unresolvable == 1 { "y is" } else { "ies are" }
        );
    }
    Ok(())
}

async fn send_delta(cli: &Cli, dir: &Path, records: Vec<SignedRecord>) -> Result<()> {
    send_index_delta(cli, dir, records, None).await
}

async fn send_index_delta(
    cli: &Cli,
    dir: &Path,
    records: Vec<SignedRecord>,
    apps: Option<AppRegistry>,
) -> Result<()> {
    let delta = encode(&IndexDelta {
        key_auth: None,
        records,
        apps,
    })?;
    let params = params_bytes(dir, &cli.slug)?;
    let key = NodeClient::contract_key(CONTRACT_WASM, &params);
    let mut client = NodeClient::connect(&cli.node).await?;
    client.update_delta(key, delta).await
}

/// What a read of the index actually told us.
///
/// Three outcomes, and conflating the last two is a real hazard: `NotFound` from
/// the node means "I could not find this contract", which covers both a
/// never-initialized index AND an index that exists but is momentarily
/// unfindable (the node returns NotFound on retry exhaustion too). Treating that
/// as "no registry yet" is how a curator ends up signing a fresh v1 registry that
/// silently discards the v7 one already on the network.
enum IndexRead {
    /// The node served a state. Boxed only to keep the enum small; a whole index
    /// state dwarfs the other two variants.
    Present(Box<IndexState>),
    /// The node served EMPTY BYTES, which means the contract exists but was never
    /// initialized — `IndexState::initialized` always carries a key_auth, so an
    /// initialized index never encodes to nothing. So this is the same "we know
    /// nothing about the registry" class as `Unfindable`, not a benign empty index.
    Empty,
    /// The node could not find the contract. NOT proof it does not exist.
    Unfindable,
}

/// `fresh` subscribes, which makes the node fetch rather than answer from whatever
/// stale copy it happens to hold. Required on any WRITE path: the node's local copy
/// can lag the network (this CLI's own `push-state` exists because of that), and a
/// stale read is how a registry edit silently discards apps it never saw.
async fn fetch_state(cli: &Cli, dir: &Path, fresh: bool) -> Result<IndexRead> {
    let params = params_bytes(dir, &cli.slug)?;
    let key = NodeClient::contract_key(CONTRACT_WASM, &params);
    let mut client = NodeClient::connect(&cli.node).await?;
    let Some(bytes) = client.get_optional(&key, fresh).await? else {
        return Ok(IndexRead::Unfindable);
    };
    if bytes.is_empty() {
        return Ok(IndexRead::Empty);
    }
    Ok(IndexRead::Present(Box::new(
        ciborium::de::from_reader(&bytes[..]).context("decoding index state")?,
    )))
}

/// Set (or, with `record: None`, remove) one app in the registry.
///
/// Read-modify-write against the live registry rather than a blind overwrite, so
/// re-pointing `delta` cannot silently drop `river`. The version increments off
/// what is actually on the network; a concurrent curator edit would be resolved
/// by the contract's max-by-(version, sig) merge, and the loser is visibly the
/// one whose change is absent from a later `atlasctl apps`.
async fn app_set(
    cli: &Cli,
    dir: &Path,
    slug: &str,
    record: Option<AppRecord>,
    expect_version: Option<u64>,
) -> Result<()> {
    let root = load_key(&dir.join("root.key"))?;
    let params = IndexParams {
        root_vk: root.verifying_key(),
        slug: cli.slug.clone(),
    };
    let current = match fetch_state(cli, dir, true).await? {
        // Refusing here is the whole point of the tri-state. If the node cannot
        // find the index we do not know what registry is on the network, and
        // proceeding would sign a body built from nothing: the contract would
        // reject it as too old (so the edit silently does not happen) or, worse,
        // accept it and drop every app the real registry held.
        IndexRead::Unfindable => bail!(
            "the node could not find this index, so the current registry is \
             unknown. That is NOT proof it does not exist — a node reports the \
             same thing when a GET exhausts its retries. Refusing to sign a \
             registry that could silently discard the apps already registered. \
             Check `atlasctl apps` / `atlasctl show` and retry once the index is \
             reachable."
        ),
        // Refused for the same reason as `Unfindable`: empty bytes mean the index
        // was never initialized, so we know nothing about what registry exists.
        // (Writing anyway happened to be safe only because `apply_delta` refuses
        // without a key_auth — accidental protection, not a reason to rely on it.)
        IndexRead::Empty => bail!(
            "this index is not initialized (the node served an empty state). Run \
             `atlasctl init` first."
        ),
        IndexRead::Present(state) => state.apps,
    };
    // NEVER let the root key sign bytes we have not verified. The fetched state
    // is whatever the node handed back; if it were doctored (hostile `--node`, a
    // node bug, a corrupt local store) its app records would be carried into the
    // new body and signed with the ROOT key, laundering an attacker-chosen
    // contract id into a legitimately root-signed registry. That is precisely
    // the authority root-signing exists to withhold.
    if let Some(cur) = &current {
        cur.verify_for(&params).map_err(|e| {
            anyhow!(
                "refusing to build on the registry this node returned: {e}. \
                 The node may be serving a forged or corrupt state."
            )
        })?;
        cur.check_structure()
            .map_err(|e| anyhow!("refusing to build on a malformed registry: {e}"))?;
    }
    let body = next_registry_body(
        current.map(|a| a.body).as_ref(),
        &cli.slug,
        slug,
        record,
        expect_version,
    )?;
    let registry = AppRegistry {
        sig: sign(&body, &root),
        body,
    };
    registry
        .check_structure()
        .map_err(|e| anyhow!("invalid app registry: {e}"))?;
    // Verify what we just built, so a bug here cannot ship a registry the
    // contract would refuse.
    registry
        .verify_for(&params)
        .map_err(|e| anyhow!("built an invalid registry: {e}"))?;
    let version = registry.body.version;
    let count = registry.body.apps.len();
    send_index_delta(cli, dir, Vec::new(), Some(registry)).await?;
    println!("app registry now at version {version} ({count} app(s))");
    Ok(())
}

/// The pure core of `app-set` / `app-unset`: given the registry currently on the
/// network, produce the next one.
///
/// Split out from the async command so the read-modify-write is testable. The
/// property that matters is that editing one app never drops another, which is the
/// whole reason this reads before writing rather than overwriting.
fn next_registry_body(
    current: Option<&AppRegistryBody>,
    index_slug: &str,
    app: &str,
    record: Option<AppRecord>,
    expect_version: Option<u64>,
) -> Result<AppRegistryBody> {
    let cur_version = current.map_or(0, |b| b.version);
    // A stale local replica is the realistic hazard, not a racing curator: a
    // stale read plus version+1 still wins the merge and silently drops an app
    // registered elsewhere. Once a registry exists, make the operator state the
    // version they believe they are superseding.
    // ALWAYS required, including `--expect-version 0` for the first-ever registry.
    // Only demanding it when `cur_version > 0` left the dangerous case open: a node
    // holding a state with no registry (or an uninitialized one) reports version 0
    // while the network is at v7, so the guard would not fire and the edit would be
    // signed as v1 — losing every app the real registry held.
    match expect_version {
        None => bail!(
            "pass --expect-version {cur_version} to confirm the registry version you \
             are superseding (this node reports {cur_version}). Required even for the \
             first registry (--expect-version 0), because a node that simply has not \
             seen the registry also reports 0."
        ),
        Some(v) if v != cur_version => bail!(
            "registry is at version {cur_version} according to this node, not {v} — \
             re-run `atlasctl apps` and retry with the version you intend to \
             supersede. If they disagree, this node's copy may be stale."
        ),
        Some(_) => {}
    }
    let mut body = current.cloned().unwrap_or_default();
    // Stamped from the INDEX slug, never carried over from the fetched body, so a
    // registry lifted from another index cannot launder its binding through an
    // edit here.
    body.index_slug = index_slug.to_string();
    match record {
        Some(rec) => {
            rec.check()
                .map_err(|e| anyhow!("invalid app record: {e}"))?;
            body.apps.insert(app.to_string(), rec);
        }
        None => {
            if body.apps.remove(app).is_none() {
                bail!("app `{app}` is not registered");
            }
        }
    }
    body.version = cur_version
        .checked_add(1)
        .ok_or_else(|| anyhow!("registry version would overflow"))?;
    Ok(body)
}

/// Render the app registry as JSON for the crawler.
///
/// Hand-rolled so the CLI does not take a serde_json dependency purely for this.
/// Safe ONLY because every interpolated field is charset-validated by
/// `AppRecord::check` / `check_app_slug`: the slug is `[a-z0-9-]`, the contract id
/// is base58, `name` is printable ASCII minus `"` and `\\`, and the template is
/// restricted to URL-suffix characters. If any of those charsets is widened, this
/// must switch to a real serializer. `apps_json_is_valid_json` pins the coupling.
fn apps_json(registry: Option<&AppRegistry>) -> String {
    let Some(registry) = registry else {
        return "{\"version\":0,\"apps\":{}}".to_string();
    };
    let apps = registry
        .body
        .apps
        .iter()
        .map(|(slug, r)| {
            format!(
                "\"{slug}\":{{\"contract_id\":\"{}\",\"name\":\"{}\",\"link_template\":\"{}\"}}",
                r.contract_id, r.name, r.link_template
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"version\":{},\"apps\":{{{apps}}}}}",
        registry.body.version
    )
}

async fn apps(cli: &Cli, dir: &Path, json: bool) -> Result<()> {
    let root = load_key(&dir.join("root.key"))?;
    let params = IndexParams {
        root_vk: root.verifying_key(),
        slug: cli.slug.clone(),
    };
    let registry = match fetch_state(cli, dir, false).await? {
        IndexRead::Unfindable => bail!("the node could not find this index"),
        IndexRead::Empty => None,
        IndexRead::Present(state) => state.apps,
    };
    // Verify before printing. `apps --json` is documented as the crawler's input,
    // and the printer is a hand-rolled serializer that is only safe because the
    // fields are charset-validated — but that validation happens at WRITE time and
    // in the contract, so a node serving a doctored state would otherwise have its
    // unvalidated field values printed straight into the JSON.
    if let Some(reg) = &registry {
        reg.verify_for(&params)
            .map_err(|e| anyhow!("refusing to print an unverified registry: {e}"))?;
        reg.check_structure()
            .map_err(|e| anyhow!("refusing to print a malformed registry: {e}"))?;
    }
    if json {
        println!("{}", apps_json(registry.as_ref()));
        return Ok(());
    }
    let Some(registry) = registry else {
        println!("(no app registry)");
        return Ok(());
    };
    println!("app registry version {}:", registry.body.version);
    for (slug, r) in &registry.body.apps {
        println!(
            "  {slug}  {}  {}\n     template {}",
            r.name, r.contract_id, r.link_template
        );
    }
    Ok(())
}

fn params_bytes(dir: &Path, slug: &str) -> Result<Vec<u8>> {
    let root = load_key(&dir.join("root.key"))?;
    Ok(IndexParams {
        root_vk: root.verifying_key(),
        slug: slug.to_string(),
    }
    .to_bytes())
}

// --- helpers ---

fn encode<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(value, &mut buf)?;
    Ok(buf)
}

fn resolve_key_dir(opt: &Option<PathBuf>) -> PathBuf {
    opt.clone().unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".config/atlas")
    })
}

fn save_key(path: &Path, key: &SigningKey) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, key.to_bytes()).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn load_key(path: &Path) -> Result<SigningKey> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let arr: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("{} is not a 32-byte key", path.display()))?;
    Ok(SigningKey::from_bytes(&arr))
}

fn parse_kind(s: &str) -> Result<Kind> {
    Ok(match s.to_lowercase().as_str() {
        "app" => Kind::App,
        "site" => Kind::Site,
        "external" => Kind::External,
        other => bail!("unknown kind '{other}' (expected app|site|external)"),
    })
}

fn parse_locator(s: &str) -> Result<Locator> {
    if let Some(rest) = s.strip_prefix("app:") {
        // `app:<slug>/<resource>[<path>]`. The resource ends at the first
        // separator, so a deep link (`app:delta/AmcVD92D3U/3/delta-sites`) keeps
        // everything after it as the path.
        let (slug, after) = rest
            .split_once('/')
            .ok_or_else(|| anyhow!("app locator must be `app:<slug>/<resource>[<path>]`"))?;
        let res_end = after.find(['/', '#', '?']).unwrap_or(after.len());
        let loc = Locator::AppResource {
            app: slug.to_string(),
            resource: after[..res_end].to_string(),
            path: after[res_end..].to_string(),
        };
        loc.check().map_err(|e| anyhow!("{e}"))?;
        return Ok(loc);
    }
    if let Some(rest) = s.strip_prefix("freenet:") {
        let is_b58 = |c: char| {
            matches!(c,
                '1'..='9' | 'A'..='H' | 'J'..='N' | 'P'..='Z' | 'a'..='k' | 'm'..='z')
        };
        let id_end = rest.find(|c: char| !is_b58(c)).unwrap_or(rest.len());
        let loc = Locator::Freenet {
            contract_id: rest[..id_end].to_string(),
            path: rest[id_end..].to_string(),
        };
        loc.check().map_err(|e| anyhow!("{e}"))?;
        Ok(loc)
    } else if s.starts_with("https://") {
        let loc = Locator::External { url: s.to_string() };
        loc.check().map_err(|e| anyhow!("{e}"))?;
        Ok(loc)
    } else {
        bail!("locator must start with `freenet:`, `app:` or `https://`")
    }
}

fn split_tags(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn b58(bytes: &[u8]) -> String {
    bs58::encode(bytes).into_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_common::{IndexEntry, Kind, Locator};

    /// The app-slug flag must NOT be named `slug`: `--slug` is a GLOBAL arg for
    /// the INDEX slug, and a subcommand flag of the same name shadowed it, so
    /// `app-set --slug delta` silently retargeted the whole command at a
    /// different index and failed with a confusing "contract not found". Only
    /// found by running it, so pin it.
    fn rec(id: &str, name: &str) -> AppRecord {
        AppRecord {
            contract_id: id.to_string(),
            name: name.to_string(),
            link_template: "/#{resource}{path}".to_string(),
        }
    }
    const ID_A: &str = "EqJ5YpEEV3XLqEvKWLQHFhGAac2qXzSUoE6k2zbdnXBr";
    const ID_B: &str = "771DvtPMwt2PumPyrFvsz7fpvU1gogcmb5qtS1yYEEH9";

    /// Editing one app must never drop another. This is the entire reason
    /// `app-set` reads before writing, and it was previously untested.
    #[test]
    fn re_pointing_one_app_keeps_the_others() {
        let mut cur = AppRegistryBody {
            version: 4,
            index_slug: "default".into(),
            ..Default::default()
        };
        cur.apps.insert("delta".into(), rec(ID_A, "Delta"));
        cur.apps.insert("river".into(), rec(ID_B, "River"));

        let next = next_registry_body(
            Some(&cur),
            "default",
            "delta",
            Some(rec(ID_B, "Delta")),
            Some(4),
        )
        .unwrap();
        assert_eq!(next.version, 5);
        assert_eq!(next.apps.len(), 2, "river must survive");
        assert_eq!(next.apps["delta"].contract_id, ID_B, "delta re-pointed");
        assert_eq!(next.apps["river"].contract_id, ID_B);
        assert_eq!(next.apps["river"].name, "River", "river untouched");
    }

    #[test]
    /// `--expect-version` is required even for the FIRST registry, and must be an
    /// explicit 0. Only demanding it once a registry exists left the dangerous case
    /// open: a node that has merely not SEEN the registry also reports 0, so the
    /// guard would not fire and the edit would be signed as v1, losing every app the
    /// real (v7, say) registry held.
    fn the_first_registry_requires_an_explicit_expect_version_zero() {
        let omitted = next_registry_body(None, "default", "delta", Some(rec(ID_A, "Delta")), None)
            .expect_err("must refuse without --expect-version, even for a first write");
        assert!(omitted.to_string().contains("--expect-version"));

        let next = next_registry_body(None, "default", "delta", Some(rec(ID_A, "Delta")), Some(0))
            .unwrap();
        assert_eq!(next.version, 1);
        assert_eq!(next.index_slug, "default");
    }

    #[test]
    fn expect_version_must_match_the_version_the_node_reports() {
        let cur = AppRegistryBody {
            version: 3,
            index_slug: "default".into(),
            ..Default::default()
        };
        let wrong = next_registry_body(
            Some(&cur),
            "default",
            "delta",
            Some(rec(ID_A, "D")),
            Some(2),
        )
        .expect_err("must refuse a mismatched version");
        let msg = wrong.to_string();
        assert!(
            msg.contains('3') && msg.contains('2'),
            "should name both: {msg}"
        );
    }

    /// The index binding is stamped from the INDEX slug, never carried over from
    /// the fetched body, so an edit cannot launder a foreign registry's binding.
    #[test]
    fn the_index_binding_is_restamped_not_inherited() {
        let cur = AppRegistryBody {
            version: 1,
            index_slug: "staging".into(),
            ..Default::default()
        };
        let next = next_registry_body(
            Some(&cur),
            "default",
            "delta",
            Some(rec(ID_A, "D")),
            Some(1),
        )
        .unwrap();
        assert_eq!(next.index_slug, "default");
    }

    #[test]
    fn unsetting_an_unregistered_app_is_an_error_and_keeps_siblings() {
        let mut cur = AppRegistryBody {
            version: 1,
            index_slug: "default".into(),
            ..Default::default()
        };
        cur.apps.insert("delta".into(), rec(ID_A, "Delta"));
        assert!(next_registry_body(Some(&cur), "default", "river", None, Some(1)).is_err());
        let next = next_registry_body(Some(&cur), "default", "delta", None, Some(1)).unwrap();
        assert!(next.apps.is_empty());
    }

    #[test]
    fn a_version_at_the_maximum_is_an_overflow_error_not_a_panic() {
        let cur = AppRegistryBody {
            version: u64::MAX,
            index_slug: "default".into(),
            ..Default::default()
        };
        let err = next_registry_body(
            Some(&cur),
            "default",
            "delta",
            Some(rec(ID_A, "D")),
            Some(u64::MAX),
        )
        .expect_err("must not wrap");
        assert!(err.to_string().contains("overflow"));
    }

    /// The hand-rolled JSON writer is safe only because the fields are
    /// charset-validated. Pin that it really does emit parseable JSON, so widening
    /// a charset without switching to a serializer fails here.
    #[test]
    fn apps_json_is_valid_json() {
        assert_eq!(apps_json(None), r#"{"version":0,"apps":{}}"#);

        let mut body = AppRegistryBody {
            version: 7,
            index_slug: "default".into(),
            ..Default::default()
        };
        body.apps.insert("delta".into(), rec(ID_A, "Delta"));
        body.apps.insert("river".into(), rec(ID_B, "River Chat"));
        let root = atlas_common::generate_key();
        let reg = AppRegistry {
            sig: sign(&body, &root),
            body,
        };
        let out = apps_json(Some(&reg));
        // BTreeMap ordering makes this deterministic.
        assert_eq!(
            out,
            format!(
                r#"{{"version":7,"apps":{{"delta":{{"contract_id":"{ID_A}","name":"Delta","link_template":"/#{{resource}}{{path}}"}},"river":{{"contract_id":"{ID_B}","name":"River Chat","link_template":"/#{{resource}}{{path}}"}}}}}}"#
            )
        );
        // Structural check independent of the exact bytes: balanced and quoted.
        assert_eq!(out.matches('{').count(), out.matches('}').count());
        assert_eq!(out.matches('"').count() % 2, 0);
    }

    #[test]
    fn app_subcommands_do_not_shadow_the_global_index_slug() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        for name in ["app-set", "app-unset"] {
            let sub = cmd
                .get_subcommands()
                .find(|c| c.get_name() == name)
                .unwrap_or_else(|| panic!("{name} subcommand missing"));
            let shadows = sub
                .get_arguments()
                .any(|a| a.get_id() == "slug" && !a.is_global_set());
            assert!(
                !shadows,
                "{name} defines its own `--slug`, which shadows the global index slug"
            );
        }
    }

    #[test]
    fn parse_locator_accepts_the_app_form_and_round_trips() {
        for uri in [
            "app:delta/AmcVD92D3U",
            "app:delta/AmcVD92D3U/3/delta-sites",
            "app:delta/AmcVD92D3U#frag",
            "app:delta/AmcVD92D3U?q=1",
        ] {
            let loc = parse_locator(uri).unwrap_or_else(|e| panic!("{uri}: {e}"));
            assert_eq!(loc.to_uri(), uri, "round-trip failed for {uri}");
            match &loc {
                Locator::AppResource { app, resource, .. } => {
                    assert_eq!(app, "delta");
                    assert_eq!(resource, "AmcVD92D3U");
                }
                other => panic!("{uri} parsed as {other:?}"),
            }
        }
    }

    #[test]
    fn parse_locator_rejects_malformed_app_uris() {
        for bad in [
            "app:delta",            // no `/`
            "app:delta/",           // empty resource
            "app:delta/#x",         // empty resource
            "app:/AmcVD92D3U",      // empty slug
            "app:Delta/AmcVD92D3U", // uppercase slug
            "app:delta/has space",
            "app:delta/AmcVD92D3U/../x", // traversal
            "ftp://example.com",
        ] {
            assert!(parse_locator(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    /// The dedup key an app-hosted locator is stored under must ignore the path,
    /// which is what makes two pages of one Delta site a single listing.
    #[test]
    fn app_locators_for_one_site_share_a_dedup_key() {
        let a = parse_locator("app:delta/AmcVD92D3U/1/home").unwrap();
        let b = parse_locator("app:delta/AmcVD92D3U/3/delta-sites").unwrap();
        let other = parse_locator("app:delta/Fe5jaFmRnp/1/about").unwrap();
        assert_eq!(a.dedup_key(), b.dedup_key());
        assert_ne!(a.dedup_key(), other.dedup_key());
    }

    fn live_record(sid: SubjectId) -> SignedRecord {
        let key = generate_key();
        let body = RecordBody::Live(IndexEntry {
            subject_id: sid,
            version: 1,
            kind: Kind::App,
            title: "t".to_string(),
            snippet: String::new(),
            tags: vec![],
            locator: Locator::External {
                url: "https://example.com".to_string(),
            },
            featured: false,
            added_at: 0,
        });
        SignedRecord {
            sig: sign(&body, &key),
            by: key.verifying_key(),
            body,
        }
    }

    fn tomb_record(sid: SubjectId) -> SignedRecord {
        let key = generate_key();
        let body = RecordBody::Tomb(Tombstone {
            subject_id: sid,
            version: 2,
        });
        SignedRecord {
            sig: sign(&body, &key),
            by: key.verifying_key(),
            body,
        }
    }

    /// CBOR-encode a state holding the given records (key_auth omitted — this only
    /// exercises record counting/classification, not signature verification).
    fn encoded_state(records: Vec<SignedRecord>) -> Vec<u8> {
        let mut st = IndexState::default();
        for r in records {
            st.records.insert(r.body.subject_id().clone(), r);
        }
        encode(&st).unwrap()
    }

    #[test]
    fn count_records_distinguishes_live_and_tombstone() {
        let bytes = encoded_state(vec![
            live_record(SubjectId::random()),
            live_record(SubjectId::random()),
            tomb_record(SubjectId::random()),
        ]);
        assert_eq!(count_records(&bytes).unwrap(), (2, 1));
    }

    /// An undecodable generation must be an ERROR. It used to read as
    /// `(0, 0)` — printed as "holds 0 live / 0 tombstone record(s)", which looks
    /// exactly like a harmlessly empty generation — and was then PUT anyway.
    #[test]
    fn an_undecodable_generation_is_an_error_not_empty() {
        assert!(count_records(b"not cbor at all").is_err());
        assert!(plan_generation(Ok(b"not cbor at all".to_vec())).is_err());
    }

    #[test]
    fn count_records_empty_bytes_is_zero() {
        assert_eq!(count_records(&[]).unwrap(), (0, 0));
    }

    /// M1: a legacy GET *error* must surface as an `Err` (which the caller turns
    /// into a real-run abort or a dry-run warning), never be folded into an empty
    /// state — which would let a rebuild land on an empty index while entries
    /// still live at the old address.
    #[test]
    fn plan_generation_surfaces_get_error() {
        let outcome: Result<Vec<u8>> = Err(anyhow!("simulated dead-end / timeout"));
        assert!(
            plan_generation(outcome).is_err(),
            "a GET error must stay an Err, not be skipped as empty"
        );
    }

    /// A *successful* but empty GET is a legitimate skip: nothing lives at that
    /// address, so there is nothing to carry forward.
    #[test]
    fn plan_generation_skips_ok_empty() {
        assert!(plan_generation(Ok(Vec::new())).unwrap().is_none());
    }

    /// A tombstone-only generation is non-empty and must still be carried forward
    /// so its takedowns propagate into the current index.
    #[test]
    fn plan_generation_merges_tombstone_only() {
        let bytes = encoded_state(vec![tomb_record(SubjectId::random())]);
        let (_, live, tomb) = plan_generation(Ok(bytes))
            .unwrap()
            .expect("a tombstone-only generation must be carried forward, not skipped");
        assert_eq!((live, tomb), (0, 1));
    }
}
