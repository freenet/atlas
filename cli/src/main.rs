//! atlasctl: the Atlas curator CLI. Manages the single-writer index contract on
//! a Freenet node. The root key authorizes an online signing key; entries are
//! signed by the online key and merged into the index by per-subject version.

mod api;
mod migration;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use atlas_common::{
    generate_key, sign, AppRecord, AppRegistry, AppRegistryBody, Audience, Classification,
    IndexDelta, IndexEntry, IndexParams, IndexState, KeyAuth, KeyAuthBody, Kind, Locator,
    RecordBody, SignedRecord, SubjectId, Tombstone, Verification, VerifyStatus, Volatility,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
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

/// `--landing`, as a CLI spelling.
///
/// A local mirror of [`Audience`] rather than `#[derive(ValueEnum)]` on the
/// schema type: `common/` compiles into the contract WASM, and the index address
/// is `hash(compiled_wasm, params)`, so adding a clap dependency there to satisfy
/// a CLI concern would re-key the live index. Both prior re-keys left the UI
/// serving a stale snapshot, once for eight days. The crate stays clap-free on
/// purpose; `classification_flag_values_parse` pins that this mirror still
/// accepts the documented spellings.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum LandingArg {
    General,
    Adult,
}

impl From<LandingArg> for Audience {
    fn from(v: LandingArg) -> Self {
        match v {
            LandingArg::General => Audience::new(Audience::GENERAL),
            LandingArg::Adult => Audience::new(Audience::ADULT),
        }
    }
}

/// `--volatility`, as a CLI spelling. See [`LandingArg`] for why it is mirrored.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum VolatilityArg {
    Static,
    Feed,
}

impl From<VolatilityArg> for Volatility {
    fn from(v: VolatilityArg) -> Self {
        match v {
            VolatilityArg::Static => Volatility::new(Volatility::STATIC),
            VolatilityArg::Feed => Volatility::new(Volatility::FEED),
        }
    }
}

/// The classification flags. `#[command(flatten)]`-ed into BOTH `add` and
/// `update` so the two spellings cannot drift apart; a curator who learns them on
/// one command must not be surprised by the other.
/// (`add_and_update_share_the_classification_flag_spelling` pins that.)
#[derive(Args, Clone, Debug, Default)]
struct ClassArgs {
    /// What a visitor lands on immediately: `general` or `adult`.
    ///
    /// Passing this is what MINTS a classification. An entry has none until
    /// then, and "none" means NOT ASSESSED — deliberately distinguishable from
    /// "assessed and found general-audience". The other flags here only REFINE a
    /// judgement; on an entry that has none they are refused rather than being
    /// completed with a fabricated landing audience. See `next_class`.
    #[arg(long, value_enum)]
    landing: Option<LandingArg>,
    /// Whether adult material exists deeper in, behind navigation or a gate.
    /// `--landing general --adult-sections true` IS the gated case.
    #[arg(long)]
    adult_sections: Option<bool>,
    /// `static` (the description keeps describing the resource) or `feed` (a
    /// live feed, so a frozen description goes stale within hours).
    #[arg(long, value_enum)]
    volatility: Option<VolatilityArg>,
    /// Which classifier taxonomy produced this judgement. Defaults to
    /// [`HAND_CLASSIFIER`].
    #[arg(long)]
    classifier: Option<u16>,
    /// Additive metadata the contract carries but never interprets, as
    /// `key=value`. Repeatable. MERGES into whatever the entry already holds.
    #[arg(long, value_name = "KEY=VALUE")]
    ext: Vec<String>,
    /// Drop every existing `ext` key before applying `--ext`.
    ///
    /// Merge is the default because `ext` is a shared surface: the crawler and
    /// the curator both write keys there, and a replace-by-default would mean
    /// whichever ran last silently deleted the other's. This flag is the
    /// deliberate escape hatch, and is also the only way to remove a single key
    /// (clear, then restate the survivors).
    #[arg(long)]
    ext_clear: bool,
}

impl ClassArgs {
    /// True when the caller named nothing here. Distinct from "named nothing
    /// that changes anything": a flag restating the current value still counts,
    /// because deciding an edit is a no-op is exactly the optimisation that
    /// would skip the version bump. See `next_entry`.
    fn is_empty(&self) -> bool {
        self.landing.is_none()
            && self.adult_sections.is_none()
            && self.volatility.is_none()
            && self.classifier.is_none()
            && self.ext.is_empty()
            && !self.ext_clear
    }
}

/// `--verified`, as a CLI spelling. See [`LandingArg`] for why it is mirrored
/// rather than deriving `ValueEnum` on the schema type directly.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum VerifyStatusArg {
    Live,
    Unreachable,
    Changed,
}

impl From<VerifyStatusArg> for VerifyStatus {
    fn from(v: VerifyStatusArg) -> Self {
        match v {
            VerifyStatusArg::Live => VerifyStatus::new(VerifyStatus::LIVE),
            VerifyStatusArg::Unreachable => VerifyStatus::new(VerifyStatus::UNREACHABLE),
            VerifyStatusArg::Changed => VerifyStatus::new(VerifyStatus::CHANGED),
        }
    }
}

/// The per-field edits `update` applies. Every field is optional, and an unset
/// one is CARRIED FORWARD from the current entry rather than reset to a default.
#[derive(Args, Clone, Debug, Default)]
struct EntryEdits {
    /// Replacement title.
    #[arg(long)]
    title: Option<String>,
    /// Pass `--snippet ""` to clear it.
    #[arg(long)]
    snippet: Option<String>,
    /// Comma-separated, REPLACING the existing list (`--tags ""` clears them).
    /// Replacing rather than merging because a tag list is one editorial
    /// statement, and there would otherwise be no way to drop a tag at all.
    #[arg(long)]
    tags: Option<String>,
    /// `--featured true|false`. Takes an explicit value, unlike `add --featured`
    /// (a bare flag): on an update, "flag absent" has to mean "leave it alone",
    /// and a bare flag cannot express that.
    #[arg(long)]
    featured: Option<bool>,
    /// Stamps `verified` directly, timestamped `now`. FOR THE CRAWLER'S
    /// re-verification sweep, not for a human: a person asserting `live`
    /// without having actually re-fetched the resource is exactly the false
    /// confidence this field exists to prevent. Takes priority over the
    /// description-changed-clears-`verified` rule below, since a caller passing
    /// this explicitly just DID the check the field is meant to record — most
    /// often alongside a `title`/`snippet` correction from the same re-check,
    /// where the ordinary rule would otherwise immediately clear what this flag
    /// just set.
    #[arg(long, value_enum)]
    verified: Option<VerifyStatusArg>,
    #[command(flatten)]
    class: ClassArgs,
}

impl EntryEdits {
    fn is_empty(&self) -> bool {
        self.title.is_none()
            && self.snippet.is_none()
            && self.tags.is_none()
            && self.featured.is_none()
            && self.verified.is_none()
            && self.class.is_empty()
    }
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
        /// Optional. Omitting every one of these mints an UNCLASSIFIED entry,
        /// which is the honest state for something nobody has looked at yet.
        #[command(flatten)]
        class: ClassArgs,
    },
    /// Tombstone a subject by id (needs the current version to supersede it).
    Remove {
        #[arg(long)]
        subject: String,
        #[arg(long)]
        cur_version: u64,
    },
    /// Re-describe or (re)classify a live subject: mints `cur_version + 1`
    /// carrying forward every field not named on the command line.
    ///
    /// Without this, an entry is minted once by `add` and can only ever be
    /// tombstoned — a stale description or a new classification cannot be
    /// applied at all.
    ///
    /// `--kind` and `--locator` are deliberately NOT editable. The locator is
    /// what the subject IS (it is the dedup key), so editing it silently
    /// re-points an established subject id at a different resource while every
    /// client keeps treating it as the same thing. `remove` the entry and `add`
    /// the new one instead, so the identity change is visible.
    Update {
        /// The subject id to re-describe (from `atlasctl show`).
        #[arg(long)]
        subject: String,
        /// The version you believe you are superseding, as for `remove`.
        ///
        /// Unlike `remove` — which never reads state, so it can only take this
        /// on trust — `update` must read the entry to carry its fields forward,
        /// and therefore checks the value against what the network actually
        /// holds.
        #[arg(long)]
        cur_version: u64,
        #[command(flatten)]
        edits: EntryEdits,
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
        /// Emit machine-readable JSON including each entry's VERSION, which the
        /// human listing omits. `atlasctl remove` needs the current version to
        /// supersede a subject, so without this a bulk removal has to guess it.
        #[arg(long)]
        json: bool,
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
            class,
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
                class,
            )
            .await
        }
        Cmd::Remove {
            subject,
            cur_version,
        } => remove(&cli, &dir, subject, *cur_version).await,
        Cmd::Update {
            subject,
            cur_version,
            edits,
        } => update(&cli, &dir, subject, *cur_version, edits).await,
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
        Cmd::Show { subscribe, json } => show(&cli, &dir, *subscribe, *json).await,
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
    class: &ClassArgs,
) -> Result<()> {
    let online = load_key(&dir.join("online.key"))?;
    // Every purely-local parse happens BEFORE the duplicate-check round trip, so
    // a mistyped `--kind` or a malformed `--ext` fails immediately instead of
    // after a network GET that can itself take a while or dead-end.
    let locator = parse_locator(locator)?;
    let kind = parse_kind(kind)?;
    let now = now_secs();
    // Unclassified unless the curator classified it on THIS command line. A
    // freshly-added entry is NOT ASSESSED, and that has to stay distinguishable
    // from assessed-and-general; see `next_class`.
    let classification = next_class(None, class, now)?;
    let ext = next_ext(None, class)?;
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
        kind,
        title: title.to_string(),
        snippet: snippet.to_string(),
        tags: split_tags(tags),
        locator,
        featured,
        added_at: now,
        class: classification,
        // Only the crawler sets this: nothing has re-checked an entry minted a
        // moment ago, and `None` says exactly that.
        verified: None,
        ext,
    };
    let subject = entry.subject_id.as_str().to_string();
    let rec = signed_live_record(entry, &online)?;
    send_delta(cli, dir, vec![rec]).await?;
    println!("added subject {subject}");
    Ok(())
}

/// Validate locally BEFORE signing and sending, then sign with the online key.
///
/// Shared by `add` and `update` so the pre-flight cannot be wired into one and
/// forgotten on the other. The contract enforces the same rules, but a rejection
/// there arrives as an opaque `InvalidUpdateWithInfo` from the node; checking
/// here names the offending field.
fn signed_live_record(entry: IndexEntry, online: &SigningKey) -> Result<SignedRecord> {
    let body = RecordBody::Live(entry);
    body.check_structure()
        .map_err(|e| anyhow!("entry would be rejected by the contract: {e}"))?;
    Ok(SignedRecord {
        sig: sign(&body, online),
        by: online.verifying_key(),
        body,
    })
}

/// `Classification::classifier` for a judgement made by a person running
/// `atlasctl`, as opposed to any automated taxonomy.
///
/// RESERVED: an automated classifier must number itself from 1. The field exists
/// so a later taxonomy change can find and re-run only ITS stale labels, and a
/// classifier that also claimed 0 would sweep up hand judgements it never made.
const HAND_CLASSIFIER: u16 = 0;

/// Resolve the `class` field for a new or edited entry.
///
/// `current` is what the entry already carries: `None` for `add`, or for one of
/// the entries minted before the taxonomy existed.
///
/// Two rules carry the weight here:
///
/// 1. NO classification flag leaves `current` untouched, INCLUDING its
///    `classified_at`. Refreshing that timestamp on a title fix would claim a
///    re-assessment that never happened, and `classified_at` is what a later
///    taxonomy sweep uses to decide which labels are stale.
/// 2. A refining flag on an entry with no existing classification is REFUSED
///    rather than completed with a default. There is no honest default for
///    `landing`: filling in `General` is precisely the "assessed and found
///    general-audience" claim that an absent `class` exists to avoid making by
///    accident, and this contract is world-readable.
fn next_class(
    current: Option<&Classification>,
    args: &ClassArgs,
    now: u64,
) -> Result<Option<Classification>> {
    if args.landing.is_none()
        && args.adult_sections.is_none()
        && args.volatility.is_none()
        && args.classifier.is_none()
    {
        return Ok(current.cloned());
    }
    let Some(landing) = args
        .landing
        .map(Audience::from)
        .or(current.map(|c| c.landing.clone()))
    else {
        bail!(
            "--landing is required to classify an entry that has never been \
             classified: the other classification flags only refine an existing \
             judgement. An absent `class` means NOT ASSESSED, and defaulting the \
             landing audience would publish a claim nobody made."
        );
    };
    Ok(Some(Classification {
        landing,
        has_adult_sections: args
            .adult_sections
            .or(current.map(|c| c.has_adult_sections))
            .unwrap_or(false),
        volatility: args
            .volatility
            .map(Volatility::from)
            .or(current.map(|c| c.volatility.clone()))
            .unwrap_or(Volatility::new(Volatility::STATIC)),
        classifier: args
            .classifier
            .or(current.map(|c| c.classifier))
            .unwrap_or(HAND_CLASSIFIER),
        // ANY classification flag is a fresh assessment, so the timestamp moves
        // with it. Rule 1 above is what keeps it still otherwise.
        classified_at: now,
    }))
}

/// Resolve the `ext` map for a new or edited entry.
///
/// Merges into `current` unless `--ext-clear` (see the flag's own note on why
/// merge is the default). Only PARSING lives here; the size and character bounds
/// stay with `check_structure`, which is what the contract itself runs.
///
/// An empty result collapses to `None` rather than `Some({})`. The two mean the
/// same thing but serialize differently, and only `None` reproduces the
/// pre-`ext` byte layout that every existing signature was minted over — the
/// property `legacy_shaped_entry_signature_still_verifies` pins.
fn next_ext(
    current: Option<&BTreeMap<String, String>>,
    args: &ClassArgs,
) -> Result<Option<BTreeMap<String, String>>> {
    let mut out = if args.ext_clear {
        BTreeMap::new()
    } else {
        current.cloned().unwrap_or_default()
    };
    let mut seen = BTreeSet::new();
    for pair in &args.ext {
        // `split_once`, not `split`: a VALUE may legitimately contain `=`
        // (base64 padding, a query string). Only the key may not.
        let (k, v) = pair
            .split_once('=')
            .ok_or_else(|| anyhow!("--ext must be `key=value`, got {pair:?}"))?;
        // A key repeated on ONE command line is a typo, not an intent to
        // overwrite: silently keeping the last would discard the other value
        // with no indication. (Overwriting a key the ENTRY already holds is
        // different, and is the point of merging.)
        if !seen.insert(k) {
            bail!("--ext key {k:?} given more than once on this command line");
        }
        out.insert(k.to_string(), v.to_string());
    }
    Ok(if out.is_empty() { None } else { Some(out) })
}

/// The pure core of `update`: given the entry currently on the network, produce
/// the next one. Split out from the async command so the read-modify-write is
/// testable, the same way `next_registry_body` is.
///
/// THE VERSION BUMP IS NOT OPTIONAL AND NOT CONDITIONAL ON THE BODY HAVING
/// CHANGED. `IndexSummary` carries per-subject VERSIONS only, and freenet-core's
/// `plan_fanout_send` returns `Skip` when two peers' summary bytes are identical,
/// before any delta is ever requested. So two peers holding DIFFERENT bodies at
/// the SAME version summarise identically, never exchange, and diverge
/// permanently — no amount of loosening the delta gate could fix it, because the
/// delta is never asked for. (`IndexSummary::apps_fingerprint` exists to close
/// exactly this hole for the app registry, which has no per-subject version to
/// bump.) Hence: a changed body always goes out at a strictly higher version, and
/// "did anything really change?" must never become a reason to skip the bump.
fn next_entry(
    current: &IndexEntry,
    cur_version: u64,
    edits: &EntryEdits,
    now: u64,
) -> Result<IndexEntry> {
    if current.version != cur_version {
        bail!(
            "subject {} is at version {} according to this node, not {cur_version} — \
             re-run `atlasctl show --json` and retry with the version you intend to \
             supersede. If they disagree, this node's copy may be stale.",
            current.subject_id.as_str(),
            current.version
        );
    }
    if edits.is_empty() {
        bail!(
            "nothing to change — pass at least one field flag. Refused rather than \
             re-signing an identical body at a higher version, which every peer \
             holding the index would then have to fetch for no reason."
        );
    }
    let title = edits.title.clone().unwrap_or_else(|| current.title.clone());
    let snippet = edits
        .snippet
        .clone()
        .unwrap_or_else(|| current.snippet.clone());
    // A DESCRIPTION change invalidates the crawler's verification. `verified`
    // asserts the crawler confirmed that THIS ENTRY still describes the
    // resource, and it never saw the new text, so carrying `Live` forward would
    // publish a confirmation nobody gave. Nothing is lost — the crawler
    // re-verifies on its own cadence, and `None` is the same state a
    // freshly-added entry is in. The other edits (tags, featured, class) do not
    // touch what was verified, so they keep it.
    let description_changed = title != current.title || snippet != current.snippet;
    Ok(IndexEntry {
        subject_id: current.subject_id.clone(),
        version: cur_version
            .checked_add(1)
            .ok_or_else(|| anyhow!("entry version would overflow"))?,
        // `kind` and `locator` carry forward verbatim: neither is editable, see
        // the note on `Cmd::Update`.
        kind: current.kind.clone(),
        title,
        snippet,
        tags: edits
            .tags
            .as_deref()
            .map(split_tags)
            .unwrap_or_else(|| current.tags.clone()),
        locator: current.locator.clone(),
        featured: edits.featured.unwrap_or(current.featured),
        // The mint time of the SUBJECT, not of this version, so it does not move.
        // `show` sorts on it, and bumping it would reshuffle the listing on every
        // typo fix.
        added_at: current.added_at,
        class: next_class(current.class.as_ref(), &edits.class, now)?,
        verified: if let Some(status) = edits.verified {
            Some(Verification {
                last_verified_at: now,
                status: status.into(),
            })
        } else if description_changed {
            None
        } else {
            current.verified.clone()
        },
        ext: next_ext(current.ext.as_ref(), &edits.class)?,
    })
}

/// Re-describe or (re)classify a live subject.
///
/// Read-modify-write against the live index, like `app-set`: the new body has to
/// carry forward every field the caller did not name, and those can only come
/// from the current entry. `fresh = true` for the same reason `app-set` uses it —
/// the node's local copy can lag the network (this CLI's own `push-state` exists
/// because of that), and building on a stale read silently reverts whatever the
/// stale copy is missing.
async fn update(
    cli: &Cli,
    dir: &Path,
    subject: &str,
    cur_version: u64,
    edits: &EntryEdits,
) -> Result<()> {
    let online = load_key(&dir.join("online.key"))?;
    let root_vk = load_key(&dir.join("root.key"))?.verifying_key();
    let subject_id = SubjectId::parse(subject).ok_or_else(|| anyhow!("malformed subject id"))?;
    // Checked here as well as inside `next_entry`, not instead of it. `next_entry`
    // owns the invariant (and is where it is tested); this is purely so an
    // argument-less invocation fails at once rather than after a fresh GET, which
    // subscribes and can take a while or dead-end.
    if edits.is_empty() {
        bail!("nothing to change — pass at least one field flag");
    }
    let state = match fetch_state(cli, dir, true).await? {
        IndexRead::Unfindable => bail!(
            "the node could not find this index, so the current entry is unknown. \
             That is NOT proof it does not exist — a node reports the same thing \
             when a GET exhausts its retries. Refusing to sign an update built from \
             nothing, which would drop every field it was supposed to carry \
             forward. Retry once the index is reachable."
        ),
        IndexRead::Empty => bail!(
            "this index is not initialized (the node served an empty state). Run \
             `atlasctl init` first."
        ),
        IndexRead::Present(state) => state,
    };
    // NEVER build on a record we have not verified. The node hands back whatever
    // it holds; if that were doctored (hostile `--node`, a node bug, a corrupt
    // local store) its fields would be carried into the new body and signed with
    // the ONLINE key. A `--title` typo fix would then launder an attacker's
    // locator, or an "adult -> general" relabel, into a legitimately signed
    // entry. Same hazard `app-set` guards against for the root key, one authority
    // level down — and one this command opens for the first time, because it is
    // the first write path that carries network-supplied fields forward.
    let key_auth = state
        .key_auth
        .as_ref()
        .ok_or_else(|| anyhow!("the index this node served carries no key_auth"))?;
    key_auth.verify_sig(&root_vk).map_err(|e| {
        anyhow!("refusing to build on a state whose key_auth is not root-signed: {e}")
    })?;
    let record = state.records.get(&subject_id).ok_or_else(|| {
        anyhow!("no record for subject {subject} in the index served by this node")
    })?;
    if !key_auth.authorizes(&record.by) {
        bail!("the record for {subject} is signed by a key this index does not authorize");
    }
    record
        .verify_sig()
        .map_err(|e| anyhow!("refusing to build on the record for {subject}: {e}"))?;
    // The signature covers the BODY, not the map key it was filed under, so a
    // valid signature does not establish that this record is the subject we
    // asked for. Without this, a node serving a genuine record for subject A
    // under key B makes `update --subject B` sign and publish a new version of
    // A: `next_entry` carries `current.subject_id` forward, so the edit lands on
    // a subject the curator never named, while the CLI reports success for B.
    //
    // `IndexState::verify` makes exactly this check (`common/src/state.rs`), and
    // `preflight_generation` repeats it on the migration path — but neither runs
    // here, because `fetch_state` deliberately does not verify.
    if record.body.subject_id() != &subject_id {
        bail!(
            "the node served a record for subject {} under the key for {subject}; \
             refusing to edit a subject that was not named",
            record.body.subject_id().as_str()
        );
    }
    let RecordBody::Live(current) = &record.body else {
        bail!(
            "subject {subject} is tombstoned — a takedown is final for that subject \
             id. `add` the resource again if it should be listed."
        )
    };
    let entry = next_entry(current, cur_version, edits, now_secs())?;
    let version = entry.version;
    let rec = signed_live_record(entry, &online)?;
    send_delta(cli, dir, vec![rec]).await?;
    println!("updated subject {subject} to version {version}");
    Ok(())
}

async fn remove(cli: &Cli, dir: &Path, subject: &str, cur_version: u64) -> Result<()> {
    let online = load_key(&dir.join("online.key"))?;
    let subject_id = SubjectId::parse(subject).ok_or_else(|| anyhow!("malformed subject id"))?;
    let body = RecordBody::Tomb(Tombstone {
        subject_id,
        // `checked_add`, matching `next_entry` and `next_registry_body`: a
        // wrapping bump would mint version 0, which `check_structure` rejects
        // outright, so the failure would surface as an opaque contract rejection
        // rather than as the arithmetic problem it is.
        version: cur_version
            .checked_add(1)
            .ok_or_else(|| anyhow!("tombstone version would overflow"))?,
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

async fn show(cli: &Cli, dir: &Path, subscribe: bool, json: bool) -> Result<()> {
    let params = params_bytes(dir, &cli.slug)?;
    let key = NodeClient::contract_key(CONTRACT_WASM, &params);
    let mut client = NodeClient::connect(&cli.node).await?;
    let bytes = client.get(&key, subscribe).await?;
    if bytes.is_empty() {
        if json {
            println!("[]");
        } else {
            println!("(index is empty / not initialized)");
        }
        return Ok(());
    }
    let state: IndexState =
        ciborium::de::from_reader(&bytes[..]).context("decoding index state")?;
    let mut entries: Vec<_> = state.live_entries().collect();
    entries.sort_by_key(|e| std::cmp::Reverse(e.added_at));
    if json {
        // Hand-built field by field (the derive would emit the internal shape,
        // including `SubjectId`'s newtype and the `Locator` enum tagging), so a
        // new schema field has to be added HERE too or it silently never
        // reaches a consumer.
        //
        // Unclassified is emitted as an explicit `null`, NOT by omitting the
        // key. Omission conflates two different facts a consumer has to tell
        // apart: "this entry has not been classified" and "the atlasctl that
        // produced this JSON predates classification". With the key always
        // present, its presence proves the field is reported and `null` proves
        // the entry carries no judgement. Same for `verified` and `ext`.
        let rows: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| {
                serde_json::json!({
                    "subject_id": e.subject_id.as_str(),
                    "version": e.version,
                    "kind": e.kind.as_str(),
                    "title": e.title,
                    "locator": e.locator.to_uri(),
                    "featured": e.featured,
                    "added_at": e.added_at,
                    "class": e.class.as_ref().map(|c| serde_json::json!({
                        "landing": c.landing.as_str(),
                        "has_adult_sections": c.has_adult_sections,
                        "volatility": c.volatility.as_str(),
                        "classifier": c.classifier,
                        "classified_at": c.classified_at,
                    })),
                    "verified": e.verified.as_ref().map(|v| serde_json::json!({
                        "last_verified_at": v.last_verified_at,
                        "status": v.status.as_str(),
                    })),
                    "ext": e.ext,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    println!("{} live entries:", entries.len());
    let mut unresolvable = 0;
    let mut unclassified = 0;
    for e in entries {
        if e.class.is_none() {
            unclassified += 1;
        }
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
            "{star}{}  [{}]  {}\n     {}\n     {}{note}\n{}",
            e.subject_id.as_str(),
            e.kind.as_str(),
            e.title,
            e.snippet,
            e.locator.to_uri(),
            entry_metadata_lines(e)
        );
    }
    if unresolvable > 0 {
        println!(
            "\n{unresolvable} entr{} unresolvable — register the app with `atlasctl app-set`",
            if unresolvable == 1 { "y is" } else { "ies are" }
        );
    }
    if unclassified > 0 {
        println!(
            "\n{unclassified} entr{} not classified — `atlasctl update --subject <id> \
             --cur-version <n> --landing general|adult`",
            if unclassified == 1 { "y is" } else { "ies are" }
        );
    }
    Ok(())
}

/// The classification / verification / `ext` lines under an entry in `show`.
///
/// The UNCLASSIFIED case is printed explicitly rather than omitted: unclassified
/// entries are the curator's work queue, and a listing that simply says nothing
/// about them makes that queue invisible — which is how ~87 entries reached the
/// index with no judgement attached in the first place. `verified` and `ext` get
/// no line when absent, because their absence is not a call to action.
///
/// Safe to print raw for the same reason `title` and `snippet` are:
/// `check_structure` rejects control characters in every one of these fields, so
/// nothing here can carry a terminal escape.
fn entry_metadata_lines(e: &IndexEntry) -> String {
    // The vocabularies are OPEN, so these render whatever value the entry
    // carries rather than mapping a closed set. A tag this build has never heard
    // of prints as itself, which is the point: an unknown value must stay
    // legible to a curator instead of becoming a panic or a wrong label.
    let mut out = match &e.class {
        None => "     class: NOT CLASSIFIED".to_string(),
        Some(c) => format!(
            "     class: {} landing, {} adult sections, {} (classifier {}, at {})",
            c.landing.as_str(),
            if c.has_adult_sections { "has" } else { "no" },
            c.volatility.as_str(),
            c.classifier,
            c.classified_at
        ),
    };
    if let Some(v) = &e.verified {
        out.push_str(&format!(
            "\n     verified: {} at {}",
            v.status.as_str(),
            v.last_verified_at
        ));
    }
    if let Some(ext) = &e.ext {
        let pairs = ext
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("  ");
        out.push_str(&format!("\n     ext: {pairs}"));
    }
    out
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

/// Only `app` and `site` are mintable. Every other declared `Kind` is refused
/// here while the READER stays exhaustive, which is the whole shape of this
/// guard: `Kind` is an externally-tagged CBOR enum with no catch-all, so a
/// reader that has never heard of a variant fails to decode the WHOLE state
/// rather than that one record. Declaring a variant early keeps every deployed
/// reader total; gating the writer is what stops one reaching the index before
/// clients know what to do with it.
fn parse_kind(s: &str) -> Result<Kind> {
    Ok(match s.to_lowercase().as_str() {
        "app" => Kind::new(Kind::APP),
        "site" => Kind::new(Kind::SITE),
        // `External` remains readable for existing entries, but the
        // curator can no longer mint one -- see `parse_locator`.
        "external" => bail!(
            "kind 'external' is retired: Atlas indexes Freenet, not the web. \
             Existing entries stay readable (and removable) but no new one may be \
             minted. Use app|site."
        ),
        // Refused for a DIFFERENT reason than `external`, and the message has to
        // say so: these are not retired, they are not designed yet. Their
        // per-kind Open semantics do not exist, so an entry minted with one could
        // not be opened by any client -- and an entry, once published, cannot be
        // un-minted, only tombstoned.
        //
        // `kind` is an OPEN vocabulary on the wire (a string, not an enum), so
        // any value already decodes everywhere and nothing needs declaring. This
        // gate is therefore purely a curator-side policy: opening one of these
        // later is an edit to this function alone, with no schema change and so
        // no contract re-key.
        "room" | "document" | "media" | "feed" => bail!(
            "kind '{s}' is reserved for future per-kind Open semantics: the wire \
             format accepts any kind, but nothing knows how to open such an entry \
             yet, and a published entry can only be tombstoned, never un-minted. \
             Use app|site."
        ),
        other => bail!("unknown kind '{other}' (expected app|site)"),
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
    } else if s.starts_with("https://") || s.starts_with("http://") {
        // Atlas indexes Freenet, not the web. The crawler stopped capturing
        // off-Freenet links at `normalize_href`; refusing here closes the other
        // half, the curator's own hand-add. Without this the CLI remains a way
        // to put a `Locator::External` into the index by hand, which is how the
        // entries this policy exists to remove got there in the first place.
        //
        // `Locator::External` is deliberately still PARSEABLE (it stays in the
        // schema) so existing entries can be read and tombstoned -- `remove`
        // takes a subject id, not a locator, so this refusal does not block the
        // purge.
        bail!(
            "off-Freenet locators are not indexed: Atlas indexes Freenet, not the web. \
             Use a `freenet:` or `app:` locator."
        )
    } else {
        bail!("locator must start with `freenet:` or `app:`")
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
    use atlas_common::{IndexEntry, Kind, Locator, Verification, VerifyStatus};

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

    /// The curator's hand-add is the OTHER way an off-Freenet entry can reach
    /// the index -- the crawler's discovery paths were closed at
    /// `normalize_href`, but this one is a person typing a URL. Both halves have
    /// to refuse or the policy is only half-enforced.
    #[test]
    fn parse_locator_refuses_off_freenet_urls() {
        for uri in [
            "https://example.com/",
            "https://freenet.org",
            "http://example.com/",
        ] {
            let err = parse_locator(uri).expect_err("must refuse");
            assert!(
                err.to_string().contains("Atlas indexes Freenet"),
                "{uri}: message should say why, got {err}"
            );
        }
    }

    /// `External` stays readable so existing entries parse and can be
    /// tombstoned, but nothing may mint a new one.
    #[test]
    fn parse_kind_refuses_external() {
        assert!(parse_kind("external").is_err());
        assert!(parse_kind("app").is_ok());
        assert!(parse_kind("site").is_ok());
    }

    /// `Room`/`Document`/`Media`/`Feed` are declared in the schema so every
    /// reader stays total (an externally-tagged CBOR enum has no catch-all, so
    /// an unknown variant kills the decode of the WHOLE state), but they must
    /// not be MINTABLE while their per-kind Open semantics are undesigned: a
    /// published entry cannot be un-minted, only tombstoned.
    ///
    /// Mirrors `parse_kind_refuses_external`, but the message has to differ —
    /// these are reserved, not retired, and not unknown either. A curator told
    /// "unknown kind" would reasonably conclude they had typoed.
    #[test]
    fn parse_kind_refuses_the_reserved_kinds() {
        for k in ["room", "document", "media", "feed", "Room", "FEED"] {
            let err = parse_kind(k).expect_err("a reserved kind must not be mintable");
            assert!(
                err.to_string().to_lowercase().contains("reserved"),
                "{k}: message should say it is reserved, not unknown — got {err}"
            );
        }
        // A genuinely unknown kind still reads as unknown…
        let err = parse_kind("wombat").expect_err("must refuse");
        assert!(err.to_string().contains("unknown kind"), "{err}");
        // …and `external` keeps its own retired-not-reserved wording.
        let err = parse_kind("external").expect_err("must refuse");
        assert!(err.to_string().contains("retired"), "{err}");
        // …and the mintable ones still mint, so none of the above is vacuous.
        assert_eq!(parse_kind("app").unwrap(), Kind::new(Kind::APP));
        assert_eq!(parse_kind("site").unwrap(), Kind::new(Kind::SITE));
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
            kind: Kind::new(Kind::APP),
            title: "t".to_string(),
            snippet: String::new(),
            tags: vec![],
            locator: Locator::External {
                url: "https://example.com".to_string(),
            },
            featured: false,
            added_at: 0,
            // `None`, never `Some(<default>)`: this fixture stands in for the
            // records already on the network, and those carry no judgement.
            class: None,
            verified: None,
            ext: None,
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

    // --- update / classification ---

    /// A live entry at `version`, unclassified — the shape everything already on
    /// the network has.
    fn entry(version: u64) -> IndexEntry {
        IndexEntry {
            subject_id: SubjectId::random(),
            version,
            kind: Kind::new(Kind::SITE),
            title: "Original title".to_string(),
            snippet: "Original snippet".to_string(),
            tags: vec!["alpha".to_string(), "beta".to_string()],
            locator: parse_locator(&format!("freenet:{ID_A}/x")).unwrap(),
            featured: true,
            added_at: 1_700_000_000,
            class: None,
            verified: None,
            ext: None,
        }
    }

    fn a_class() -> Classification {
        Classification {
            landing: Audience::new(Audience::ADULT),
            has_adult_sections: true,
            volatility: Volatility::new(Volatility::FEED),
            classifier: 3,
            classified_at: 111,
        }
    }

    fn a_verification() -> Verification {
        Verification {
            last_verified_at: 222,
            status: VerifyStatus::new(VerifyStatus::LIVE),
        }
    }

    /// The point of `update`: one field changes and NOTHING else moves. Every
    /// field is asserted individually, because a carry-forward that drops a
    /// field is silent — the write succeeds, the entry just quietly loses its
    /// classification or its tags.
    #[test]
    fn update_carries_forward_every_unspecified_field() {
        let cur = IndexEntry {
            class: Some(a_class()),
            verified: Some(a_verification()),
            ext: Some([("k".to_string(), "v".to_string())].into_iter().collect()),
            ..entry(7)
        };
        let edits = EntryEdits {
            featured: Some(false),
            ..Default::default()
        };
        let next = next_entry(&cur, 7, &edits, 999).unwrap();

        assert_eq!(next.version, 8);
        assert!(!next.featured, "the one named field changed");
        assert_eq!(next.subject_id, cur.subject_id);
        assert_eq!(next.kind, cur.kind);
        assert_eq!(next.title, cur.title);
        assert_eq!(next.snippet, cur.snippet);
        assert_eq!(next.tags, cur.tags);
        assert_eq!(next.locator, cur.locator);
        assert_eq!(
            next.added_at, cur.added_at,
            "added_at is the SUBJECT's mint time, not this version's"
        );
        assert_eq!(next.class, cur.class);
        assert_eq!(next.verified, cur.verified);
        assert_eq!(next.ext, cur.ext);
    }

    #[test]
    fn update_refuses_a_wrong_cur_version() {
        let cur = entry(7);
        let edits = EntryEdits {
            title: Some("New title".to_string()),
            ..Default::default()
        };
        let err = next_entry(&cur, 6, &edits, 0).expect_err("must refuse to supersede blind");
        let msg = err.to_string();
        assert!(
            msg.contains('7') && msg.contains('6'),
            "should name the version the node reports AND the one asked for: {msg}"
        );
        // …and the right one is accepted, so this is not green for the wrong reason.
        assert_eq!(next_entry(&cur, 7, &edits, 0).unwrap().version, 8);
    }

    /// A CHANGED BODY MUST NEVER GO OUT AT AN UNCHANGED VERSION. `IndexSummary`
    /// carries per-subject versions only, and freenet-core skips fan-out outright
    /// when two peers' summary bytes are identical — before any delta is
    /// requested — so two different bodies at one version never exchange and
    /// diverge permanently.
    ///
    /// The mutation this pins is the plausible-looking optimisation "the values
    /// are the same, so skip the write / reuse the version". Restating a field
    /// with its current value still bumps.
    #[test]
    fn update_always_bumps_the_version_even_for_a_no_op_valued_edit() {
        let cur = entry(7);
        let edits = EntryEdits {
            title: Some(cur.title.clone()),
            featured: Some(cur.featured),
            ..Default::default()
        };
        let next = next_entry(&cur, 7, &edits, 0).unwrap();
        assert_eq!(
            next.version,
            cur.version + 1,
            "an unchanged-value edit must still supersede, or two peers could \
             hold different bodies at one version and never reconcile"
        );
        assert_eq!(next.title, cur.title);
    }

    /// Refused rather than bumped: with no flags at all there is nothing the
    /// curator could have meant, and re-signing an identical body would make
    /// every peer fetch it for nothing.
    #[test]
    fn update_with_no_edits_at_all_is_refused() {
        let err = next_entry(&entry(7), 7, &EntryEdits::default(), 0)
            .expect_err("an empty edit must be refused");
        assert!(err.to_string().contains("nothing to change"), "{err}");
    }

    #[test]
    fn update_refuses_a_version_at_the_maximum_instead_of_wrapping() {
        let edits = EntryEdits {
            featured: Some(false),
            ..Default::default()
        };
        let err = next_entry(&entry(u64::MAX), u64::MAX, &edits, 0).expect_err("must not wrap");
        assert!(err.to_string().contains("overflow"), "{err}");
    }

    /// `None` means NOT ASSESSED and must stay distinguishable from a real
    /// "general audience" judgement, so nothing may synthesise one. The refining
    /// flags cannot complete a classification that does not exist.
    #[test]
    fn a_classification_is_minted_only_when_landing_is_given() {
        // No classification flag at all: still unclassified.
        assert!(next_class(None, &ClassArgs::default(), 5)
            .unwrap()
            .is_none());

        // A refining flag alone must not invent a landing audience.
        let err = next_class(
            None,
            &ClassArgs {
                volatility: Some(VolatilityArg::Feed),
                ..Default::default()
            },
            5,
        )
        .expect_err("must refuse to complete a judgement nobody made");
        assert!(err.to_string().contains("--landing"), "{err}");

        // `--landing` is what mints one, with the documented defaults.
        let c = next_class(
            None,
            &ClassArgs {
                landing: Some(LandingArg::Adult),
                ..Default::default()
            },
            5,
        )
        .unwrap()
        .unwrap();
        assert_eq!(c.landing, Audience::new(Audience::ADULT));
        assert!(!c.has_adult_sections);
        assert_eq!(c.volatility, Volatility::new(Volatility::STATIC));
        assert_eq!(c.classifier, HAND_CLASSIFIER);
        assert_eq!(c.classified_at, 5);
    }

    /// A refining flag edits an EXISTING judgement in place, keeping what it did
    /// not name. `--landing general --adult-sections true` is the gated case the
    /// schema records as two observations rather than a `gated` flag.
    #[test]
    fn a_refining_flag_updates_an_existing_classification_in_place() {
        let cur = Some(Classification {
            landing: Audience::new(Audience::GENERAL),
            has_adult_sections: false,
            volatility: Volatility::new(Volatility::STATIC),
            classifier: 3,
            classified_at: 111,
        });
        let c = next_class(
            cur.as_ref(),
            &ClassArgs {
                adult_sections: Some(true),
                ..Default::default()
            },
            999,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            c.landing,
            Audience::new(Audience::GENERAL),
            "carried forward"
        );
        assert!(c.has_adult_sections);
        assert_eq!(c.classifier, 3, "carried forward");
        assert_eq!(
            c.classified_at, 999,
            "changing a classification IS a fresh assessment"
        );
    }

    /// …but an edit that touches no classification flag must not restate the
    /// judgement OR move its timestamp. `classified_at` is what a later taxonomy
    /// sweep uses to find stale labels, so a title fix that refreshed it would
    /// hide a label that has never been revisited.
    #[test]
    fn an_unrelated_edit_does_not_restate_or_re_time_a_judgement() {
        let cur = a_class();
        let out = next_class(Some(&cur), &ClassArgs::default(), 999)
            .unwrap()
            .expect("an existing judgement must survive an unrelated edit");
        assert_eq!(out, cur, "including classified_at, which must stay at 111");
    }

    /// `verified` asserts the CRAWLER confirmed this entry still describes the
    /// resource. Rewrite the description and that confirmation is about text the
    /// crawler never saw, so keeping `Live` would publish a confirmation nobody
    /// gave. Nothing is lost: `None` is where a freshly-added entry starts and
    /// the crawler re-verifies on its own cadence.
    #[test]
    fn a_description_edit_clears_the_crawler_verification_but_other_edits_do_not() {
        let cur = IndexEntry {
            verified: Some(a_verification()),
            ..entry(1)
        };
        let with = |edits: EntryEdits| next_entry(&cur, 1, &edits, 0).unwrap();

        assert!(
            with(EntryEdits {
                title: Some("Different".to_string()),
                ..Default::default()
            })
            .verified
            .is_none(),
            "a re-titled entry is not the entry that was verified"
        );
        assert!(with(EntryEdits {
            snippet: Some("Different".to_string()),
            ..Default::default()
        })
        .verified
        .is_none());

        // Edits that do not touch the description keep it…
        assert_eq!(
            with(EntryEdits {
                featured: Some(false),
                ..Default::default()
            })
            .verified,
            cur.verified
        );
        assert_eq!(
            with(EntryEdits {
                class: ClassArgs {
                    landing: Some(LandingArg::Adult),
                    ..Default::default()
                },
                ..Default::default()
            })
            .verified,
            cur.verified,
            "a classification is about the resource, not about the description"
        );
        // …and restating the SAME title is not a description change.
        assert_eq!(
            with(EntryEdits {
                title: Some(cur.title.clone()),
                ..Default::default()
            })
            .verified,
            cur.verified
        );
    }

    /// `--verified` is for the crawler's re-verification sweep: it stamps the
    /// exact status given, at `now`, regardless of what was there before.
    #[test]
    fn verified_flag_stamps_the_given_status_at_now() {
        let cur = IndexEntry {
            verified: None,
            ..entry(1)
        };
        let out = next_entry(
            &cur,
            1,
            &EntryEdits {
                verified: Some(VerifyStatusArg::Unreachable),
                ..Default::default()
            },
            555,
        )
        .unwrap();
        let v = out.verified.expect("verified must be set");
        assert_eq!(v.status, VerifyStatus::new(VerifyStatus::UNREACHABLE));
        assert_eq!(v.last_verified_at, 555);
    }

    /// `--verified` alongside a description correction (the crawler's own
    /// "content changed, re-describe, and stamp the re-check" case) must WIN
    /// over the ordinary "a description change clears verified" rule -- the
    /// flag being passed at all means the caller just did the verification the
    /// field records, and clearing it back to `None` in the same call would
    /// make the crawler's re-verification pass unable to ever mark anything
    /// `live` again.
    #[test]
    fn verified_flag_survives_a_simultaneous_description_change() {
        let cur = IndexEntry {
            verified: Some(a_verification()),
            title: "Old Title".to_string(),
            ..entry(1)
        };
        let out = next_entry(
            &cur,
            1,
            &EntryEdits {
                title: Some("New Title".to_string()),
                verified: Some(VerifyStatusArg::Live),
                ..Default::default()
            },
            999,
        )
        .unwrap();
        let v = out.verified.expect("verified must be set, not cleared");
        assert_eq!(v.status, VerifyStatus::new(VerifyStatus::LIVE));
        assert_eq!(v.last_verified_at, 999);
    }

    #[test]
    fn ext_merges_by_default_and_clears_only_on_request() {
        let cur: BTreeMap<String, String> = [
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "2".to_string()),
        ]
        .into_iter()
        .collect();
        let args = |ext: &[&str], clear: bool| ClassArgs {
            ext: ext.iter().map(|s| s.to_string()).collect(),
            ext_clear: clear,
            ..Default::default()
        };

        let merged = next_ext(Some(&cur), &args(&["b=9", "c=3"], false))
            .unwrap()
            .unwrap();
        assert_eq!(
            merged.get("a").map(String::as_str),
            Some("1"),
            "another writer's key must survive a curator edit"
        );
        assert_eq!(merged.get("b").map(String::as_str), Some("9"));
        assert_eq!(merged.get("c").map(String::as_str), Some("3"));

        let cleared = next_ext(Some(&cur), &args(&["c=3"], true))
            .unwrap()
            .unwrap();
        assert_eq!(cleared.keys().collect::<Vec<_>>(), vec!["c"]);

        // An empty result collapses to `None`, not `Some({})`: they mean the same
        // thing but serialize differently, and only `None` reproduces the
        // pre-`ext` byte layout every existing signature was minted over.
        assert!(next_ext(Some(&cur), &args(&[], true)).unwrap().is_none());
        assert!(next_ext(None, &ClassArgs::default()).unwrap().is_none());
    }

    #[test]
    fn ext_parsing_rejects_a_missing_separator_or_a_repeated_key() {
        let args = |ext: &[&str]| ClassArgs {
            ext: ext.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        assert!(next_ext(None, &args(&["novalue"])).is_err());

        let dup = next_ext(None, &args(&["k=1", "k=2"]))
            .expect_err("a repeated key is a typo; keeping the last would drop a value silently");
        assert!(dup.to_string().contains("more than once"), "{dup}");

        // A VALUE may contain `=` (base64 padding, a query string); only the key
        // may not, which is why this splits once rather than on every `=`.
        let ok = next_ext(None, &args(&["k=a=b=="])).unwrap().unwrap();
        assert_eq!(ok["k"], "a=b==");
    }

    /// The `ext` bounds have to bite on the CLI path, not merely exist in
    /// `common`. `signed_live_record` is the shared pre-flight `add` and `update`
    /// both go through, so exercising it covers both — and if either command ever
    /// stops calling it, the bound stops being checked before the node sees it.
    #[test]
    fn ext_bound_violations_are_rejected_before_signing() {
        let key = generate_key();
        let with_ext = |pairs: Vec<(String, String)>| {
            let e = IndexEntry {
                ext: Some(pairs.into_iter().collect()),
                ..entry(1)
            };
            signed_live_record(e, &key)
        };

        // Over the TOTAL byte budget — the bound that actually protects the
        // 50 MiB state cap, and the one a per-key limit alone would not catch.
        let err = with_ext(vec![(
            "k".to_string(),
            "v".repeat(atlas_common::MAX_EXT_TOTAL_BYTES),
        )])
        .expect_err("must refuse");
        assert!(err.to_string().contains("budget"), "{err}");

        // Too many keys, each individually tiny.
        let too_many: Vec<(String, String)> = (0..=atlas_common::MAX_EXT_KEYS)
            .map(|i| (format!("k{i}"), String::new()))
            .collect();
        assert!(with_ext(too_many).is_err());

        // An over-long key.
        assert!(with_ext(vec![(
            "k".repeat(atlas_common::MAX_EXT_KEY + 1),
            String::new()
        )])
        .is_err());

        // A control character — this is what keeps `atlasctl show` safe to print
        // the pairs raw.
        assert!(with_ext(vec![("k".to_string(), "a\u{1b}[31m".to_string())]).is_err());

        // …and a legal map goes through, so none of the above passes vacuously.
        assert!(with_ext(vec![("k".to_string(), "v".to_string())]).is_ok());
    }

    /// `IndexEntry` EXACTLY as it was before `class` / `verified` / `ext` were
    /// added, so a test can mint a record the way the deployed curator minted the
    /// live ones.
    ///
    /// Deliberately NOT expressed in terms of the current type: an "equivalence"
    /// built out of the code under test proves only that the code agrees with
    /// itself. This is a frozen copy of the OLD shape.
    #[derive(Serialize)]
    struct LegacyEntry {
        subject_id: SubjectId,
        version: u64,
        kind: Kind,
        title: String,
        snippet: String,
        tags: Vec<String>,
        locator: Locator,
        featured: bool,
        added_at: u64,
    }

    #[derive(Serialize)]
    enum LegacyBody {
        Live(LegacyEntry),
        #[allow(dead_code)]
        Tomb(Tombstone),
    }

    /// A CBOR index state holding ONE record whose signed bytes are in the
    /// pre-taxonomy shape, plus the params it verifies under. With
    /// `correctly_keyed = false` the record is filed under a different subject
    /// id — the failure `IndexState::verify` adds over `check_structure`, and
    /// therefore the one that proves `preflight_generation` can say no.
    fn legacy_generation(correctly_keyed: bool) -> (Vec<u8>, IndexParams) {
        let root = generate_key();
        let online = generate_key();
        let ka_body = KeyAuthBody {
            version: 1,
            authorized: vec![online.verifying_key()],
        };
        let key_auth = KeyAuth {
            sig: sign(&ka_body, &root),
            body: ka_body,
        };
        let sid = SubjectId::random();
        let legacy = LegacyBody::Live(LegacyEntry {
            subject_id: sid.clone(),
            version: 1,
            kind: Kind::new(Kind::SITE),
            title: "A site indexed before the taxonomy existed".to_string(),
            snippet: "Minted by the old curator.".to_string(),
            tags: vec!["freenet".to_string()],
            locator: parse_locator(&format!("freenet:{ID_A}/")).unwrap(),
            featured: false,
            added_at: 1_700_000_000,
        });
        // Sign the LEGACY bytes, exactly as the deployed curator did, then decode
        // them with the CURRENT type — which is what `migrate` does.
        let sig = sign(&legacy, &online);
        let bytes = atlas_common::canonical(&legacy);
        let body: RecordBody = ciborium::de::from_reader(&bytes[..])
            .expect("a legacy record must still decode under the current type");
        if let RecordBody::Live(e) = &body {
            assert!(
                e.class.is_none() && e.verified.is_none() && e.ext.is_none(),
                "a legacy record must decode as NOT ASSESSED, never as a default judgement"
            );
        }
        let mut st = IndexState::initialized(key_auth);
        let filed = if correctly_keyed {
            sid
        } else {
            SubjectId::random()
        };
        st.records.insert(
            filed,
            SignedRecord {
                body,
                by: online.verifying_key(),
                sig,
            },
        );
        let params = IndexParams {
            root_vk: root.verifying_key(),
            slug: "default".to_string(),
        };
        (encode(&st).unwrap(), params)
    }

    /// `preflight_generation` decodes LEGACY bytes with the CURRENT types and
    /// runs `IndexState::verify`, which re-serializes each body to check a
    /// signature minted against the old shape. The three fields added to
    /// `IndexEntry` are `skip_serializing_if`, so an absent one re-serializes
    /// byte-for-byte and the signature still verifies — but only while nothing
    /// re-introduces a field that serializes when absent.
    ///
    /// Migration is all-or-nothing: ONE failing record rejects the whole
    /// generation. So this test failing means the live entries become
    /// unmigratable, not that one of them is skipped.
    #[test]
    fn preflight_accepts_a_generation_minted_before_the_taxonomy_existed() {
        let (state, params) = legacy_generation(true);
        preflight_generation(&state, &params)
            .expect("a pre-taxonomy generation must still pass the contract's rules");
    }

    /// The control. Without it, the test above could be green because
    /// `preflight_generation` accepts everything.
    #[test]
    fn preflight_still_rejects_a_record_stored_under_the_wrong_subject_id() {
        let (state, params) = legacy_generation(false);
        let err = preflight_generation(&state, &params).expect_err("must reject");
        assert!(err.to_string().contains("wrong subject id"), "{err}");
    }

    /// The classification flags must spell identically on `add` and `update`.
    /// They are one `#[command(flatten)]` struct precisely so they cannot drift;
    /// this fails if someone re-declares them on one command by hand.
    #[test]
    fn add_and_update_share_the_classification_flag_spelling() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        let flags = |name: &str| -> Vec<String> {
            cmd.get_subcommands()
                .find(|c| c.get_name() == name)
                .unwrap_or_else(|| panic!("{name} subcommand missing"))
                .get_arguments()
                .filter_map(|a| a.get_long().map(str::to_string))
                .collect()
        };
        let add = flags("add");
        let update = flags("update");
        for f in [
            "landing",
            "adult-sections",
            "volatility",
            "classifier",
            "ext",
            "ext-clear",
        ] {
            assert!(add.contains(&f.to_string()), "`add` is missing --{f}");
            assert!(update.contains(&f.to_string()), "`update` is missing --{f}");
        }
    }

    /// Pin the accepted VALUES, not just the flag names: the CLI enums are local
    /// mirrors of the schema ones (see `LandingArg`), so nothing else would catch
    /// a rename of a variant here.
    #[test]
    fn classification_flag_values_parse() {
        let parsed = Cli::try_parse_from([
            "atlasctl",
            "update",
            "--subject",
            "2NEpo7TZRhna7vSvL",
            "--cur-version",
            "1",
            "--landing",
            "adult",
            "--volatility",
            "feed",
            "--adult-sections",
            "true",
            "--classifier",
            "7",
            "--ext",
            "k=v",
        ])
        .expect("the documented spelling must parse");
        match parsed.cmd {
            Cmd::Update {
                cur_version, edits, ..
            } => {
                assert_eq!(cur_version, 1);
                assert_eq!(edits.class.landing, Some(LandingArg::Adult));
                assert_eq!(edits.class.volatility, Some(VolatilityArg::Feed));
                assert_eq!(edits.class.adult_sections, Some(true));
                assert_eq!(edits.class.classifier, Some(7));
                assert_eq!(edits.class.ext, vec!["k=v".to_string()]);
            }
            _ => panic!("parsed as the wrong subcommand"),
        }
        // An unknown audience is rejected by clap rather than silently defaulted.
        assert!(Cli::try_parse_from([
            "atlasctl",
            "update",
            "--subject",
            "2NEpo7TZRhna7vSvL",
            "--cur-version",
            "1",
            "--landing",
            "maybe",
        ])
        .is_err());
    }

    /// `show` must render UNCLASSIFIED distinguishably from
    /// classified-as-general. Rendering an absent judgement as "general" (or as
    /// nothing at all) is the same mistake as storing `Some(General)` for it:
    /// the curator loses the ability to see what still needs looking at.
    #[test]
    fn show_renders_unclassified_distinguishably_from_classified_general() {
        let unclassified = entry_metadata_lines(&entry(1));
        assert!(
            unclassified.contains("NOT CLASSIFIED"),
            "got {unclassified:?}"
        );

        let general = entry_metadata_lines(&IndexEntry {
            class: Some(Classification {
                landing: Audience::new(Audience::GENERAL),
                has_adult_sections: false,
                volatility: Volatility::new(Volatility::STATIC),
                classifier: 2,
                classified_at: 555,
            }),
            ..entry(1)
        });
        assert!(general.contains("general landing"), "got {general:?}");
        assert!(general.contains("no adult sections"), "got {general:?}");
        assert!(!general.contains("NOT CLASSIFIED"));

        // `verified` and `ext` get a line only when present: their absence is not
        // a call to action, unlike an unmade judgement.
        assert!(!unclassified.contains("verified") && !unclassified.contains("ext"));
        let full = entry_metadata_lines(&IndexEntry {
            class: Some(a_class()),
            verified: Some(a_verification()),
            ext: Some(
                [("src".to_string(), "crawler".to_string())]
                    .into_iter()
                    .collect(),
            ),
            ..entry(1)
        });
        assert!(full.contains("adult landing") && full.contains("has adult sections"));
        assert!(full.contains("verified: live at 222"), "got {full:?}");
        assert!(full.contains("ext: src=crawler"), "got {full:?}");
    }

    /// `--landing` on `add` mints the classification with the entry; omitting it
    /// leaves the entry NOT ASSESSED. This is the `add` half of
    /// `a_classification_is_minted_only_when_landing_is_given`.
    #[test]
    fn add_mints_a_classification_only_when_asked() {
        assert!(next_class(None, &ClassArgs::default(), 42)
            .unwrap()
            .is_none());
        let c = next_class(
            None,
            &ClassArgs {
                landing: Some(LandingArg::General),
                adult_sections: Some(true),
                volatility: Some(VolatilityArg::Feed),
                classifier: Some(9),
                ..Default::default()
            },
            42,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            (c.landing, c.has_adult_sections),
            (Audience::new(Audience::GENERAL), true),
            "general landing with adult sections is the GATED case, not a contradiction"
        );
        assert_eq!(c.volatility, Volatility::new(Volatility::FEED));
        assert_eq!(c.classifier, 9);
        assert_eq!(c.classified_at, 42);
    }
}
