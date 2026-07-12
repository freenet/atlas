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
    generate_key, sign, IndexDelta, IndexEntry, IndexParams, IndexState, KeyAuth, KeyAuthBody,
    Kind, Locator, RecordBody, SignedRecord, SubjectId, Tombstone,
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
        /// `freenet:<full-id><path>` or `https://...`
        #[arg(long)]
        locator: String,
        #[arg(long)]
        featured: bool,
    },
    /// Tombstone a subject by id (needs the current version to supersede it).
    Remove {
        #[arg(long)]
        subject: String,
        #[arg(long)]
        cur_version: u64,
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
    /// Carry the curated index forward after a rebuild re-keyed it: GET the
    /// state from the newest legacy address that still holds entries and PUT it
    /// into the current address (subscribing so the node hosts it). Idempotent —
    /// the contract merges rather than overwrites, so re-running is safe.
    Migrate {
        /// Probe/GET only; report what WOULD be carried forward without PUTting.
        #[arg(long)]
        dry_run: bool,
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
        } => add(&cli, &dir, kind, title, snippet, tags, locator, *featured).await,
        Cmd::Remove {
            subject,
            cur_version,
        } => remove(&cli, &dir, subject, *cur_version).await,
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
        Cmd::Migrate { dry_run } => migrate(&cli, &dir, *dry_run).await,
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
/// changes the WASM moves it. This GETs the state from the newest legacy
/// (pre-rebuild) address that still holds entries and PUTs it into the current
/// address, so the entries are not stranded. The contract merges on update, so
/// running this twice is safe (the second run is a no-op merge).
async fn migrate(cli: &Cli, dir: &Path, dry_run: bool) -> Result<()> {
    let params = params_bytes(dir, &cli.slug)?;
    let current_key = NodeClient::contract_key(CONTRACT_WASM, &params);
    let legacy_keys = migration::legacy_index_keys(&params);
    if legacy_keys.is_empty() {
        bail!("no legacy code hashes registered — nothing to migrate from");
    }
    println!("current index: {}", current_key.id());

    let mut client = NodeClient::connect(&cli.node).await?;
    let current_state = client.get(&current_key, false).await.unwrap_or_default();
    let current_live = count_live(&current_state);
    println!("current index holds {current_live} live entries");

    // Probe legacy addresses newest-first; pick the one holding the most live
    // entries as the source of truth.
    let mut best: Option<(usize, Vec<u8>, ContractInstanceId)> = None;
    for (i, key) in legacy_keys.iter().enumerate() {
        let state = client.get(key, false).await.unwrap_or_default();
        let live = count_live(&state);
        println!("legacy[{i}] {} holds {live} live entries", key.id());
        if live > best.as_ref().map(|(n, ..)| *n).unwrap_or(0) {
            best = Some((live, state, *key.id()));
        }
    }

    let Some((live, state, from_id)) = best else {
        println!("no legacy address holds any entries — nothing to carry forward");
        return Ok(());
    };
    if live <= current_live {
        println!(
            "current index already holds >= the {live} legacy entries (from {from_id}); \
             nothing to carry forward"
        );
        return Ok(());
    }

    if dry_run {
        println!(
            "[dry-run] would carry {live} entries from {from_id} into {}",
            current_key.id()
        );
        return Ok(());
    }

    // PUT-over the (fresh) current contract; the node applies it as a merging
    // update and, with subscribe, hosts it so cross-node GETs can find it.
    client
        .put(CONTRACT_WASM, params, state, true)
        .await
        .context("PUT carried state into the current index")?;
    let after = client.get(&current_key, false).await.unwrap_or_default();
    println!(
        "migrated {live} entries from {from_id} into {}; it now holds {} live entries",
        current_key.id(),
        count_live(&after)
    );
    Ok(())
}

/// Decode an index state and count live (non-tombstoned) entries; 0 if empty or
/// undecodable (best-effort, for logging only).
fn count_live(state: &[u8]) -> usize {
    if state.is_empty() {
        return 0;
    }
    match ciborium::de::from_reader::<IndexState, &[u8]>(state) {
        Ok(st) => st.live_entries().count(),
        Err(_) => 0,
    }
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
) -> Result<()> {
    let online = load_key(&dir.join("online.key"))?;
    let entry = IndexEntry {
        subject_id: SubjectId::random(),
        version: 1,
        kind: parse_kind(kind)?,
        title: title.to_string(),
        snippet: snippet.to_string(),
        tags: split_tags(tags),
        locator: parse_locator(locator)?,
        featured,
        added_at: now_secs(),
    };
    let subject = entry.subject_id.as_str().to_string();
    let body = RecordBody::Live(entry);
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
    for e in entries {
        let star = if e.featured { "★ " } else { "  " };
        println!(
            "{star}{}  [{:?}]  {}\n     {}\n     {}",
            e.subject_id.as_str(),
            e.kind,
            e.title,
            e.snippet,
            e.locator.to_uri()
        );
    }
    Ok(())
}

async fn send_delta(cli: &Cli, dir: &Path, records: Vec<SignedRecord>) -> Result<()> {
    let delta = encode(&IndexDelta {
        key_auth: None,
        records,
    })?;
    let params = params_bytes(dir, &cli.slug)?;
    let key = NodeClient::contract_key(CONTRACT_WASM, &params);
    let mut client = NodeClient::connect(&cli.node).await?;
    client.update_delta(key, delta).await
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
        bail!("locator must start with `freenet:` or `https://`")
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
