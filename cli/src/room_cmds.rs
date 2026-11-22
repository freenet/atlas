//! `atlasctl room-add`, `room-refresh` and `rooms`: curator-side indexing of
//! public River rooms. The decisions live in the `atlas-rooms` crate; this file
//! is the node I/O and the command wiring.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use atlas_common::{IndexState, RecordBody, SignedRecord};
use atlas_rooms::contract::room_instance_id;
use atlas_rooms::eligibility::{
    evaluate, owner_b58, owner_of_locator, parse_owner, Decision, Exclusion,
};
use atlas_rooms::entry::{Characterization, TextSource};
use atlas_rooms::reconcile::{plan, Action, MintOptions};
use atlas_rooms::search::{search, RoomFilter, RoomHit};
use clap::ValueEnum;
use ed25519_dalek::VerifyingKey;
use freenet_migrate::pointer::{
    resolve_app_pointer, PointerFetch, PointerFloor, PointerIo, PointerOutcome,
};
use freenet_stdlib::prelude::ContractInstanceId;

use crate::api::NodeClient;
use crate::{
    fetch_state, load_key, next_class, next_ext, now_secs, send_delta, signed_live_record,
    split_tags, ClassArgs, Cli, IndexRead,
};

/// The `app_id` River publishes the room-contract pointer under.
pub const ROOM_CONTRACT_APP_ID: &[u8] = b"river.room-contract";

/// River's author verifying key, spelled as in River's `FREENET.md`. The whole
/// trust anchor for the pointer: a record is accepted only if it verifies
/// against this key. Pinned by `author_key_derives_the_documented_pointer`.
pub const RIVER_AUTHOR_VK: &str = "9Ebskq4y7NvJpTQTrF1FAxU8g6bR4Rhe4TRikXba55EJ";

/// The pointer's documented fixed address, used only by a test.
#[cfg(test)]
const ROOM_CONTRACT_POINTER_KEY: &str = "Ai4VLoC2jGdhpcB2UU8VPo3efUoxjm1Ju9VKXqRC63Az";

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharSource {
    Curator,
    Llm,
}

impl From<CharSource> for TextSource {
    fn from(c: CharSource) -> Self {
        match c {
            CharSource::Curator => TextSource::Curator,
            CharSource::Llm => TextSource::Llm,
        }
    }
}

fn river_author_vk() -> Result<VerifyingKey> {
    parse_owner(RIVER_AUTHOR_VK).map_err(|e| anyhow!("RIVER_AUTHOR_VK: {e}"))
}

/// [`PointerIo`] over `atlasctl`'s node connection. Every transport condition is
/// `Unreachable`, never `Absent`: a `NotFound` on the live network is mostly a
/// routing dead-end for a contract that exists, and `Absent` is the one answer
/// that licenses falling back to a baked-in key. We have no baked-in key, so
/// there is nothing to unlock, and reporting it would only be a downgrade
/// primitive.
struct NodePointerIo<'a> {
    client: &'a mut NodeClient,
}

impl PointerIo for NodePointerIo<'_> {
    type Error = std::convert::Infallible;

    async fn get_pointer(&mut self, id: ContractInstanceId) -> Result<PointerFetch, Self::Error> {
        Ok(match self.client.get_instance_optional(id, false).await {
            Ok(Some(bytes)) => PointerFetch::State(bytes),
            Ok(None) | Err(_) => PointerFetch::Unreachable,
        })
    }
}

/// The room-contract code hash River's signed pointer currently names.
///
/// Fails, rather than falling back to anything, on every outcome that does not
/// carry a verified record: indexing against a guessed room contract would read
/// the wrong address and then mark every listed room unreachable.
async fn current_room_code_hash(client: &mut NodeClient) -> Result<[u8; 32]> {
    let author = river_author_vk()?;
    let mut io = NodePointerIo { client };
    let outcome = resolve_app_pointer(
        &mut io,
        &author,
        ROOM_CONTRACT_APP_ID,
        // No floor is persisted between runs, so this resolution has no
        // anti-rollback protection against a replayed older record. See the
        // open question on the branch report.
        PointerFloor::never_resolved(),
    )
    .await
    .map_err(|e| anyhow!("resolving River's room-contract pointer: {e:?}"))?;
    match outcome.resolved() {
        Some(r) => Ok(r.code_hash()),
        None => bail!(
            "River's room-contract pointer did not resolve to a verified record ({}); \
             refusing to guess a room contract. Retry, or pass --state-file with a \
             room state you fetched yourself.",
            outcome_name(&outcome)
        ),
    }
}

fn outcome_name(o: &PointerOutcome) -> &'static str {
    match o {
        PointerOutcome::Resolved(_) => "resolved",
        PointerOutcome::Unchanged(_) => "unchanged",
        PointerOutcome::CompetingRecord => "competing record",
        PointerOutcome::Unavailable => "unavailable",
        PointerOutcome::NeverPublished => "never published",
        PointerOutcome::Withdrawn => "withdrawn",
        _ => "no usable record",
    }
}

/// What reading a room told us. `None` is "could not read it", which is NOT
/// evidence the room is gone.
async fn read_room_state(
    cli: &Cli,
    owner: &VerifyingKey,
    state_file: Option<&Path>,
) -> Result<Option<Vec<u8>>> {
    if let Some(p) = state_file {
        return Ok(Some(
            std::fs::read(p).with_context(|| format!("reading {}", p.display()))?,
        ));
    }
    let mut client = NodeClient::connect(&cli.node).await?;
    let code_hash = current_room_code_hash(&mut client).await?;
    let id = ContractInstanceId::new(room_instance_id(owner, &code_hash));
    client.get_instance_optional(id, false).await
}

fn decision_for(owner: &VerifyingKey, state: Option<Vec<u8>>) -> Decision {
    match state {
        // The room is not a secret from us, so a missing state is only an
        // unverifiable read. `submitted` is true: this command IS the explicit
        // curator submission, and a refresh only ever runs on rooms already
        // accepted into the index.
        Some(bytes) => evaluate(owner, &bytes, true),
        None => Decision::Excluded(
            Exclusion::Unverifiable("the node could not find the room contract".into()),
            None,
        ),
    }
}

async fn require_index(cli: &Cli, dir: &Path) -> Result<IndexState> {
    // `fresh = true`: this is a write path, and a stale read is how a refresh
    // would republish or tombstone against a copy that lags the network.
    match fetch_state(cli, dir, true).await? {
        IndexRead::Present(s) => Ok(*s),
        IndexRead::Empty => bail!("the index exists but was never initialized"),
        IndexRead::Unfindable => bail!(
            "the node could not find the index; refusing to plan against an unknown state \
             (a duplicate check against nothing would mint a second listing)"
        ),
    }
}

fn sign_actions(dir: &Path, p: atlas_rooms::reconcile::Plan) -> Result<Vec<SignedRecord>> {
    let online = load_key(&dir.join("online.key"))?;
    p.actions
        .into_iter()
        .map(|a| match a {
            Action::Add(e) | Action::Update(e) => signed_live_record(*e, &online),
            Action::Tombstone(t) => {
                let body = RecordBody::Tomb(t);
                body.check_structure()
                    .map_err(|e| anyhow!("tombstone would be rejected by the contract: {e}"))?;
                Ok(SignedRecord {
                    sig: atlas_common::sign(&body, &online),
                    by: online.verifying_key(),
                    body,
                })
            }
        })
        .collect()
}

async fn sync_one(
    cli: &Cli,
    dir: &Path,
    state: &IndexState,
    owner: &VerifyingKey,
    room_state: Option<Vec<u8>>,
    ch: Option<&Characterization>,
    mint: &MintOptions,
    now: u64,
) -> Result<(Vec<SignedRecord>, Vec<String>)> {
    let _ = cli;
    let decision = decision_for(owner, room_state);
    let p = plan(state, owner, &decision, ch, mint, now).map_err(|e| anyhow!("{e}"))?;
    let notes = p.notes.clone();
    Ok((sign_actions(dir, p)?, notes))
}

#[allow(clippy::too_many_arguments)]
pub async fn room_add(
    cli: &Cli,
    dir: &Path,
    owner: &str,
    state_file: Option<&PathBuf>,
    characterization: Option<&str>,
    source: CharSource,
    tags: &str,
    featured: bool,
    dry_run: bool,
    class: &ClassArgs,
) -> Result<()> {
    let owner = parse_owner(owner).map_err(|e| anyhow!("{e}"))?;
    let now = now_secs();
    let ch = characterization
        .map(|t| Characterization::new(t, source.into()))
        .transpose()
        .map_err(|e| anyhow!("{e}"))?;
    let mint = MintOptions {
        tags: split_tags(tags),
        featured,
        class: next_class(None, class, now)?,
    };
    // Validated here so a bad `--ext` fails before any network round trip; the
    // keys themselves are merged into the entry by `refresh`-style updates.
    let _ = next_ext(None, class)?;
    let index = require_index(cli, dir).await?;
    let room = read_room_state(cli, &owner, state_file.map(|p| p.as_path())).await?;
    let (records, notes) =
        sync_one(cli, dir, &index, &owner, room, ch.as_ref(), &mint, now).await?;
    for n in &notes {
        println!("{n}");
    }
    if records.is_empty() {
        if notes.iter().any(|n| n.starts_with("not indexed")) {
            bail!("room {} was not added", owner_b58(&owner));
        }
        return Ok(());
    }
    if dry_run {
        println!("dry run: {} record(s) not sent", records.len());
        return Ok(());
    }
    let n = records.len();
    send_delta(cli, dir, records).await?;
    println!("published {n} record(s) for room {}", owner_b58(&owner));
    Ok(())
}

pub async fn room_refresh(
    cli: &Cli,
    dir: &Path,
    owner: Option<&str>,
    state_file: Option<&PathBuf>,
    dry_run: bool,
) -> Result<()> {
    let now = now_secs();
    let index = require_index(cli, dir).await?;
    let owners: Vec<VerifyingKey> = match owner {
        Some(o) => vec![parse_owner(o).map_err(|e| anyhow!("{e}"))?],
        None => {
            if state_file.is_some() {
                bail!("--state-file needs a single owner key");
            }
            let mut v: Vec<VerifyingKey> = index
                .live_entries()
                .filter_map(|e| owner_of_locator(&e.locator))
                .collect();
            v.sort_by_key(|k| k.to_bytes());
            v.dedup();
            v
        }
    };
    let mut all: Vec<SignedRecord> = Vec::new();
    let mut failures = 0usize;
    for o in &owners {
        // One unreadable room must not stop the sweep.
        let room = match read_room_state(cli, o, state_file.map(|p| p.as_path())).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{}: read failed: {e:#}", owner_b58(o));
                failures += 1;
                None
            }
        };
        let (recs, notes) = sync_one(
            cli,
            dir,
            &index,
            o,
            room,
            None,
            &MintOptions::default(),
            now,
        )
        .await?;
        for n in &notes {
            println!("{}: {n}", owner_b58(o));
        }
        all.extend(recs);
    }
    if all.is_empty() {
        println!("{} room(s) checked, nothing to publish", owners.len());
    } else if dry_run {
        println!("dry run: {} record(s) not sent", all.len());
    } else {
        let n = all.len();
        send_delta(cli, dir, all).await?;
        println!("published {n} record(s) across {} room(s)", owners.len());
    }
    if failures > 0 {
        bail!("{failures} room read(s) failed (the index was only marked unreachable, never delisted)");
    }
    Ok(())
}

fn hit_json(h: &RoomHit) -> serde_json::Value {
    serde_json::json!({
        "owner_key": h.owner_key,
        "subject_id": h.subject_id,
        "title": h.title,
        "snippet": h.snippet,
        "snippet_source": match h.snippet_source {
            TextSource::Owner => "owner",
            TextSource::Curator => "curator",
            TextSource::Llm => "llm",
        },
        "tags": h.tags,
        "featured": h.featured,
        "added_at": h.added_at,
        "availability": format!("{:?}", h.availability).to_lowercase(),
        "last_verified_at": h.last_verified_at,
        "config_version": h.config_version,
    })
}

pub async fn rooms(
    cli: &Cli,
    dir: &Path,
    query: Option<&str>,
    tags: &[String],
    all: bool,
    only_live: bool,
    json: bool,
) -> Result<()> {
    let index = match fetch_state(cli, dir, false).await? {
        IndexRead::Present(s) => *s,
        IndexRead::Empty | IndexRead::Unfindable => bail!("the node could not read the index"),
    };
    let hits = search(
        &index,
        &RoomFilter {
            query: query.map(str::to_string),
            tags: tags.to_vec(),
            safe_search: !all,
            only_live,
            stale_after_secs: Some(atlas_rooms::entry::REVERIFY_AFTER_SECS * 2),
            now: now_secs(),
        },
    );
    if json {
        let v: Vec<_> = hits.iter().map(hit_json).collect();
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        for h in &hits {
            println!(
                "{}  {}  [{} / {:?}]\n    {}",
                h.owner_key,
                h.title,
                h.snippet_source_label(),
                h.availability,
                h.snippet
            );
        }
        println!("{} room(s)", hits.len());
    }
    Ok(())
}

trait SourceLabel {
    fn snippet_source_label(&self) -> &'static str;
}

impl SourceLabel for RoomHit {
    fn snippet_source_label(&self) -> &'static str {
        match self.snippet_source {
            TextSource::Owner => "owner text",
            TextSource::Curator => "curator text",
            TextSource::Llm => "llm text",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use freenet_migrate::pointer::PointerResolver;
    use freenet_migrate::Step;

    /// The trust anchor and the app id must derive the address River documents.
    /// A typo in either would resolve somebody else's pointer, or nothing.
    #[test]
    fn author_key_derives_the_documented_pointer() {
        let author = river_author_vk().unwrap();
        let mut r = PointerResolver::new(
            &author,
            ROOM_CONTRACT_APP_ID,
            PointerFloor::never_resolved(),
        )
        .unwrap();
        match r.next_action() {
            Step::Get(id) => assert_eq!(id.to_string(), ROOM_CONTRACT_POINTER_KEY),
            other => panic!("expected a GET, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_room_is_unverifiable_not_gone() {
        let sk = ed25519_dalek::SigningKey::from_bytes(&[3u8; 32]);
        let d = decision_for(&sk.verifying_key(), None);
        match d {
            Decision::Excluded(e, _) => assert!(!e.delists()),
            other => panic!("{other:?}"),
        }
    }
}
