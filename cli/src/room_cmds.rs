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
use atlas_rooms::search::{find_room, search, RoomFilter, RoomHit};
use clap::ValueEnum;
use ed25519_dalek::VerifyingKey;
use freenet_migrate::pointer::{
    resolve_app_pointer, PointerFetch, PointerFloor, PointerIo, PointerOutcome,
};
use freenet_stdlib::prelude::ContractInstanceId;
use serde::{Deserialize, Serialize};

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

/// File in the key directory holding the anti-rollback floor for River's
/// room-contract pointer.
const FLOOR_FILE: &str = "river_room_pointer_floor.json";

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

// ---------------------------------------------------------------------------
// Pointer floor persistence
// ---------------------------------------------------------------------------

/// On-disk form of a [`PointerFloor`]. `withdrawn` is its own column rather than
/// an inference from a zeroed hash, so one bad row cannot read as a withdrawal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct StoredFloor {
    version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    code_hash: Option<String>,
    #[serde(default)]
    withdrawn: bool,
}

impl StoredFloor {
    fn from_floor(f: &PointerFloor) -> Self {
        if f.is_withdrawn() {
            return Self {
                version: f.version(),
                code_hash: None,
                withdrawn: true,
            };
        }
        Self {
            version: f.version(),
            code_hash: f.code_hash().map(|h| bs58::encode(h).into_string()),
            withdrawn: false,
        }
    }

    /// Every failure is an error, never `never_resolved()`: that is the state in
    /// which a replayed older pointer would be accepted again.
    fn to_floor(&self) -> Result<PointerFloor> {
        if self.withdrawn {
            return PointerFloor::withdrawn_at(self.version)
                .map_err(|e| anyhow!("stored floor records an unusable withdrawal: {e}"));
        }
        let enc = self
            .code_hash
            .as_deref()
            .ok_or_else(|| anyhow!("stored floor has no code hash and is not a withdrawal"))?;
        let bytes = bs58::decode(enc)
            .into_vec()
            .map_err(|e| anyhow!("stored floor code hash is not base58: {e}"))?;
        let hash: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("stored floor code hash is not 32 bytes"))?;
        PointerFloor::at(self.version, hash).map_err(|e| anyhow!("stored floor unusable: {e}"))
    }
}

fn load_floor(dir: &Path) -> Result<PointerFloor> {
    let path = dir.join(FLOOR_FILE);
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str::<StoredFloor>(&text)
            .map_err(|e| anyhow!("{}: {e}", path.display()))
            .and_then(|s| s.to_floor())
            .with_context(|| {
                format!(
                    "the pointer floor is unreadable. atlasctl will not treat that as \"never \
                     resolved\", because that would accept a replayed older pointer. Inspect or \
                     delete {} to continue",
                    path.display()
                )
            }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(PointerFloor::never_resolved()),
        Err(e) => Err(anyhow!("reading {}: {e}", path.display())),
    }
}

fn save_floor(dir: &Path, floor: &PointerFloor) -> Result<()> {
    let path = dir.join(FLOOR_FILE);
    let tmp = dir.join(format!("{FLOOR_FILE}.tmp"));
    std::fs::write(
        &tmp,
        serde_json::to_string_pretty(&StoredFloor::from_floor(floor))?,
    )?;
    std::fs::rename(&tmp, &path).with_context(|| format!("writing {}", path.display()))
}

// ---------------------------------------------------------------------------
// Reading rooms
// ---------------------------------------------------------------------------

/// [`PointerIo`] over `atlasctl`'s node connection. Every transport condition is
/// `Unreachable`, never `Absent`: a `NotFound` on the live network is mostly a
/// routing dead-end for a contract that exists, and `Absent` is the one answer
/// that licenses falling back to a baked-in key. We have none, so reporting it
/// would only be a downgrade primitive.
struct NodePointerIo<'a> {
    client: &'a mut NodeClient,
}

impl PointerIo for NodePointerIo<'_> {
    type Error = std::convert::Infallible;

    async fn get_pointer(&mut self, id: ContractInstanceId) -> Result<PointerFetch, Self::Error> {
        // A fresh read (`subscribe = true`): the pointer is what stops an older
        // room-contract generation being trusted, so a cached copy is not good
        // enough.
        Ok(match self.client.get_instance_optional(id, true).await {
            Ok(Some(bytes)) => PointerFetch::State(bytes),
            Ok(None) | Err(_) => PointerFetch::Unreachable,
        })
    }
}

fn outcome_name(o: &PointerOutcome) -> &'static str {
    match o {
        PointerOutcome::Resolved(_) => "resolved",
        PointerOutcome::Unchanged(_) => "unchanged",
        PointerOutcome::CompetingRecord { .. } => "competing record",
        PointerOutcome::Unavailable => "unavailable",
        PointerOutcome::NeverPublished => "never published",
        PointerOutcome::Withdrawn { .. } => "withdrawn",
        _ => "no usable record",
    }
}

/// Reads room states from a node. Built ONCE per command: the pointer is
/// resolved a single time, so a failure aborts the whole command before any
/// record is signed instead of turning into one failed read per room.
struct RoomReader {
    client: NodeClient,
    code_hash: [u8; 32],
}

impl RoomReader {
    /// Resolve River's room-contract pointer against the persisted floor and
    /// save the advanced floor. Fails, rather than falling back to anything, on
    /// every outcome without a verified record: indexing against a guessed room
    /// contract would read the wrong address and then mark every room
    /// unreachable.
    async fn connect(cli: &Cli, dir: &Path) -> Result<Self> {
        let mut client = NodeClient::connect(&cli.node).await?;
        let author = river_author_vk()?;
        let floor = load_floor(dir)?;
        let mut io = NodePointerIo {
            client: &mut client,
        };
        let outcome = resolve_app_pointer(&mut io, &author, ROOM_CONTRACT_APP_ID, floor)
            .await
            .map_err(|e| anyhow!("resolving River's room-contract pointer: {e:?}"))?;
        let Some(resolved) = outcome.resolved() else {
            bail!(
                "River's room-contract pointer did not resolve to a verified record ({}); \
                 refusing to guess a room contract. Retry, or pass --state-file with a room \
                 state you fetched yourself.",
                outcome_name(&outcome)
            );
        };
        let code_hash = resolved.code_hash();
        if let Some(next) = outcome.next_floor() {
            save_floor(dir, &next)?;
        }
        Ok(Self { client, code_hash })
    }

    /// A FRESH read (`subscribe = true`) of the room: the node's cached copy can
    /// predate the owner making the room private, and a first `Add` has no stored
    /// configuration version to detect that. `None` is "could not read it", NOT
    /// evidence the room is gone.
    async fn read(&mut self, owner: &VerifyingKey) -> Result<Option<Vec<u8>>> {
        let id = ContractInstanceId::new(room_instance_id(owner, &self.code_hash));
        self.client.get_instance_optional(id, true).await
    }
}

fn read_state_file(p: &Path) -> Result<Vec<u8>> {
    std::fs::read(p).with_context(|| format!("reading {}", p.display()))
}

fn decision_for(owner: &VerifyingKey, state: Option<Vec<u8>>) -> Decision {
    match state {
        // `submitted` is true: `room-add` IS the explicit curator submission,
        // and a refresh only runs on rooms already accepted into the index.
        // This means a room with no `public_admission` policy is eligible, so an
        // owner opt-out is honoured only if this reader understands the field
        // (see FREENET.md).
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

#[allow(clippy::too_many_arguments)]
fn sync_one(
    dir: &Path,
    state: &IndexState,
    owner: &VerifyingKey,
    room_state: Option<Vec<u8>>,
    ch: Option<&Characterization>,
    mint: &MintOptions,
    now: u64,
) -> Result<(Vec<SignedRecord>, Vec<String>)> {
    let decision = decision_for(owner, room_state);
    let p = plan(state, owner, &decision, ch, mint, now).map_err(|e| anyhow!("{e}"))?;
    let notes = p.notes.clone();
    Ok((sign_actions(dir, p)?, notes))
}

/// Flags that only shape a NEW entry. On a room that is already listed they would
/// be silently dropped, so they are refused: `atlasctl update` edits an existing
/// entry.
fn refuse_new_entry_flags_on_existing(
    existing: Option<&RoomHit>,
    tags: &str,
    featured: bool,
    class: &ClassArgs,
) -> Result<()> {
    if let Some(h) = existing {
        if !split_tags(tags).is_empty() || featured || !class.is_empty() {
            bail!(
                "room {} is already indexed as subject {}; --tags, --featured and the \
                 classification/--ext flags only apply when a room is first added. Use \
                 `atlasctl update --subject {} --cur-version <n> ...` to change them.",
                h.owner_key,
                h.subject_id,
                h.subject_id
            );
        }
    }
    Ok(())
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
        // Parsed here so a malformed `--ext` fails before any round trip.
        ext: next_ext(None, class)?,
    };
    let index = require_index(cli, dir).await?;
    let existing = find_room(
        &index,
        &owner_b58(&owner),
        &RoomFilter {
            safe_search: false,
            ..Default::default()
        },
    );
    refuse_new_entry_flags_on_existing(existing.as_ref(), tags, featured, class)?;
    let room = match state_file {
        Some(p) => Some(read_state_file(p)?),
        None => RoomReader::connect(cli, dir).await?.read(&owner).await?,
    };
    let (records, notes) = sync_one(dir, &index, &owner, room, ch.as_ref(), &mint, now)?;
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
    // Resolve the pointer ONCE, before anything is planned. If it cannot be
    // resolved the whole sweep aborts having signed nothing.
    let mut reader = match state_file {
        Some(_) => None,
        None => Some(RoomReader::connect(cli, dir).await?),
    };
    let mut all: Vec<SignedRecord> = Vec::new();
    let mut failures = 0usize;
    for o in &owners {
        let room = match (&mut reader, state_file) {
            (_, Some(p)) => Some(read_state_file(p)?),
            (Some(r), None) => match r.read(o).await {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("{}: read failed: {e:#}", owner_b58(o));
                    failures += 1;
                    None
                }
            },
            (None, None) => unreachable!("a reader exists whenever there is no state file"),
        };
        let (recs, notes) = sync_one(dir, &index, o, room, None, &MintOptions::default(), now)?;
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
        bail!(
            "{failures} room read(s) failed (a failed read never delists, and only marks an \
             entry unreachable once it has gone a day without a good read)"
        );
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
            Some(TextSource::Curator) => "curator",
            Some(TextSource::Llm) => "llm",
            Some(TextSource::Owner) | None => "none",
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
                match h.snippet_source {
                    Some(TextSource::Curator) => "curator text",
                    Some(TextSource::Llm) => "llm text",
                    Some(TextSource::Owner) | None => "no description",
                },
                h.availability,
                h.snippet
            );
        }
        println!("{} room(s)", hits.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_common::{sign, IndexDelta, IndexParams, KeyAuth, KeyAuthBody};
    use ed25519_dalek::SigningKey;
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
        let sk = SigningKey::from_bytes(&[3u8; 32]);
        let d = decision_for(&sk.verifying_key(), None);
        match d {
            Decision::Excluded(e, _) => assert!(!e.delists()),
            other => panic!("{other:?}"),
        }
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("atlasctl-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_pointer_floor_round_trips_and_a_corrupt_one_is_an_error() {
        let dir = tmpdir("floor");
        // Missing file: never resolved.
        assert_eq!(load_floor(&dir).unwrap(), PointerFloor::never_resolved());
        let floor = PointerFloor::at(7, [9u8; 32]).unwrap();
        save_floor(&dir, &floor).unwrap();
        assert_eq!(load_floor(&dir).unwrap(), floor);
        let w = PointerFloor::withdrawn_at(8).unwrap();
        save_floor(&dir, &w).unwrap();
        assert_eq!(load_floor(&dir).unwrap(), w);
        // A corrupt file must NOT silently reset to never_resolved.
        std::fs::write(dir.join(FLOOR_FILE), "{not json").unwrap();
        assert!(load_floor(&dir).is_err());
        std::fs::write(
            dir.join(FLOOR_FILE),
            r#"{"version":3,"code_hash":"zzz","withdrawn":false}"#,
        )
        .unwrap();
        assert!(load_floor(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- the wiring from a room state through `sync_one` to signed records --

    // A minimal signed room state: only the fields the reader needs.
    #[derive(Serialize)]
    struct Cfg {
        configuration_version: u32,
        privacy_mode: &'static str,
        display: Disp,
    }
    #[derive(Serialize)]
    struct Disp {
        name: Name,
    }
    #[derive(Serialize)]
    enum Name {
        Public { value: Vec<u8> },
    }
    #[derive(Serialize)]
    struct Auth {
        configuration: Cfg,
        signature: ed25519_dalek::Signature,
    }
    #[derive(Serialize)]
    struct St {
        configuration: Auth,
    }

    fn room_state(sk: &SigningKey, v: u32, mode: &'static str, name: &str) -> Vec<u8> {
        use ed25519_dalek::Signer;
        let c = Cfg {
            configuration_version: v,
            privacy_mode: mode,
            display: Disp {
                name: Name::Public {
                    value: name.as_bytes().to_vec(),
                },
            },
        };
        let mut signed = Vec::new();
        ciborium::ser::into_writer(&c, &mut signed).unwrap();
        let st = St {
            configuration: Auth {
                configuration: c,
                signature: sk.sign(&signed),
            },
        };
        let mut out = Vec::new();
        ciborium::ser::into_writer(&st, &mut out).unwrap();
        out
    }

    fn index_with_keys(dir: &Path) -> (IndexState, IndexParams) {
        let root = SigningKey::from_bytes(&[1u8; 32]);
        let online = SigningKey::from_bytes(&[2u8; 32]);
        std::fs::write(dir.join("online.key"), online.to_bytes()).ok();
        let body = KeyAuthBody {
            version: 1,
            authorized: vec![online.verifying_key()],
        };
        let ka = KeyAuth {
            sig: sign(&body, &root),
            body,
        };
        (
            IndexState::initialized(ka),
            IndexParams {
                root_vk: root.verifying_key(),
                slug: "t".into(),
            },
        )
    }

    /// `room-add` submits the room (`submitted = true`), so a public room with no
    /// policy is signed, added, and accepted by the contract's own verification;
    /// then a refresh against a now-private state signs a tombstone for it.
    #[test]
    fn sync_one_adds_a_public_room_and_a_refresh_delists_it_when_private() {
        let dir = tmpdir("sync");
        let (mut state, params) = index_with_keys(&dir);
        let sk = SigningKey::from_bytes(&[5u8; 32]);
        let vk = sk.verifying_key();
        let st = room_state(&sk, 1, "Public", "Rust Chat");
        let (recs, notes) = sync_one(
            &dir,
            &state,
            &vk,
            Some(st),
            None,
            &MintOptions::default(),
            100,
        )
        .unwrap();
        assert_eq!(recs.len(), 1, "{notes:?}");
        state
            .apply_delta(
                &IndexDelta {
                    key_auth: None,
                    records: recs,
                    apps: None,
                },
                &params,
            )
            .unwrap();
        assert_eq!(state.live_entries().count(), 1);

        let private = room_state(&sk, 2, "Private", "Rust Chat");
        let (recs, _) = sync_one(
            &dir,
            &state,
            &vk,
            Some(private),
            None,
            &MintOptions::default(),
            200,
        )
        .unwrap();
        assert_eq!(recs.len(), 1);
        state
            .apply_delta(
                &IndexDelta {
                    key_auth: None,
                    records: recs,
                    apps: None,
                },
                &params,
            )
            .unwrap();
        assert_eq!(state.live_entries().count(), 0);

        // A failed read (no state) changes nothing authoritative.
        let (recs, _) =
            sync_one(&dir, &state, &vk, None, None, &MintOptions::default(), 300).unwrap();
        assert!(recs.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn existing_rooms_refuse_flags_that_only_shape_a_new_entry() {
        let hit = RoomHit {
            owner_key: "k".into(),
            subject_id: "s".into(),
            title: "t".into(),
            snippet: String::new(),
            snippet_source: None,
            tags: vec![],
            featured: false,
            added_at: 0,
            availability: atlas_rooms::search::Availability::Live,
            last_verified_at: None,
            class: None,
            config_version: None,
        };
        let none = ClassArgs::default();
        assert!(refuse_new_entry_flags_on_existing(Some(&hit), "", false, &none).is_ok());
        assert!(refuse_new_entry_flags_on_existing(Some(&hit), "a", false, &none).is_err());
        assert!(refuse_new_entry_flags_on_existing(Some(&hit), "", true, &none).is_err());
        let ext = ClassArgs {
            ext: vec!["a=b".into()],
            ..Default::default()
        };
        assert!(refuse_new_entry_flags_on_existing(Some(&hit), "", false, &ext).is_err());
        // A new room accepts all of them.
        assert!(refuse_new_entry_flags_on_existing(None, "a", true, &ext).is_ok());
    }
}
