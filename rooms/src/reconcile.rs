//! Deciding what to publish for one room, given what the index already holds.
//!
//! Pure: no network, no signing. The caller supplies the current `IndexState`
//! and the freshly evaluated room, and gets back the records to sign.

use std::collections::BTreeMap;

use atlas_common::{IndexEntry, IndexState, Kind, Tombstone};
use ed25519_dalek::VerifyingKey;

use crate::eligibility::{room_locator, Decision};
use crate::entry::{
    mark_unreachable, new_entry, next_entry, stored_config_version, Characterization, ROOM_KIND,
};

/// Options only used when a room is minted for the first time.
#[derive(Clone, Debug, Default)]
pub struct MintOptions {
    pub tags: Vec<String>,
    pub featured: bool,
    pub class: Option<atlas_common::Classification>,
    /// Extra `ext` pairs from the curator (merged under the keys this crate owns).
    pub ext: Option<BTreeMap<String, String>>,
}

#[derive(Debug)]
pub enum Action {
    /// Publish a new live entry (version 1, fresh subject id).
    Add(Box<IndexEntry>),
    /// Publish a higher-versioned live entry for an existing subject.
    Update(Box<IndexEntry>),
    /// Publish a tombstone: the room is private, unlisted, or a duplicate.
    Tombstone(Tombstone),
}

#[derive(Debug)]
pub struct Plan {
    pub actions: Vec<Action>,
    /// One human-readable line per decision, including the do-nothing ones.
    pub notes: Vec<String>,
}

fn same_locator<'a>(
    state: &'a IndexState,
    owner: &VerifyingKey,
) -> impl Iterator<Item = &'a IndexEntry> {
    let key = room_locator(owner).dedup_key();
    state
        .live_entries()
        .filter(move |e| e.locator.dedup_key() == key)
}

/// Live `Room` entries for this owner key, canonical first: the highest version
/// (the one a curator most recently touched), ties to the earliest mint.
fn live_for<'a>(state: &'a IndexState, owner: &VerifyingKey) -> Vec<&'a IndexEntry> {
    let room = Kind::new(ROOM_KIND);
    let mut v: Vec<&IndexEntry> = same_locator(state, owner)
        .filter(|e| e.kind == room)
        .collect();
    v.sort_by(|a, b| {
        b.version
            .cmp(&a.version)
            .then(a.added_at.cmp(&b.added_at))
            .then(a.subject_id.cmp(&b.subject_id))
    });
    v
}

fn tomb(e: &IndexEntry) -> Result<Tombstone, String> {
    Ok(Tombstone {
        subject_id: e.subject_id.clone(),
        version: e.version.checked_add(1).ok_or("version overflow")?,
    })
}

/// The newest owner-signed configuration version any listing was built from.
fn max_stored_cv(existing: &[&IndexEntry]) -> Option<u32> {
    existing
        .iter()
        .filter_map(|e| stored_config_version(e))
        .max()
}

/// Plan the publishes for `owner`'s room.
///
/// - **Duplicates** (several live `Room` entries for one owner key): the
///   canonical one is kept and updated, the rest are tombstoned.
/// - **Renamed**: the owner's new name flows into `title` as a higher version of
///   the SAME subject; the identity is the owner key, never the name.
/// - **Stale**: a read at a lower owner-signed `configuration_version` than ANY
///   listing for this owner was built from changes nothing, in either direction
///   (it neither reverts a rename nor delists on an old `Private`). Equal is not
///   stale.
/// - **Private / unlisted / unsupported**: delists, including any duplicates.
/// - **Unverifiable** (undecodable state, bad signature): never delists; the
///   entry is marked unreachable once it has gone unverified past a grace period.
/// - **Collision**: a live entry of a different kind (`Site`, `App`) at this
///   room's locator is never retitled or tombstoned by room logic; adding is
///   refused with an error naming it, so a curator resolves it deliberately.
pub fn plan(
    state: &IndexState,
    owner: &VerifyingKey,
    decision: &Decision,
    ch: Option<&Characterization>,
    mint: &MintOptions,
    now: u64,
) -> Result<Plan, String> {
    let existing = live_for(state, owner);
    let max_cv = max_stored_cv(&existing);
    let mut actions = Vec::new();
    let mut notes = Vec::new();

    match decision {
        Decision::Eligible(desc) => {
            if max_cv.is_some_and(|m| desc.configuration_version < m) {
                notes.push(format!(
                    "ignored stale read (configuration {} is older than {})",
                    desc.configuration_version,
                    max_cv.unwrap_or(0)
                ));
                return Ok(Plan { actions, notes });
            }
            match existing.split_first() {
                None => {
                    if let Some(f) =
                        same_locator(state, owner).find(|e| e.kind != Kind::new(ROOM_KIND))
                    {
                        return Err(format!(
                            "locator {} is already used by subject {} of kind {}; \
                             remove or re-kind that entry before adding the room",
                            f.locator.to_uri(),
                            f.subject_id.as_str(),
                            f.kind.as_str()
                        ));
                    }
                    let e = new_entry(
                        desc,
                        ch,
                        mint.tags.clone(),
                        mint.featured,
                        mint.class.clone(),
                        mint.ext.clone(),
                        now,
                    )?;
                    notes.push(format!("add {:?}", e.title));
                    actions.push(Action::Add(Box::new(e)));
                }
                Some((canon, dupes)) => {
                    match next_entry(canon, desc, ch, now)? {
                        Some(e) => {
                            notes.push(if e.title != canon.title {
                                format!("rename {:?} -> {:?}", canon.title, e.title)
                            } else {
                                format!("update {:?}", e.title)
                            });
                            actions.push(Action::Update(Box::new(e)));
                        }
                        None => notes.push(format!("{:?} is up to date", canon.title)),
                    }
                    for d in dupes {
                        notes.push(format!(
                            "tombstone duplicate subject {}",
                            d.subject_id.as_str()
                        ));
                        actions.push(Action::Tombstone(tomb(d)?));
                    }
                }
            }
        }
        Decision::Excluded(why, cv) => {
            if existing.is_empty() {
                notes.push(format!("not indexed: {why}"));
            } else if why.delists() {
                let stale = matches!((cv, max_cv), (Some(v), Some(m)) if *v < m);
                if stale {
                    notes.push(format!("ignored stale read ({why})"));
                } else {
                    for e in &existing {
                        notes.push(format!("delist {:?}: {why}", e.title));
                        actions.push(Action::Tombstone(tomb(e)?));
                    }
                }
            } else {
                // Unverifiable: change nothing authoritative.
                if let Some(first) = existing.first() {
                    match mark_unreachable(first, now)? {
                        Some(e) => {
                            notes.push(format!("mark {:?} unreachable ({why})", e.title));
                            actions.push(Action::Update(Box::new(e)));
                        }
                        None => notes.push(format!("read failed, entry left as is ({why})")),
                    }
                }
            }
        }
    }
    Ok(Plan { actions, notes })
}
