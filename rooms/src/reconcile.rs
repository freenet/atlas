//! Deciding what to publish for one room, given what the index already holds.
//!
//! Pure: no network, no signing. The caller supplies the current `IndexState`
//! and the freshly evaluated room, and gets back the records to sign.

use atlas_common::{IndexEntry, IndexState, Tombstone};
use ed25519_dalek::VerifyingKey;

use crate::eligibility::{room_locator, Decision, Exclusion};
use crate::entry::{
    mark_unreachable, new_entry, next_entry, stored_config_version, Characterization,
};

/// Options only used when a room is minted for the first time.
#[derive(Clone, Debug, Default)]
pub struct MintOptions {
    pub tags: Vec<String>,
    pub featured: bool,
    pub class: Option<atlas_common::Classification>,
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

fn live_for<'a>(state: &'a IndexState, owner: &VerifyingKey) -> Vec<&'a IndexEntry> {
    let key = room_locator(owner).dedup_key();
    let mut v: Vec<&IndexEntry> = state
        .live_entries()
        .filter(|e| e.locator.dedup_key() == key)
        .collect();
    // Canonical subject first: the highest version, which is the one a curator
    // most recently touched; ties go to the earliest mint.
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

/// Plan the publishes for `owner`'s room.
///
/// - **Duplicates** (several live entries for one owner key, however they got
///   there): the canonical one is kept and updated, the rest are tombstoned.
/// - **Renamed**: the owner's new name flows into `title` as a higher version of
///   the SAME subject; the identity is the owner key, never the name.
/// - **Stale**: a read at a lower owner-signed `configuration_version` than the
///   entry was built from changes nothing, in either direction (it neither
///   reverts a rename nor delists on an old `Private`).
/// - **Private / unlisted**: delists, including any duplicates.
/// - **Unverifiable** (undecodable state, bad signature): never delists; the
///   entry is marked unreachable.
pub fn plan(
    state: &IndexState,
    owner: &VerifyingKey,
    decision: &Decision,
    ch: Option<&Characterization>,
    mint: &MintOptions,
    now: u64,
) -> Result<Plan, String> {
    let existing = live_for(state, owner);
    let mut actions = Vec::new();
    let mut notes = Vec::new();

    match decision {
        Decision::Eligible(desc) => match existing.split_first() {
            None => {
                let e = new_entry(
                    desc,
                    ch,
                    mint.tags.clone(),
                    mint.featured,
                    mint.class.clone(),
                    None,
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
        },
        Decision::Excluded(why, cv) => {
            if existing.is_empty() {
                notes.push(format!("not indexed: {why}"));
            } else if why.delists() {
                let stale = cv.is_some_and(|v| {
                    existing
                        .iter()
                        .filter_map(|e| stored_config_version(e))
                        .max()
                        .is_some_and(|stored| v < stored)
                });
                if stale {
                    notes.push(format!("ignored stale read ({why})"));
                } else {
                    for e in &existing {
                        notes.push(format!("delist {:?}: {why}", e.title));
                        actions.push(Action::Tombstone(tomb(e)?));
                    }
                }
            } else {
                // `Unverifiable`: say so, change nothing authoritative.
                debug_assert!(matches!(why, Exclusion::Unverifiable(_)));
                match existing.first().map(|e| mark_unreachable(e, now)) {
                    Some(Ok(Some(e))) => {
                        notes.push(format!("mark {:?} unreachable ({why})", e.title));
                        actions.push(Action::Update(Box::new(e)));
                    }
                    Some(Ok(None)) => notes.push(format!("already marked unreachable ({why})")),
                    Some(Err(e)) => return Err(e),
                    None => {}
                }
            }
        }
    }
    Ok(Plan { actions, notes })
}
