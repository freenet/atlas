//! Turning a [`RoomDescriptor`] into an `IndexEntry`, and refreshing one.
//!
//! # Schema (documented in the crate docs, README and `FREENET.md`)
//!
//! | `IndexEntry` field | Room meaning |
//! |---|---|
//! | `kind` | `"Room"` |
//! | `locator` | `AppResource { app: "river", resource: <owner key base58>, path: "" }` |
//! | `title` | the owner's room NAME, a sanitised snapshot (`ext.ts = "o"`); required non-empty by the contract |
//! | `snippet` | Atlas's own characterisation of the room, or empty; its source is `ext.ss` |
//! | `ext.ss` | `c` curator or `l` LLM; absent when `snippet` is empty |
//! | `ext.cv` | owner-signed `configuration_version` the name was read at |
//! | `ext.rv` | room-descriptor schema version, `"1"` |
//! | `tags`, `class`, `featured` | curator-asserted, as for any entry |
//! | `verified` | when Atlas last read the room's signed state, and what it found |
//!
//! # The owner's description is NOT stored in the index
//!
//! The owner-signed `RoomDisplayMetadata` description is already authoritative in
//! the room's own state, so a client (River's Discover Rooms) reads it live from
//! the room. The Atlas entry carries only what Atlas itself asserts, so the two
//! can never be confused or merged, and a stale copy of the owner's words can
//! never outlive an edit. The one owner-sourced field is the `title` snapshot,
//! which the contract requires and which exists so a card can render without
//! first fetching the room.
//!
//! Everything lives in existing open fields. There is no new typed field, so no
//! change under `common/` and no re-key. `ext` uses under 24 bytes of its
//! 64-byte budget.

use std::collections::BTreeMap;

use atlas_common::{
    Classification, IndexEntry, Kind, SubjectId, Verification, VerifyStatus, MAX_SNIPPET,
};

use crate::eligibility::{room_locator, RoomDescriptor};
use crate::text::sanitize_line;
use ed25519_dalek::VerifyingKey;

pub const ROOM_KIND: &str = "Room";
pub const EXT_TITLE_SRC: &str = "ts";
pub const EXT_SNIPPET_SRC: &str = "ss";
pub const EXT_CONFIG_VERSION: &str = "cv";
pub const EXT_SCHEMA: &str = "rv";
pub const SCHEMA_VERSION: &str = "1";

/// A listing is re-stamped as verified at most this often when nothing else
/// changed, so a refresh sweep does not republish every room on every run.
pub const REVERIFY_AFTER_SECS: u64 = 7 * 24 * 3600;

/// Who wrote the text on a card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextSource {
    Owner,
    Curator,
    Llm,
}

impl TextSource {
    pub fn code(self) -> &'static str {
        match self {
            TextSource::Owner => "o",
            TextSource::Curator => "c",
            TextSource::Llm => "l",
        }
    }
    pub fn from_code(s: &str) -> Option<Self> {
        match s {
            "o" => Some(TextSource::Owner),
            "c" => Some(TextSource::Curator),
            "l" => Some(TextSource::Llm),
            _ => None,
        }
    }
}

/// An independent characterisation of a room, by a curator or an LLM. Never the
/// owner: owner text comes from the room state.
#[derive(Clone, Debug)]
pub struct Characterization {
    pub text: String,
    pub source: TextSource,
}

impl Characterization {
    pub fn new(text: &str, source: TextSource) -> Result<Self, String> {
        if source == TextSource::Owner {
            return Err("a characterization is never owner-sourced".to_string());
        }
        let text = sanitize_line(text, MAX_SNIPPET)
            .ok_or_else(|| "characterization has nothing printable in it".to_string())?;
        Ok(Self { text, source })
    }
}

fn ext_for(
    prev: Option<&BTreeMap<String, String>>,
    cv: u32,
    snippet_src: Option<TextSource>,
) -> BTreeMap<String, String> {
    // Carry forward keys this module does not own (classification flags, the
    // crawler's keys) so a refresh cannot erase another writer's metadata.
    let mut m = prev.cloned().unwrap_or_default();
    m.insert(EXT_TITLE_SRC.into(), TextSource::Owner.code().into());
    match snippet_src {
        Some(src) => m.insert(EXT_SNIPPET_SRC.into(), src.code().into()),
        None => m.remove(EXT_SNIPPET_SRC),
    };
    m.insert(EXT_CONFIG_VERSION.into(), cv.to_string());
    m.insert(EXT_SCHEMA.into(), SCHEMA_VERSION.into());
    m
}

fn verified_live(now: u64) -> Verification {
    Verification {
        last_verified_at: now,
        status: VerifyStatus::new(VerifyStatus::LIVE),
    }
}

/// The snippet and its source: Atlas's characterisation if there is one, else
/// empty. Never the owner's description (see the module docs).
fn card_text(ch: Option<&Characterization>) -> (String, Option<TextSource>) {
    match ch {
        Some(c) => (c.text.clone(), Some(c.source)),
        None => (String::new(), None),
    }
}

/// A brand-new room entry at version 1. `class` and `tags` come from the curator;
/// without a classification the entry is NOT ASSESSED and Atlas's safe search
/// will hide it, which is the honest default.
pub fn new_entry(
    desc: &RoomDescriptor,
    ch: Option<&Characterization>,
    tags: Vec<String>,
    featured: bool,
    class: Option<Classification>,
    extra_ext: Option<BTreeMap<String, String>>,
    now: u64,
) -> Result<IndexEntry, String> {
    let owner = VerifyingKey::from_bytes(&desc.owner).map_err(|e| e.to_string())?;
    let (snippet, src) = card_text(ch);
    Ok(IndexEntry {
        subject_id: SubjectId::random(),
        version: 1,
        kind: Kind::new(ROOM_KIND),
        title: desc.name.clone(),
        snippet,
        tags,
        locator: room_locator(&owner),
        featured,
        added_at: now,
        class,
        verified: Some(verified_live(now)),
        ext: Some(ext_for(extra_ext.as_ref(), desc.configuration_version, src)),
    })
}

/// The stored `cv`, if the entry has one.
pub fn stored_config_version(e: &IndexEntry) -> Option<u32> {
    e.ext.as_ref()?.get(EXT_CONFIG_VERSION)?.parse().ok()
}

/// Who wrote the snippet: `None` when it is empty or unmarked.
pub fn snippet_source(e: &IndexEntry) -> Option<TextSource> {
    e.ext
        .as_ref()
        .and_then(|m| m.get(EXT_SNIPPET_SRC))
        .and_then(|c| TextSource::from_code(c))
        .filter(|s| *s != TextSource::Owner)
}

/// The next version of `current` after re-reading the room, or `None` when
/// nothing worth publishing changed.
///
/// A changed body is always at a strictly higher version (see `atlasctl`'s
/// `next_entry` for why that is not optional), and an unchanged one is never
/// re-signed: peers would fetch it for nothing.
///
/// Renames come through here: the owner's new name replaces `title`. The
/// snippet is Atlas's and is kept unless a new characterisation is supplied.
pub fn next_entry(
    current: &IndexEntry,
    desc: &RoomDescriptor,
    ch: Option<&Characterization>,
    now: u64,
) -> Result<Option<IndexEntry>, String> {
    let cv = desc.configuration_version;
    if let Some(stored) = stored_config_version(current) {
        if cv < stored {
            // A node answering from a copy older than the one this entry was
            // built from. Acting on it would revert a rename.
            return Ok(None);
        }
    }
    let (snippet, src) = match ch {
        Some(c) => (c.text.clone(), Some(c.source)),
        None => (current.snippet.clone(), snippet_source(current)),
    };
    let ext = ext_for(current.ext.as_ref(), cv, src);
    let live_and_fresh = current.verified.as_ref().is_some_and(|v| {
        v.status.as_str() == VerifyStatus::LIVE
            && now.saturating_sub(v.last_verified_at) < REVERIFY_AFTER_SECS
    });
    if current.title == desc.name
        && current.snippet == snippet
        && current.ext.as_ref() == Some(&ext)
        && live_and_fresh
    {
        return Ok(None);
    }
    Ok(Some(IndexEntry {
        version: current
            .version
            .checked_add(1)
            .ok_or("entry version would overflow")?,
        title: desc.name.clone(),
        snippet,
        ext: Some(ext),
        verified: Some(verified_live(now)),
        ..current.clone()
    }))
}

/// Mark an entry as not currently reachable, after a read that proved nothing.
/// `None` if it is already marked, so repeated failures do not churn versions.
/// This is not a takedown: transient unreachability is normal on Freenet.
pub fn mark_unreachable(current: &IndexEntry, now: u64) -> Result<Option<IndexEntry>, String> {
    if current
        .verified
        .as_ref()
        .is_some_and(|v| v.status.as_str() == VerifyStatus::UNREACHABLE)
    {
        return Ok(None);
    }
    Ok(Some(IndexEntry {
        version: current
            .version
            .checked_add(1)
            .ok_or("entry version would overflow")?,
        verified: Some(Verification {
            last_verified_at: now,
            status: VerifyStatus::new(VerifyStatus::UNREACHABLE),
        }),
        ..current.clone()
    }))
}
