//! The client-facing read API for River's Discover Rooms.
//!
//! A client already holds the whole `IndexState` (that is how the index is read),
//! so this is a pure function over it rather than a service. It mirrors the
//! Atlas UI's own rules so a River client cannot show something Atlas hides:
//! safe search matches the exact permissive `landing`, and an unclassified entry
//! is hidden while it is on.
//!
//! The `owner_key` on a hit is the FULL owner key. It is what River resolves to
//! a contract (via its own pointer) and what a `?room=` link carries; the card
//! text is only a snapshot, and River should read the owner's current name and
//! description from the room state itself once it has joined or previewed it.

use atlas_common::{Audience, Classification, IndexEntry, IndexState, Kind, VerifyStatus};

use crate::eligibility::{owner_of_locator, parse_owner};
use crate::entry::{snippet_source, stored_config_version, TextSource, ROOM_KIND};

/// What Atlas last learned about whether the room can be joined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Availability {
    /// Read and verified recently enough.
    Live,
    /// The last read proved nothing. Not a takedown.
    Unreachable,
    /// Verified, but longer ago than the caller's `stale_after_secs`.
    Stale,
    /// Never verified.
    Unknown,
}

#[derive(Clone, Debug)]
pub struct RoomHit {
    /// Full owner verifying key, base58. The room's identity.
    pub owner_key: String,
    pub subject_id: String,
    pub title: String,
    pub snippet: String,
    /// Who wrote `snippet`: the owner, a curator, or an LLM.
    pub snippet_source: TextSource,
    pub tags: Vec<String>,
    pub featured: bool,
    pub added_at: u64,
    pub availability: Availability,
    pub last_verified_at: Option<u64>,
    pub class: Option<Classification>,
    /// Owner-signed configuration version the snapshot was read at.
    pub config_version: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct RoomFilter {
    /// Case-insensitive substring over title, snippet and tags.
    pub query: Option<String>,
    /// Every listed tag must be present (exact, case-insensitive).
    pub tags: Vec<String>,
    /// Hide anything whose landing is not exactly `general`, and everything
    /// unclassified. On by default, matching the Atlas UI.
    pub safe_search: bool,
    /// Drop rooms whose availability is not `Live`.
    pub only_live: bool,
    /// Verifications older than this many seconds count as `Stale`.
    pub stale_after_secs: Option<u64>,
    /// "Now", for staleness. Callers pass it; there is no clock in here.
    pub now: u64,
}

impl Default for RoomFilter {
    fn default() -> Self {
        Self {
            query: None,
            tags: Vec::new(),
            safe_search: true,
            only_live: false,
            stale_after_secs: None,
            now: 0,
        }
    }
}

fn availability(e: &IndexEntry, f: &RoomFilter) -> Availability {
    let Some(v) = &e.verified else {
        return Availability::Unknown;
    };
    match v.status.as_str() {
        VerifyStatus::UNREACHABLE => Availability::Unreachable,
        VerifyStatus::LIVE => {
            let stale = f
                .stale_after_secs
                .is_some_and(|s| f.now.saturating_sub(v.last_verified_at) > s);
            if stale {
                Availability::Stale
            } else {
                Availability::Live
            }
        }
        // `changed` or an unknown future value: do not present as joinable.
        _ => Availability::Unknown,
    }
}

fn passes_safe(h: &RoomHit) -> bool {
    // Exact permissive value only, so an unknown future landing fails closed.
    h.class
        .as_ref()
        .is_some_and(|c| c.landing.as_str() == Audience::GENERAL)
}

fn hit(e: &IndexEntry, f: &RoomFilter) -> Option<RoomHit> {
    if e.kind != Kind::new(ROOM_KIND) {
        return None;
    }
    // A room entry whose locator does not name a valid owner key is malformed.
    // Skip it rather than hand a client a key it cannot use.
    let owner = owner_of_locator(&e.locator)?;
    Some(RoomHit {
        owner_key: crate::eligibility::owner_b58(&owner),
        subject_id: e.subject_id.as_str().to_string(),
        title: e.title.clone(),
        snippet: e.snippet.clone(),
        snippet_source: snippet_source(e),
        tags: e.tags.clone(),
        featured: e.featured,
        added_at: e.added_at,
        availability: availability(e, f),
        last_verified_at: e.verified.as_ref().map(|v| v.last_verified_at),
        class: e.class.clone(),
        config_version: stored_config_version(e),
    })
}

fn score(h: &RoomHit, q: &str) -> u8 {
    let q = q.to_lowercase();
    if h.title.to_lowercase().contains(&q) {
        3
    } else if h.tags.iter().any(|t| t.to_lowercase().contains(&q)) {
        2
    } else if h.snippet.to_lowercase().contains(&q) {
        1
    } else {
        0
    }
}

/// Live room entries matching `filter`, best first.
///
/// Order: query relevance (title, then tag, then snippet), featured, newest
/// listing, then owner key so the order is total and stable across clients.
/// Duplicate listings of one owner key collapse to the highest-versioned entry.
pub fn search(state: &IndexState, filter: &RoomFilter) -> Vec<RoomHit> {
    let mut best: std::collections::BTreeMap<String, (u64, RoomHit)> = Default::default();
    for e in state.live_entries() {
        let Some(h) = hit(e, filter) else { continue };
        match best.get(&h.owner_key) {
            Some((v, _)) if *v >= e.version => {}
            _ => {
                best.insert(h.owner_key.clone(), (e.version, h));
            }
        }
    }
    let wanted: Vec<String> = filter.tags.iter().map(|t| t.to_lowercase()).collect();
    let mut hits: Vec<(u8, RoomHit)> = best
        .into_values()
        .map(|(_, h)| h)
        .filter(|h| !filter.safe_search || passes_safe(h))
        .filter(|h| !filter.only_live || h.availability == Availability::Live)
        .filter(|h| {
            wanted
                .iter()
                .all(|w| h.tags.iter().any(|t| t.to_lowercase() == *w))
        })
        .filter_map(
            |h| match filter.query.as_deref().filter(|q| !q.is_empty()) {
                None => Some((0, h)),
                Some(q) => {
                    let s = score(&h, q);
                    (s > 0).then_some((s, h))
                }
            },
        )
        .collect();
    hits.sort_by(|(sa, a), (sb, b)| {
        sb.cmp(sa)
            .then(b.featured.cmp(&a.featured))
            .then(b.added_at.cmp(&a.added_at))
            .then(a.owner_key.cmp(&b.owner_key))
    });
    hits.into_iter().map(|(_, h)| h).collect()
}

/// The single listing for one owner key, if any, ignoring safe search.
pub fn find_room(state: &IndexState, owner_key: &str, filter: &RoomFilter) -> Option<RoomHit> {
    let owner = crate::eligibility::owner_b58(&parse_owner(owner_key).ok()?);
    let f = RoomFilter {
        safe_search: false,
        ..filter.clone()
    };
    search(state, &f).into_iter().find(|h| h.owner_key == owner)
}
