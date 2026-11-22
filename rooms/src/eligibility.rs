//! Who gets indexed. The gate is deliberately a list of REASONS TO EXCLUDE, each
//! of which maps to a different action when the room is already listed.

use atlas_common::{Locator, MAX_TITLE};
use ed25519_dalek::VerifyingKey;

use crate::text::sanitize_line;
use crate::view::{Privacy, RoomConfig, ViewError};

/// App slug under which River rooms are indexed (`Locator::AppResource::app`).
pub const RIVER_APP: &str = "river";

/// Parse a room owner key from its base58 text form (the full key, 32 bytes).
pub fn parse_owner(s: &str) -> Result<VerifyingKey, String> {
    let bytes = bs58::decode(s)
        .into_vec()
        .map_err(|e| format!("owner key is not base58: {e}"))?;
    let arr: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| format!("owner key is {} bytes, expected 32", bytes.len()))?;
    let vk = VerifyingKey::from_bytes(&arr)
        .map_err(|e| format!("owner key is not a valid ed25519 key: {e}"))?;
    // A small-order key can validate signatures over arbitrary messages, so it
    // cannot identify an owner.
    if vk.is_weak() {
        return Err("owner key is a weak (small-order) key".to_string());
    }
    Ok(vk)
}

pub fn owner_b58(owner: &VerifyingKey) -> String {
    bs58::encode(owner.to_bytes()).into_string()
}

/// The canonical locator for a room: the FULL owner key, empty path. Never a
/// contract id (which changes with every room-WASM re-key) and never a short
/// handle (display-only, per the river#972 interface decisions).
pub fn room_locator(owner: &VerifyingKey) -> Locator {
    Locator::AppResource {
        app: RIVER_APP.to_string(),
        resource: owner_b58(owner),
        path: String::new(),
    }
}

/// The owner key a River room locator names, or `None` for any other locator.
pub fn owner_of_locator(loc: &Locator) -> Option<VerifyingKey> {
    match loc {
        Locator::AppResource { app, resource, .. } if app == RIVER_APP => {
            parse_owner(resource).ok()
        }
        _ => None,
    }
}

/// A room that passed the gate. Holds only what the index stores: the owner key,
/// the configuration version the read was at, and the sanitised room NAME. The
/// owner's description is deliberately absent (clients read it from the room).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoomDescriptor {
    pub owner: [u8; 32],
    /// Owner-signed `configuration_version`. Monotonic, so a lower value than
    /// one already seen is a stale read.
    pub configuration_version: u32,
    /// Owner-supplied room name, sanitised and cut to `MAX_TITLE` bytes.
    pub name: String,
}

/// Why a room must NOT be (or stay) in the index.
///
/// The split matters to the caller: the first group is an authoritative,
/// owner-signed statement about the room and means DELIST; the second group
/// says only that this read could not establish anything, and must never delist
/// (one hostile or lagging node would otherwise be able to remove any room).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Exclusion {
    /// Owner-signed `PrivacyMode::Private`.
    Private,
    /// Public mode but the name is encrypted, so there is nothing to show.
    EncryptedName,
    /// The owner explicitly opted out (`listed == false`), or, with no policy
    /// field in existence, nobody has submitted the room.
    NotListed,
    /// The name is empty, not UTF-8, or has nothing printable in it.
    UnusableName,
    /// The read proved nothing: undecodable state or a bad owner signature.
    Unverifiable(String),
    /// The configuration is owner-signed but this reader cannot interpret it.
    /// Fails closed: an owner statement we cannot read must not keep a room
    /// listed.
    UnsupportedConfig(String),
}

impl Exclusion {
    /// True when the exclusion is an authoritative statement about the room and
    /// an existing listing should be removed. False for [`Exclusion::Unverifiable`].
    pub fn delists(&self) -> bool {
        !matches!(self, Exclusion::Unverifiable(_))
    }
}

impl std::fmt::Display for Exclusion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exclusion::Private => write!(f, "room is private"),
            Exclusion::EncryptedName => write!(f, "room name is encrypted"),
            Exclusion::NotListed => write!(f, "room is not listed for discovery"),
            Exclusion::UnusableName => write!(f, "room name is empty or unprintable"),
            Exclusion::Unverifiable(e) => write!(f, "could not verify room state: {e}"),
            Exclusion::UnsupportedConfig(e) => {
                write!(f, "owner-signed configuration not understood: {e}")
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Eligible(RoomDescriptor),
    /// The reason, and the owner-signed `configuration_version` it was read at
    /// (`None` when nothing was verified). The version lets a caller discard a
    /// stale read instead of acting on it.
    Excluded(Exclusion, Option<u32>),
}

/// Apply the gate to a room's raw state.
///
/// A room is eligible only if ALL of:
/// 1. the owner's signature over its configuration verifies against `owner`;
/// 2. `privacy_mode` is `Public`;
/// 3. the owner has not opted out, and either opted in (`listed == Some(true)`,
///    river#974) or the room was explicitly `submitted` to Atlas by a curator or
///    the owner. `submitted` is the pre-#974 stand-in for the listed flag.
/// 4. its name is plaintext with something printable in it.
///
/// Private rooms are rejected on the owner-signed mode alone, BEFORE any text is
/// read, so nothing about a private room's content ever reaches a descriptor.
pub fn evaluate(owner: &VerifyingKey, state: &[u8], submitted: bool) -> Decision {
    let cfg = match RoomConfig::from_state(state, owner) {
        Ok(c) => c,
        Err(ViewError::SignedButUnreadable(e)) => {
            return Decision::Excluded(Exclusion::UnsupportedConfig(e), None)
        }
        Err(e @ (ViewError::Undecodable(_) | ViewError::BadSignature)) => {
            return Decision::Excluded(Exclusion::Unverifiable(e.to_string()), None)
        }
    };
    decide(owner, &cfg, submitted)
}

pub(crate) fn decide(owner: &VerifyingKey, cfg: &RoomConfig, submitted: bool) -> Decision {
    let cv = Some(cfg.configuration_version);
    if cfg.privacy != Privacy::Public {
        return Decision::Excluded(Exclusion::Private, cv);
    }
    // The listed hook (river#974). An explicit opt-out wins over a submission.
    match cfg.listed {
        Some(false) => return Decision::Excluded(Exclusion::NotListed, cv),
        Some(true) => {}
        None if submitted => {}
        None => return Decision::Excluded(Exclusion::NotListed, cv),
    }
    let Some(name_bytes) = cfg.display.name.public_bytes() else {
        return Decision::Excluded(Exclusion::EncryptedName, cv);
    };
    let name = std::str::from_utf8(name_bytes)
        .ok()
        .and_then(|s| sanitize_line(s, MAX_TITLE));
    let Some(name) = name else {
        return Decision::Excluded(Exclusion::UnusableName, cv);
    };
    Decision::Eligible(RoomDescriptor {
        owner: owner.to_bytes(),
        configuration_version: cfg.configuration_version,
        name,
    })
}
