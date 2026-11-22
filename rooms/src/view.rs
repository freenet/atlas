//! A minimal, read-only view of a River room's state: just the owner-signed
//! configuration fields Atlas needs.
//!
//! # Why this mirrors River's types instead of importing them
//!
//! `river-core` needs freenet-stdlib 0.12 and `atlasctl` links 0.8; two stdlib
//! versions can never link into one binary (`__frnt_set_id` is `#[no_mangle]`).
//! So the shapes read here are copied, and everything that is not read is
//! ignored (serde skips unknown map keys), which also makes this reader tolerant
//! of River adding fields.
//!
//! # Authenticity
//!
//! The configuration is signed by the room owner over
//! `CBOR(Configuration)` (`AuthorizedConfigurationV1::verify_signature`). The
//! configuration is decoded to a generic CBOR value first, and the signature is
//! checked over that value's re-encoding, so the check covers every field,
//! including ones this view does not model (a `public_admission` added by
//! river#974, say). A mirrored struct could not do that: it would drop the
//! unknown field and the re-encoding would no longer match the signed bytes.
//! The round trip is pinned against configurations produced by real
//! `river-core` (the `golden` module in `rooms/tests/rooms.rs`).

use ciborium::Value;
use ed25519_dalek::{Signature, VerifyingKey};
use serde::Deserialize;

#[derive(Debug)]
pub enum ViewError {
    /// Not a CBOR room state, or it has no readable configuration.
    Undecodable(String),
    /// The configuration is not signed by the key the locator names.
    BadSignature,
    /// The signature VERIFIED but the signed configuration does not parse as
    /// this reader expects (River added a privacy variant, reshaped a field).
    /// That is an authoritative owner statement this reader cannot interpret, so
    /// it must fail CLOSED (delist), never be treated as a transient read error.
    SignedButUnreadable(String),
}

impl std::fmt::Display for ViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViewError::Undecodable(e) => write!(f, "room state not decodable: {e}"),
            ViewError::BadSignature => {
                write!(f, "configuration is not signed by the room owner key")
            }
            ViewError::SignedButUnreadable(e) => {
                write!(f, "owner-signed configuration not understood: {e}")
            }
        }
    }
}

#[derive(Deserialize)]
struct StateView {
    configuration: AuthView,
}

#[derive(Deserialize)]
struct AuthView {
    configuration: Value,
    signature: Signature,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Privacy {
    Public,
    Private,
}

/// River's `SealedBytes`. Only the public form carries readable text.
#[derive(Deserialize, Clone, PartialEq, Debug)]
pub enum Sealed {
    Public { value: Vec<u8> },
    Private {},
}

impl Sealed {
    pub fn public_bytes(&self) -> Option<&[u8]> {
        match self {
            Sealed::Public { value } => Some(value),
            Sealed::Private {} => None,
        }
    }
}

#[derive(Deserialize, Clone, PartialEq, Debug)]
pub struct Display {
    pub name: Sealed,
    // The owner's description is deliberately not read: it is not indexed, and a
    // client reads it live from the room.
}

/// PENDING river#974 (branch `feat/public-admission`, not yet merged):
/// `Configuration.public_admission: Option<PublicRoomPolicy>`. Only `listed` is
/// read. The name is pinned by golden fixtures generated from that branch's
/// `river-core`, so a rename fails `golden::public_admission_*` instead of
/// silently turning every opt-out into "no policy" (which is eligible).
/// The field name is the coordinator's working shape and is #974's to finalise;
/// see [`RoomConfig::listed`].
#[derive(Deserialize, Clone, Debug)]
struct PolicyView {
    #[serde(default)]
    listed: bool,
}

#[derive(Deserialize)]
struct ConfigView {
    configuration_version: u32,
    privacy_mode: Privacy,
    display: Display,
    // PENDING river#974: until it merges, no deployed configuration carries
    // this and it is always `None`.
    #[serde(default)]
    public_admission: Option<PolicyView>,
}

/// What Atlas reads from a room, after the owner signature has been verified.
#[derive(Clone, Debug)]
pub struct RoomConfig {
    pub configuration_version: u32,
    pub privacy: Privacy,
    pub display: Display,
    /// The owner's `Configuration.public_admission.listed` opt-in.
    ///
    /// HOOK for river#974. `None` means the configuration carries no policy
    /// (every room today); callers must then fall back to an explicit curator or
    /// owner submission. `Some(false)` is an explicit opt-out and always wins.
    pub listed: Option<bool>,
}

impl RoomConfig {
    /// Decode a room's CBOR state and verify the owner's signature over its
    /// configuration with the key the locator names.
    pub fn from_state(state: &[u8], owner: &VerifyingKey) -> Result<Self, ViewError> {
        let view: StateView =
            ciborium::de::from_reader(state).map_err(|e| ViewError::Undecodable(e.to_string()))?;
        let mut signed = Vec::new();
        ciborium::ser::into_writer(&view.configuration.configuration, &mut signed)
            .map_err(|e| ViewError::Undecodable(e.to_string()))?;
        // `verify_strict`, like `atlas_common::verify`: rejects small-order keys
        // and non-canonical signatures, so a weak owner key cannot validate a
        // signature over arbitrary bytes.
        owner
            .verify_strict(&signed, &view.configuration.signature)
            .map_err(|_| ViewError::BadSignature)?;
        let cfg: ConfigView = view
            .configuration
            .configuration
            .deserialized()
            .map_err(|e| ViewError::SignedButUnreadable(e.to_string()))?;
        Ok(RoomConfig {
            configuration_version: cfg.configuration_version,
            privacy: cfg.privacy_mode,
            display: cfg.display,
            listed: cfg.public_admission.map(|p| p.listed),
        })
    }
}
