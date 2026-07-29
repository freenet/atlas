//! Shared types for the Atlas discovery layer.
//!
//! This crate is used by both the index contract (compiled to
//! `wasm32-unknown-unknown`, verify-only) and the native curator tools. To keep
//! the contract WASM free of `getrandom`/wasm-bindgen placeholders, anything
//! that needs a CSPRNG (key generation, random subject ids) lives behind the
//! `rng` feature, which only the native crates enable.

use ed25519_dalek::{Signature, SignatureError, Signer, SigningKey, VerifyingKey};
use serde::Serialize;

mod path;
mod state;
mod types;

pub use state::{contract_web_href, IndexDelta, IndexState, IndexSummary};
pub use types::{
    AppRecord, AppRegistry, AppRegistryBody, IndexEntry, IndexParams, KeyAuth, KeyAuthBody, Kind,
    Locator, RecordBody, SignedRecord, SubjectId, Tombstone,
};

/// Max records in a single index contract. Keeps the full state inside the
/// cold-fetch budget; the node also hard-caps contract state at 50 MiB. Beyond
/// this, the index shards (see the design doc, "Scaling beyond one contract").
pub const MAX_ENTRIES: usize = 20_000;
/// Cap on online keys a single key_auth may authorize (bounds verify cost and
/// state size; only the root can set this, but a fat-fingered list shouldn't
/// bloat the index).
pub const MAX_AUTHORIZED: usize = 32;
pub const MAX_TITLE: usize = 200;
pub const MAX_SNIPPET: usize = 500;
pub const MAX_TAGS: usize = 16;
pub const MAX_TAG_LEN: usize = 40;

/// Cap on registered apps in the app registry. The registry is root-signed and
/// resolves every `Locator::AppResource`, so it is small, curated, and bounded
/// well below the entry count.
pub const MAX_APPS: usize = 64;
/// Max length of an app slug (`delta`, `river`, …).
pub const MAX_APP_SLUG: usize = 32;
/// Max length of an app's display name.
pub const MAX_APP_NAME: usize = 64;
/// Max length of an app's link template.
pub const MAX_LINK_TEMPLATE: usize = 128;
/// Max length of an app-hosted resource handle (a Delta site prefix, a River
/// room handle, …). Generous enough for a full base58 verifying key.
pub const MAX_RESOURCE: usize = 64;
/// Max length of an external `https://` url. Previously unbounded, which made
/// `MAX_ENTRIES` meaningless as a byte bound.
pub const MAX_EXTERNAL_URL: usize = 1024;
/// Max length of the path suffix inside any locator. Bounds state growth from a
/// pathological deep link.
pub const MAX_LOCATOR_PATH: usize = 512;

/// Canonical CBOR bytes used as the signing payload for any signed struct.
/// Signing and verification must both go through this so the bytes match.
pub fn canonical<T: Serialize>(value: &T) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(value, &mut buf)
        .expect("CBOR serialization of in-memory Atlas types is infallible");
    buf
}

/// Sign the canonical bytes of `value`. Deterministic (ed25519), no RNG needed.
pub fn sign<T: Serialize>(value: &T, key: &SigningKey) -> Signature {
    key.sign(&canonical(value))
}

/// Verify a signature over the canonical bytes of `value`.
pub fn verify<T: Serialize>(
    value: &T,
    sig: &Signature,
    vk: &VerifyingKey,
) -> Result<(), SignatureError> {
    // verify_strict (not verify) so the per-record (version, sig) merge tie-break
    // rests on non-malleable signatures: it rejects small-order keys and
    // non-canonical R, closing signature-malleability that could otherwise let
    // two distinct valid signatures exist for one (key, message).
    vk.verify_strict(&canonical(value), sig)
}

/// Generate a fresh signing key (native crates only).
#[cfg(feature = "rng")]
pub fn generate_key() -> SigningKey {
    SigningKey::generate(&mut rand::rngs::OsRng)
}
