//! Shared types for the Atlas discovery layer.
//!
//! This crate is used by both the index contract (compiled to
//! `wasm32-unknown-unknown`, verify-only) and the native curator tools. To keep
//! the contract WASM free of `getrandom`/wasm-bindgen placeholders, anything
//! that needs a CSPRNG (key generation, random subject ids) lives behind the
//! `rng` feature, which only the native crates enable.

use ed25519_dalek::{Signature, SignatureError, Signer, SigningKey, VerifyingKey};
use serde::Serialize;

pub mod path;
mod state;
mod types;

pub use state::{contract_web_href, IndexDelta, IndexState, IndexSummary};
pub use types::{
    has_raw_control_char, AppRecord, AppRegistry, AppRegistryBody, Audience, Classification,
    IndexEntry, IndexParams, KeyAuth, KeyAuthBody, Kind, Locator, RecordBody, SignedRecord,
    SubjectId, Tombstone, Verification, VerifyStatus, Volatility,
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
///
/// Sized so that `MAX_ENTRIES` worst-case records still fit the node's 50 MiB
/// state cap — at 1024 they did NOT (51.5 MiB), which made the bound finite but
/// not sufficient. `max_entries_worst_case_fits_the_node_state_cap` pins the
/// relationship, so raising any per-entry bound fails loudly instead of quietly
/// making a full index unstorable.
pub const MAX_EXTERNAL_URL: usize = 512;
/// Max length of the path suffix inside any locator. Bounds state growth from a
/// pathological deep link.
pub const MAX_LOCATOR_PATH: usize = 512;

/// Max length of an open-vocabulary tag (`kind`, `landing`, `volatility`,
/// `status`).
///
/// These are newtypes over `String` rather than enums so a new value costs no
/// contract re-key (see the note in `types.rs`). The bound is what stops an open
/// vocabulary from becoming an unbounded field: it is sized to fit the longest
/// value in use ("unreachable", 11) with room to spare, and it feeds
/// `max_entries_worst_case_fits_the_node_state_cap` -- four vocabulary fields at
/// this length are part of the worst-case record.
pub const MAX_VOCAB: usize = 16;

/// Bounds on `IndexEntry::ext`, the opaque additive-metadata map.
///
/// The contract validates SIZE and character-safety only, and never interprets
/// a key. That is the entire point: a new piece of per-entry metadata can be
/// added by writing a new key here, with no Rust type change, therefore no new
/// contract artifact, therefore no re-key and no migration. Every `common/`
/// edit moves the index address (`hash(compiled_wasm, params)`), and both prior
/// re-keys left the live UI serving a stale snapshot, once for eight days --
/// so an additive change that does not require one is worth real money.
///
/// The trade is that the contract cannot enforce anything ABOUT this metadata.
/// Anything the contract must validate, or that the UI must be able to rely on,
/// belongs in a typed field instead. `ext` is for metadata the network only
/// needs to carry.
///
/// SIZED BY MEASUREMENT, NOT BY TASTE, and the budget is now FULLY ALLOCATED.
/// `max_entries_worst_case_fits_the_node_state_cap` prints the arithmetic: the
/// worst legal record is 2609 B, so `MAX_ENTRIES` of them is 49.76 MiB of the
/// 50 MiB cap, leaving about 12 B per entry spare. A first attempt at 16 keys x
/// (32 B key + 256 B value) measured 7151 B per entry, i.e. 136 MiB against
/// that cap.
///
/// Hence a TOTAL-bytes bound rather than per-key maxima. Bounding a count and a
/// per-item size independently bounds their PRODUCT, which is what exploded;
/// bounding the sum bounds the thing that actually has to fit, and lets the
/// budget be spent as one long value or several short ones.
///
/// READ THIS BEFORE ADDING A FIELD. Twelve bytes of slack is not an oversight
/// and does not want "fixing": this 96 B IS the pre-allocated growth space, and
/// future per-entry metadata is meant to land inside it rather than as a new
/// typed field. Growing here costs nothing, since no Rust type changes and so
/// the contract does not re-key. Adding a new typed field, or raising any bound
/// including this one, now requires trading it against another bound or against
/// `MAX_ENTRIES` — and re-running that test, which is the only thing that will
/// tell you whether you can afford it.
pub const MAX_EXT_KEYS: usize = 8;
/// Max length of a single `ext` key. A sanity bound; [`MAX_EXT_TOTAL_BYTES`] is
/// the one that binds.
pub const MAX_EXT_KEY: usize = 32;
/// Max SUM of `key.len() + value.len()` across every `ext` pair.
pub const MAX_EXT_TOTAL_BYTES: usize = 96;

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

#[cfg(test)]
mod bound_tests {
    use super::*;
    use crate::types::{
        Audience, Classification, IndexEntry, Kind, Locator, RecordBody, SignedRecord, SubjectId,
        Verification, VerifyStatus, Volatility,
    };
    use ed25519_dalek::SigningKey;

    /// The node hard-caps contract state at 50 MiB, and `MAX_ENTRIES` is only a
    /// meaningful bound if a maximally-full index of maximally-large VALID entries
    /// still fits under it. It did not: `MAX_EXTERNAL_URL` at 1024 put the worst
    /// case at 51.5 MiB, so an index could become unstorable while every
    /// individual entry was legal.
    ///
    /// This is the invariant that ties the per-entry bounds to the cap. Raising any
    /// of them without re-checking is what this catches.
    #[test]
    fn max_entries_worst_case_fits_the_node_state_cap() {
        const NODE_STATE_CAP: usize = 50 * 1024 * 1024;
        let key = SigningKey::from_bytes(&[7u8; 32]);
        let id = "EqJ5YpEEV3XLqEvKWLQHFhGAac2qXzSUoE6k2zbdnXBr";

        let worst = |loc: Locator| -> usize {
            let e = IndexEntry {
                subject_id: SubjectId::parse(&bs58::encode([9u8; 9]).into_string()).unwrap(),
                version: u64::MAX,
                // Vocabulary fields at their BOUND, not at the length of the
                // values in use today. `Kind`/`landing`/`volatility`/`status`
                // are open vocabularies, so a future value may be any legal tag
                // up to MAX_VOCAB; measuring "Site" here would make this test
                // pass while a legal record still blew the cap.
                kind: Kind::new("k".repeat(MAX_VOCAB)),
                title: "t".repeat(MAX_TITLE),
                snippet: "s".repeat(MAX_SNIPPET),
                tags: (0..MAX_TAGS).map(|_| "g".repeat(MAX_TAG_LEN)).collect(),
                locator: loc,
                featured: true,
                added_at: u64::MAX,
                // The optional fields are part of the worst case and must be
                // POPULATED here. They are `skip_serializing_if`, so leaving
                // them `None` would measure a smaller record than the contract
                // will accept and quietly restore the exact "finite bound that
                // is not sufficient" hole this test exists to close.
                class: Some(Classification {
                    landing: Audience::new("a".repeat(MAX_VOCAB)),
                    has_adult_sections: true,
                    volatility: Volatility::new("v".repeat(MAX_VOCAB)),
                    classifier: u16::MAX,
                    classified_at: u64::MAX,
                }),
                verified: Some(Verification {
                    last_verified_at: u64::MAX,
                    status: VerifyStatus::new("s".repeat(MAX_VOCAB)),
                }),
                // The largest LEGAL ext: the full key count, with the whole
                // byte budget spent. Split evenly so both bounds bind at once.
                ext: Some(
                    (0..MAX_EXT_KEYS)
                        .map(|i| {
                            let per = MAX_EXT_TOTAL_BYTES / MAX_EXT_KEYS;
                            (format!("{i:0>2}"), "v".repeat(per - 2))
                        })
                        .collect(),
                ),
            };
            let body = RecordBody::Live(e);
            // Must be VALID, or the bound is not actually reachable and the test
            // would be measuring something the contract would reject anyway.
            body.check_structure()
                .expect("the worst case must be a valid entry");
            let rec = SignedRecord {
                sig: sign(&body, &key),
                by: key.verifying_key(),
                body,
            };
            let mut buf = Vec::new();
            ciborium::ser::into_writer(&rec, &mut buf).unwrap();
            buf.len()
        };

        let cases = [
            Locator::External {
                url: format!("https://e.example/{}", "x".repeat(MAX_EXTERNAL_URL - 20)),
            },
            Locator::AppResource {
                app: "a".repeat(MAX_APP_SLUG),
                resource: "b".repeat(MAX_RESOURCE),
                path: "/p".repeat(MAX_LOCATOR_PATH / 2 - 1),
            },
            Locator::Freenet {
                contract_id: id.to_string(),
                path: "/p".repeat(MAX_LOCATOR_PATH / 2 - 1),
            },
        ];
        let worst_record = cases.into_iter().map(worst).max().unwrap();
        let total = worst_record * MAX_ENTRIES;
        // Report the headroom, not just pass/fail. Sizing a new per-entry field
        // needs a NUMBER, and a test that only speaks up on overflow forces the
        // next person to guess and then discover the overflow — which is how the
        // first version of `ext` came to be 48x its budget. Run with
        // `--nocapture` to read it.
        let headroom = NODE_STATE_CAP.saturating_sub(total);
        println!(
            "worst-case record {worst_record} B x MAX_ENTRIES {MAX_ENTRIES} = {:.2} MiB \
             of the {} MiB cap; headroom {:.2} MiB = {} B per entry",
            total as f64 / 1048576.0,
            NODE_STATE_CAP / 1024 / 1024,
            headroom as f64 / 1048576.0,
            headroom / MAX_ENTRIES,
        );
        assert!(
            total < NODE_STATE_CAP,
            "MAX_ENTRIES ({MAX_ENTRIES}) x worst-case record ({worst_record} B) = \
             {:.1} MiB, over the {} MiB node cap — lower a per-entry bound or \
             MAX_ENTRIES",
            total as f64 / 1048576.0,
            NODE_STATE_CAP / 1024 / 1024
        );
    }
}
