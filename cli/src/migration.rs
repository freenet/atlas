//! Backward-probe migration for the Atlas index contract.
//!
//! The index's on-network address is `hash(index_wasm, params)`. Any dependency
//! bump that changes the compiled WASM (e.g. the freenet-stdlib 0.6 -> 0.8 bump)
//! re-keys the index, so its curated entries would be stranded at the *old*
//! address unless carried forward. Atlas has no protocol-level migration, so we
//! do what River and Delta do: keep a registry of every *previous* generation's
//! code hash, re-derive the old key from the same (unchanged) params, GET the
//! old state, and PUT it into the new key.
//!
//! The params (`root_vk || slug`) are a fixed byte layout that never changes on
//! a dependency bump (see `atlas_common::IndexParams`), so the ONLY thing that
//! moves on a rebuild is the code hash — which is exactly what this registry
//! pins.

use freenet_stdlib::prelude::{ContractKey, Parameters};

/// Code hashes (base58, BITCOIN alphabet — i.e. `CodeHash::encode()`) of every
/// PRIOR index-contract WASM generation, newest-first. The CURRENT generation
/// is deliberately absent: its key is derived from the embedded WASM at runtime.
///
/// Prepend the OUTGOING code hash here every time the committed WASM is rebuilt
/// in a way that changes its hash, BEFORE shipping the rebuild — otherwise the
/// entries at the retired key are orphaned, AND preserve that generation's WASM
/// under `contracts/index-contract/legacy/`.
/// `every_registered_hash_matches_its_preserved_wasm` pins both halves.
///
/// - `C6vpLoy2sdzbw9crd9wiAtUJQdeofN4NrANbqAGcLGTU`
///   freenet-stdlib 0.8.3 generation, retired by the app-registry change (which
///   touched `common/`, so the contract re-keyed).
///
///   **This generation IS published and holds the NEWEST state.** Probed
///   2026-07-29 it had 36 records / 29 live entries, MORE than the 0.6.0 address
///   (30 / 23), so skipping it in a migration loses more than skipping 0.6.0
///   would. The 0.8.3 migration ran and the UI was rebuilt against it.
///
///   (This comment has been wrong twice, in opposite directions, which is the
///   useful part. First it asserted this generation was never published — wrong,
///   and dangerous, because registering a generation on the belief that it holds
///   nothing and then reading a NotFound as confirmation is how you lose exactly
///   the entries you were migrating. Then the correction over-reached and claimed
///   the published UI had been left reading 0.6.0; that was inferred from a stale
///   default in the UI source without checking the deployed bytes, and the bytes
///   say otherwise. Probe before you conclude, in both directions.)
/// - `GDt9A4DteAP6SYPmFXzoTScQuPfwufMaoZxxJaDDB1Yt`
///   freenet-stdlib 0.6.0 generation, retired by the 0.8.3 bump. Still holds the
///   pre-bump snapshot, so it is worth merging, but it is NOT what the published UI
///   reads.
pub const LEGACY_INDEX_CODE_HASHES: &[&str] = &[
    // Retired 2026-08-03 by the freenet-stdlib 0.8.3 -> 0.8.5 bump, which the
    // crawler's river-core 0.1.19 requirement forced. The contract's own source
    // did not change; stdlib is a workspace dependency of it, so the artifact
    // (and therefore the index address) moved anyway.
    "4oYXG4CMegsqQ2Hn1vXfkcnCpTnMTfeyTUTWgTabbBAA",
    "C6vpLoy2sdzbw9crd9wiAtUJQdeofN4NrANbqAGcLGTU",
    "GDt9A4DteAP6SYPmFXzoTScQuPfwufMaoZxxJaDDB1Yt",
];

/// Re-derive the legacy index keys (newest-first) for the given params bytes.
/// A malformed constant is skipped rather than aborting the whole probe.
pub fn legacy_index_keys(params: &[u8]) -> Vec<ContractKey> {
    LEGACY_INDEX_CODE_HASHES
        .iter()
        .filter_map(|h| ContractKey::from_params(*h, Parameters::from(params.to_vec())).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use freenet_stdlib::prelude::{ContractCode, ContractInstanceId};

    // The retired WASMs, preserved so each registered code hash can be verified
    // against the real bytes it claims to describe. Newest-first, matching the
    // order of `LEGACY_INDEX_CODE_HASHES`.
    // Named `-app-registry` rather than plain `-v0.8.3-stdlib`: BOTH this
    // generation and the one below were built against stdlib 0.8.3 (this one was
    // retired by the 0.8.5 bump, the other by the app-registry source change),
    // so the stdlib version alone no longer distinguishes them.
    const LEGACY_V083_APP_REGISTRY_WASM: &[u8] = include_bytes!(
        "../../contracts/index-contract/legacy/atlas_index_contract-v0.8.3-stdlib-app-registry.wasm"
    );
    const LEGACY_V083_WASM: &[u8] = include_bytes!(
        "../../contracts/index-contract/legacy/atlas_index_contract-v0.8.3-stdlib.wasm"
    );
    const LEGACY_V06_WASM: &[u8] = include_bytes!(
        "../../contracts/index-contract/legacy/atlas_index_contract-v0.6.0-stdlib.wasm"
    );
    const PRESERVED_LEGACY_WASMS: &[&[u8]] = &[
        LEGACY_V083_APP_REGISTRY_WASM,
        LEGACY_V083_WASM,
        LEGACY_V06_WASM,
    ];
    // The current WASM the CLI embeds.
    const CURRENT_WASM: &[u8] =
        include_bytes!("../../contracts/index-contract/atlas_index_contract.wasm");

    // A fixed, non-secret params blob so key derivation is deterministic in the
    // test. Layout mirrors `IndexParams::to_bytes`: 32-byte root vk || slug.
    fn test_params() -> Vec<u8> {
        let mut p = vec![7u8; 32];
        p.extend_from_slice(b"default");
        p
    }

    /// EVERY registered legacy code hash must be exactly its retired WASM's
    /// hash, so each derived legacy key equals a real prior address. Guards
    /// against a wrong/typo'd constant silently pointing the migration at a
    /// non-existent contract, and against a new generation being registered
    /// without preserving the WASM that backs the claim.
    #[test]
    fn every_registered_hash_matches_its_preserved_wasm() {
        let params = test_params();
        let derived = legacy_index_keys(&params);
        assert_eq!(
            derived.len(),
            LEGACY_INDEX_CODE_HASHES.len(),
            "a registered hash failed to parse into a key"
        );
        assert_eq!(
            PRESERVED_LEGACY_WASMS.len(),
            LEGACY_INDEX_CODE_HASHES.len(),
            "every registered generation must have its WASM preserved under \
             contracts/index-contract/legacy/ (newest-first, same order)"
        );
        for (i, (key, wasm)) in derived.iter().zip(PRESERVED_LEGACY_WASMS).enumerate() {
            let from_wasm = ContractKey::from_params_and_code(
                Parameters::from(params.clone()),
                ContractCode::from(wasm.to_vec()),
            );
            assert_eq!(
                key.id(),
                from_wasm.id(),
                "LEGACY_INDEX_CODE_HASHES[{i}] does not reproduce its preserved WASM's key"
            );
        }
    }

    /// The registered generations must be DISTINCT. Registering the same hash
    /// twice, or pasting the wrong one, would silently shrink the probe set the
    /// migration walks.
    #[test]
    fn registered_generations_are_distinct() {
        let params = test_params();
        let mut ids: Vec<String> = legacy_index_keys(&params)
            .iter()
            .map(|k| k.id().to_string())
            .collect();
        let before = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate legacy generation registered");
    }

    /// The 0.8 rebuild must actually re-key: the current key must differ from
    /// the legacy key (otherwise there is nothing to migrate and the registry
    /// entry is stale / wrong).
    #[test]
    fn current_key_differs_from_legacy() {
        let params = test_params();
        let legacy = legacy_index_keys(&params)[0].id().to_owned();
        let current = ContractKey::from_params_and_code(
            Parameters::from(params.clone()),
            ContractCode::from(CURRENT_WASM.to_vec()),
        )
        .id()
        .to_owned();
        assert_ne!(
            legacy, current,
            "current WASM hashes to the legacy key — the rebuild did not re-key"
        );
    }

    /// A stored base58 code hash reproduces the same instance id as hashing the
    /// full WASM, so registering just the 32-byte hash (not the whole WASM) is
    /// sufficient. This is the invariant the whole registry rests on.
    #[test]
    fn from_params_matches_from_code_for_current() {
        let params = test_params();
        let via_code = ContractKey::from_params_and_code(
            Parameters::from(params.clone()),
            ContractCode::from(CURRENT_WASM.to_vec()),
        );
        let code_hash_b58 = via_code.encoded_code_hash();
        let via_hash =
            ContractKey::from_params(code_hash_b58, Parameters::from(params.clone())).unwrap();
        assert_eq!(via_code.id(), via_hash.id());
        // sanity: ids parse/format round-trip
        let s = via_code.id().to_string();
        assert_eq!(s.parse::<ContractInstanceId>().unwrap(), *via_code.id());
    }
}
