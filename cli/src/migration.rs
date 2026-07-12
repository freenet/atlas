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
/// entries at the retired key are orphaned. `migration_registry_matches_legacy_wasm`
/// pins that the newest entry equals the preserved legacy WASM's code hash.
///
/// - `GDt9A4DteAP6SYPmFXzoTScQuPfwufMaoZxxJaDDB1Yt`
///   freenet-stdlib 0.6.0 generation, retired by the 0.8.3 bump. Its WASM is
///   preserved at `contracts/index-contract/legacy/`.
pub const LEGACY_INDEX_CODE_HASHES: &[&str] = &["GDt9A4DteAP6SYPmFXzoTScQuPfwufMaoZxxJaDDB1Yt"];

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

    // The retired 0.6.0 WASM, preserved so the registered code hash can be
    // verified against the real bytes it claims to describe.
    const LEGACY_V06_WASM: &[u8] = include_bytes!(
        "../../contracts/index-contract/legacy/atlas_index_contract-v0.6.0-stdlib.wasm"
    );
    // The current (0.8.3) WASM the CLI embeds.
    const CURRENT_WASM: &[u8] =
        include_bytes!("../../contracts/index-contract/atlas_index_contract.wasm");

    // A fixed, non-secret params blob so key derivation is deterministic in the
    // test. Layout mirrors `IndexParams::to_bytes`: 32-byte root vk || slug.
    fn test_params() -> Vec<u8> {
        let mut p = vec![7u8; 32];
        p.extend_from_slice(b"default");
        p
    }

    /// The registered legacy code hash must be exactly the retired WASM's hash,
    /// so the derived legacy key equals the address the live 0.6 index sits at.
    /// Guards against a wrong/typo'd constant silently pointing the migration at
    /// a non-existent contract.
    #[test]
    fn migration_registry_matches_legacy_wasm() {
        let params = test_params();
        let from_registry = &legacy_index_keys(&params)[0];
        let from_wasm = ContractKey::from_params_and_code(
            Parameters::from(params.clone()),
            ContractCode::from(LEGACY_V06_WASM.to_vec()),
        );
        assert_eq!(
            from_registry.id(),
            from_wasm.id(),
            "registered legacy code hash does not reproduce the 0.6 WASM's key"
        );
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
