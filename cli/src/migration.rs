//! Backward-probe migration registry for the Atlas index contract, on the
//! shared `freenet-migrate` machinery.
//!
//! The index's on-network address is `hash(index_wasm, params)`. Any dependency
//! bump that changes the compiled WASM (e.g. the freenet-stdlib 0.6 -> 0.8 bump)
//! re-keys the index, so its curated entries would be stranded at the *old*
//! address unless carried forward. Atlas has no protocol-level migration, so we
//! do what River and Delta do — keep a registry of every *previous* generation's
//! code hash, re-derive each old key from the same (unchanged) params, GET the
//! old state, and merge it into the new key — except the registry shape, key
//! derivation, and probe sequencing now come from `freenet-migrate` instead of
//! being hand-rolled here.
//!
//! The registry itself lives in `cli/legacy_contracts.toml` (freenet-migrate's
//! unified `[[contract]]` schema, with the how-to-retire-a-generation
//! instructions at the top); `build.rs` decodes and validates every hash AT
//! BUILD TIME via `freenet-migrate-build`, so a malformed hash is a build
//! failure — the retired hand-rolled `&[&str]` registry silently skipped one at
//! runtime instead.
//!
//! The params (`root_vk || slug`) are a fixed byte layout that never changes on
//! a dependency bump (see `atlas_common::IndexParams`), so the ONLY thing that
//! moves on a rebuild is the code hash — which is exactly what the registry
//! pins.

use freenet_migrate::{contract_id_from_code_hash, ContractLineageEntry};
use freenet_stdlib::prelude::{ContractInstanceId, Parameters};

mod generated {
    //! Codegen output from `cli/legacy_contracts.toml`. The codegen emits an
    //! (empty) `DELEGATE_LINEAGE` alongside; Atlas has no delegate, so it is
    //! unused by design.
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/lineage.rs"));
}

/// Every PRIOR index-contract generation, slice order newest-first (pinned by
/// `slice_order_is_generation_descending` to agree with the probe driver's
/// generation-descending order). The CURRENT generation is deliberately
/// absent: its key is derived from the embedded WASM at runtime.
pub const CONTRACT_LINEAGE: &[ContractLineageEntry] = generated::CONTRACT_LINEAGE;

/// Re-derive the legacy index ids (newest-first) for the given params bytes.
/// Infallible: the hashes were decoded and validated at build time.
pub fn legacy_index_keys(params: &[u8]) -> Vec<ContractInstanceId> {
    let params = Parameters::from(params.to_vec());
    CONTRACT_LINEAGE
        .iter()
        .map(|e| contract_id_from_code_hash(&e.code_hash, &params))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use freenet_migrate::{contract_probe, FoldAllAck, ProbeStateOps, SelectionPolicy, Step};
    use freenet_stdlib::prelude::{ContractCode, ContractInstanceId, ContractKey, Parameters};

    // The retired WASMs, preserved so each registered code hash can be verified
    // against the real bytes it claims to describe. Newest-first, matching the
    // registry's slice order.
    // Named `-app-registry` rather than plain `-v0.8.3-stdlib`: BOTH this
    // generation and the one below were built against stdlib 0.8.3 (this one was
    // retired by the 0.8.5 bump, the other by the app-registry source change),
    // so the stdlib version alone no longer distinguishes them.
    const LEGACY_V085_PRE_CLASSIFICATION_WASM: &[u8] = include_bytes!(
        "../../contracts/index-contract/legacy/atlas_index_contract-v0.8.5-stdlib-pre-classification.wasm"
    );
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
        LEGACY_V085_PRE_CLASSIFICATION_WASM,
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

    /// The 4 recorded generations, newest-first, exactly as the retired
    /// hand-rolled registry held them. Every one is (or may be) a published
    /// address that still holds state: losing, reordering, or editing an entry
    /// in `legacy_contracts.toml` silently strands whoever's state lives there,
    /// so the expected list is spelled out literally rather than derived.
    #[test]
    fn the_four_generations_survive_in_order() {
        let expected = [
            "EMp9LZM2awHm7QeG6CsKGDNVfmgHehBTSht6RE2uDvhD",
            "4oYXG4CMegsqQ2Hn1vXfkcnCpTnMTfeyTUTWgTabbBAA",
            "C6vpLoy2sdzbw9crd9wiAtUJQdeofN4NrANbqAGcLGTU",
            "GDt9A4DteAP6SYPmFXzoTScQuPfwufMaoZxxJaDDB1Yt",
        ];
        let got: Vec<String> = CONTRACT_LINEAGE.iter().map(|e| e.code_hash_b58()).collect();
        assert_eq!(got, expected, "the registry lost or reordered a generation");
    }

    /// Slice order must equal generation-DESCENDING order: `atlasctl keys`
    /// prints slice order, while the probe driver orders by the `generation`
    /// field — this is what keeps the two identical (newest-first).
    #[test]
    fn slice_order_is_generation_descending() {
        let generations: Vec<u32> = CONTRACT_LINEAGE.iter().map(|e| e.generation).collect();
        assert!(
            generations.windows(2).all(|w| w[0] > w[1]),
            "registry slice order is not strictly generation-descending: {generations:?}"
        );
    }

    /// The driver probes exactly the registered generations, newest-first —
    /// the same ids in the same order `legacy_index_keys` reports. Pinned by
    /// pumping a real `contract_probe` driver (the entry point `migrate`'s
    /// drive goes through), not by re-deriving the order in the test.
    #[test]
    fn probe_order_matches_keys_order_newest_first() {
        struct NullOps;
        impl ProbeStateOps for NullOps {
            type State = ();
            fn decode(&self, _bytes: &[u8]) -> Option<()> {
                None
            }
            fn is_real(&self, _state: &()) -> bool {
                false
            }
            fn merge_with_local(&self, _recovered: (), _local: &()) {}
        }
        let params_bytes = test_params();
        let params = Parameters::from(params_bytes.clone());
        let mut driver = contract_probe(
            NullOps,
            (),
            &params,
            CONTRACT_LINEAGE,
            SelectionPolicy::FoldAll(
                FoldAllAck::i_understand_fold_all_resurrects_without_tombstones(),
            ),
        );
        let mut probed = Vec::new();
        while let Step::Get(id) = driver.next_action() {
            probed.push(id);
            driver.on_unknown(id);
        }
        assert_eq!(
            probed,
            legacy_index_keys(&params_bytes),
            "probe order must be the newest-first order `atlasctl keys` reports"
        );
    }

    /// EVERY registered legacy code hash must be exactly its retired WASM's
    /// hash, so each derived legacy key equals a real prior address. Guards
    /// against a wrong/typo'd registry entry silently pointing the migration at
    /// a non-existent contract, and against a new generation being registered
    /// without preserving the WASM that backs the claim. (Because the derived
    /// side comes from `freenet-migrate`'s key reconstruction and the expected
    /// side from stdlib's own `from_params_and_code`, this also cross-checks
    /// the crate's derivation against stdlib on every build.)
    #[test]
    fn every_registered_hash_matches_its_preserved_wasm() {
        let params = test_params();
        let derived = legacy_index_keys(&params);
        assert_eq!(
            PRESERVED_LEGACY_WASMS.len(),
            CONTRACT_LINEAGE.len(),
            "every registered generation must have its WASM preserved under \
             contracts/index-contract/legacy/ (newest-first, same order)"
        );
        for (i, (id, wasm)) in derived.iter().zip(PRESERVED_LEGACY_WASMS).enumerate() {
            let from_wasm = ContractKey::from_params_and_code(
                Parameters::from(params.clone()),
                ContractCode::from(wasm.to_vec()),
            );
            assert_eq!(
                id,
                from_wasm.id(),
                "CONTRACT_LINEAGE[{i}] does not reproduce its preserved WASM's key"
            );
        }
    }

    /// The registered generations must be DISTINCT. Registering the same hash
    /// twice, or pasting the wrong one, would silently shrink the probe set the
    /// migration walks.
    #[test]
    fn registered_generations_are_distinct() {
        let mut ids: Vec<String> = legacy_index_keys(&test_params())
            .iter()
            .map(|k| k.to_string())
            .collect();
        let before = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate legacy generation registered");
    }

    /// The newest registered generation must actually be RETIRED: the current
    /// key must differ from it (otherwise there is nothing to migrate and the
    /// registry entry is stale / wrong).
    #[test]
    fn current_key_differs_from_legacy() {
        let params = test_params();
        let legacy = legacy_index_keys(&params)[0];
        let current = *ContractKey::from_params_and_code(
            Parameters::from(params.clone()),
            ContractCode::from(CURRENT_WASM.to_vec()),
        )
        .id();
        assert_ne!(
            legacy, current,
            "current WASM hashes to the legacy key — the rebuild did not re-key"
        );
    }

    /// A stored 32-byte code hash reproduces the same instance id as hashing
    /// the full WASM, so registering just the hash (not the whole WASM) is
    /// sufficient. This is the invariant the whole registry rests on — checked
    /// both through stdlib's own `from_params` and through `freenet-migrate`'s
    /// reconstruction, so the two derivations are pinned to agree here too.
    #[test]
    fn from_params_matches_from_code_for_current() {
        let params = test_params();
        let via_code = ContractKey::from_params_and_code(
            Parameters::from(params.clone()),
            ContractCode::from(CURRENT_WASM.to_vec()),
        );
        let code_hash_b58 = via_code.encoded_code_hash();
        let via_hash =
            ContractKey::from_params(code_hash_b58.clone(), Parameters::from(params.clone()))
                .unwrap();
        assert_eq!(via_code.id(), via_hash.id());
        let via_crate = freenet_migrate::contract_id_from_code_hash_b58(
            &code_hash_b58,
            &Parameters::from(params.clone()),
        )
        .unwrap();
        assert_eq!(via_code.id(), &via_crate);
        // sanity: ids parse/format round-trip
        let s = via_code.id().to_string();
        assert_eq!(s.parse::<ContractInstanceId>().unwrap(), *via_code.id());
    }
}
