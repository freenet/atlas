//! The Atlas index contract.
//!
//! State is an [`IndexState`]: a root-signed authorization of online signing
//! keys plus one current record per subject. The contract's whole job is to make
//! the state self-verifying on untrusted peers and to merge updates as a
//! commutative monoid. Ranking, search, and display are client-side; this
//! contract enforces only signatures, authorization, structure, bounds, and the
//! versioned merge.

use atlas_common::{IndexDelta, IndexParams, IndexState, IndexSummary};
use ciborium::{de::from_reader, ser::into_writer};
use freenet_stdlib::prelude::ContractError;
use freenet_stdlib::prelude::*;

// The struct and helper are only reached through the FFI exports the
// `#[contract]` macro generates, so they read as dead code in a plain build.
#[allow(dead_code)]
struct AtlasIndex;

#[allow(dead_code)]
fn parse_params(parameters: &Parameters<'static>) -> Result<IndexParams, ContractError> {
    IndexParams::from_bytes(parameters.as_ref())
        .ok_or_else(|| ContractError::Deser("invalid index parameters".to_string()))
}

#[contract]
impl ContractInterface for AtlasIndex {
    fn validate_state(
        parameters: Parameters<'static>,
        state: State<'static>,
        _related: RelatedContracts<'static>,
    ) -> Result<ValidateResult, ContractError> {
        let bytes = state.as_ref();
        // An empty state is a valid not-yet-initialized index.
        if bytes.is_empty() {
            return Ok(ValidateResult::Valid);
        }
        let params = parse_params(&parameters)?;
        let st = from_reader::<IndexState, &[u8]>(bytes)
            .map_err(|e| ContractError::Deser(e.to_string()))?;
        st.verify(&params)
            .map(|_| ValidateResult::Valid)
            .map_err(|reason| ContractError::InvalidUpdateWithInfo { reason })
    }

    fn update_state(
        parameters: Parameters<'static>,
        state: State<'static>,
        data: Vec<UpdateData<'static>>,
    ) -> Result<UpdateModification<'static>, ContractError> {
        let params = parse_params(&parameters)?;
        let mut st: IndexState = if state.as_ref().is_empty() {
            IndexState::default()
        } else {
            from_reader::<IndexState, &[u8]>(state.as_ref())
                .map_err(|e| ContractError::Deser(e.to_string()))?
        };

        for update in data {
            match update {
                UpdateData::State(new_state) => {
                    let incoming = from_reader::<IndexState, &[u8]>(new_state.as_ref())
                        .map_err(|e| ContractError::Deser(e.to_string()))?;
                    // A full state from a peer is untrusted: verify before merging.
                    incoming
                        .verify(&params)
                        .map_err(|reason| ContractError::InvalidUpdateWithInfo { reason })?;
                    st.merge(&incoming);
                }
                UpdateData::Delta(d) => {
                    if d.as_ref().is_empty() {
                        continue;
                    }
                    let delta = from_reader::<IndexDelta, &[u8]>(d.as_ref())
                        .map_err(|e| ContractError::Deser(e.to_string()))?;
                    st.apply_delta(&delta, &params)
                        .map_err(|reason| ContractError::InvalidUpdateWithInfo { reason })?;
                }
                UpdateData::StateAndDelta { state: s, delta: d } => {
                    let incoming = from_reader::<IndexState, &[u8]>(s.as_ref())
                        .map_err(|e| ContractError::Deser(e.to_string()))?;
                    incoming
                        .verify(&params)
                        .map_err(|reason| ContractError::InvalidUpdateWithInfo { reason })?;
                    st.merge(&incoming);
                    if !d.as_ref().is_empty() {
                        let delta = from_reader::<IndexDelta, &[u8]>(d.as_ref())
                            .map_err(|e| ContractError::Deser(e.to_string()))?;
                        st.apply_delta(&delta, &params)
                            .map_err(|reason| ContractError::InvalidUpdateWithInfo { reason })?;
                    }
                }
                // `UpdateData` is `#[non_exhaustive]`; reject shapes we don't
                // handle rather than panicking inside the WASM runtime.
                _ => return Err(ContractError::InvalidUpdate),
            }
        }

        // Defense in depth: the merged result must satisfy the same invariant as
        // validate_state (signatures, authorization against the final key_auth,
        // bounds). The merge logic already guarantees this; re-checking ensures
        // update_state can never emit a state its own validate_state rejects.
        st.verify(&params)
            .map_err(|reason| ContractError::InvalidUpdateWithInfo { reason })?;

        let mut out = Vec::new();
        into_writer(&st, &mut out).map_err(|e| ContractError::Deser(e.to_string()))?;
        Ok(UpdateModification::valid(out.into()))
    }

    fn summarize_state(
        _parameters: Parameters<'static>,
        state: State<'static>,
    ) -> Result<StateSummary<'static>, ContractError> {
        if state.as_ref().is_empty() {
            return Ok(StateSummary::from(Vec::new()));
        }
        let st = from_reader::<IndexState, &[u8]>(state.as_ref())
            .map_err(|e| ContractError::Deser(e.to_string()))?;
        let mut out = Vec::new();
        into_writer(&st.summarize(), &mut out).map_err(|e| ContractError::Deser(e.to_string()))?;
        Ok(StateSummary::from(out))
    }

    fn get_state_delta(
        _parameters: Parameters<'static>,
        state: State<'static>,
        summary: StateSummary<'static>,
    ) -> Result<StateDelta<'static>, ContractError> {
        let st = from_reader::<IndexState, &[u8]>(state.as_ref())
            .map_err(|e| ContractError::Deser(e.to_string()))?;
        let summ = if summary.as_ref().is_empty() {
            IndexSummary::default()
        } else {
            from_reader::<IndexSummary, &[u8]>(summary.as_ref())
                .map_err(|e| ContractError::Deser(e.to_string()))?
        };
        let delta = st.delta(&summ);
        // Emit ZERO BYTES when there is nothing to send. Serialising an all-empty
        // `IndexDelta` yields a 3-key CBOR map (~26 bytes), and freenet-core judges
        // convergence by `!delta.as_ref().is_empty()` (`ring/interest.rs`), so a
        // non-empty encoding of "no changes" means the node can NEVER conclude that
        // two peers have converged: every byte-differing summary pair escalates to a
        // staleness probe that always answers "send", on every fan-out and every
        // anti-entropy heartbeat, indefinitely.
        //
        // Symmetric with `update_state`, which already skips an empty delta.
        if delta.is_logically_empty() {
            return Ok(StateDelta::from(Vec::new()));
        }
        let mut out = Vec::new();
        into_writer(&delta, &mut out).map_err(|e| ContractError::Deser(e.to_string()))?;
        Ok(StateDelta::from(out))
    }
}
