//! Deriving a room's contract id from its owner key and a room-contract code
//! hash, without any room-WASM bytes.
//!
//! A room's contract id is `BLAKE3(code_hash || CBOR(ChatRoomParametersV1 { owner }))`
//! (`freenet_migrate::contract_id_from_code_hash`). The code hash is NOT stored
//! here or in the index: it moves with every room-contract release, and the
//! current value is whatever River's signed pointer record (`river.room-contract`)
//! names. `atlasctl` resolves that pointer at run time and calls this.

use ed25519_dalek::VerifyingKey;
use serde::Serialize;

/// River's `ChatRoomParametersV1`, mirrored: `CBOR({owner})`, the contract
/// parameters. Must stay byte-identical to River's encoding, which the `golden`
/// test pins against real `river-core` output.
#[derive(Serialize)]
struct ChatRoomParameters<'a> {
    owner: &'a VerifyingKey,
}

/// The CBOR contract parameters for `owner`'s room.
pub fn room_params(owner: &VerifyingKey) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(&ChatRoomParameters { owner }, &mut buf)
        .expect("CBOR encoding of the room parameters cannot fail");
    buf
}

/// The 32-byte contract instance id of `owner`'s room under `code_hash`.
pub fn room_instance_id(owner: &VerifyingKey, code_hash: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(code_hash);
    h.update(&room_params(owner));
    *h.finalize().as_bytes()
}
