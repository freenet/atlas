//! Indexing public River rooms in Atlas (atlas#83, river#972).
//!
//! Room indexing uses only what the index schema already has: kind `"Room"`,
//! `Locator::AppResource { app: "river", resource: <owner key>, path: "" }`, and
//! `IndexEntry::ext`. Nothing here is linked into the index contract and nothing
//! under `common/` changed, so the index address does not move.
//!
//! - [`eligibility`]: the gate. Public AND listed (or explicitly submitted until
//!   river#974 adds `PublicRoomPolicy.listed`), owner signature verified.
//! - [`entry`]: the descriptor schema (see its module docs) and entry refresh.
//! - [`reconcile`]: duplicates, renames, stale reads, delisting.
//! - [`search`]: the read API for River's Discover Rooms.
//! - [`text`]: sanitising owner text.
//!
//! # Identity
//!
//! The locator carries the FULL owner verifying key in base58. It survives
//! room-WASM re-keys because River derives the contract from the owner key and
//! the CURRENT room-contract code hash, which it resolves from its own published
//! pointer record (`river.room-contract`). Atlas never stores a contract id for a
//! room and never hardcodes a room-WASM hash. Short handles (#973) are display
//! only.

pub mod contract;
pub mod eligibility;
pub mod entry;
pub mod reconcile;
pub mod search;
pub mod text;
pub mod view;
