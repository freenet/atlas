//! Read room messages from the `river-mirror` SQLite replica.
//!
//! This REPLACES deriving the room's contract key and GETting it ourselves.
//! That old path is why the Official room was read from a dead generation twice
//! (13 days, then 3 days): the key came from a bundled `room_contract.wasm`
//! that goes stale on every River re-key, and a stale bundle DEFINES what
//! "current" means, so no self-contained check could see it.
//!
//! The mirror owns room resolution now, and does it with an independently
//! maintained `riverctl` plus a generation attestation
//! (`mirror_state.generation_ok`) that is only set when the key it actually read
//! matches the separately-rotated pin. One component tracks re-keys instead of
//! four, and it is the one that already does it correctly.
//!
//! Removing this crate's `river-core` dependency is not incidental. `river-core
//! 0.1.19` requires `freenet-stdlib 0.8.5`, cargo unifies that across the
//! workspace, and the index contract depends on stdlib — so bumping river-core
//! to fix the crawler RE-KEYED the live Atlas index (2026-08-03). With
//! river-core gone, no crawler-side dependency change can move a contract
//! address again.
//!
//! **Trust boundary, stated rather than quietly dropped.** The old path verified
//! each message's ed25519 signature against its author's key. The threat that
//! guarded against was specifically A LYING LOCAL NODE — not invalid network
//! state — because `author` feeds the per-author spend share, so a forged author
//! is a rate-limit bucket anyone could mint. Naming it precisely matters,
//! because "the contract enforces signatures" answers a DIFFERENT question
//! (what other peers accept), not this one.
//!
//! Reading the mirror does not widen that exposure, for a reason worth being
//! explicit about: the old code was never independent of the local node anyway.
//! It took the room CONFIGURATION signature, the contract key resolution and the
//! entire WS transport from that same node; a node willing to forge a message
//! author could equally forge the member list the signature was checked against.
//! What changes is only WHERE the check lives — the mirror ingests through a
//! root-owned `riverctl` in the same security domain, reading the same node.
//!
//! So this is a deliberate transfer, not an oversight, and not a new hole. If
//! the local node ever stops being trusted, the fix belongs in the mirror (one
//! place) rather than in each consumer.

use anyhow::{Context, Result};
use rusqlite::{Connection, OpenFlags};
use std::path::Path;

/// One room message, as the crawler needs it.
pub struct MirroredMessage {
    /// Mirror-assigned, monotonic. The crawler's cursor and its earliest-poster
    /// attribution both key on this.
    pub seq: i64,
    pub author_id: String,
    pub content: String,
}

/// How stale the mirror's last successful reconcile may be before we refuse to
/// trust it. The mirror reconciles every 15 min by default, so this allows two
/// missed cycles plus slack.
const MAX_RECONCILE_AGE_SECS: i64 = 2400;

/// Why the mirror was refused. Kept as a message rather than an enum because
/// every caller does the same thing with it: log it loudly and skip the room.
pub struct MirrorUnusable(pub String);

/// Open the mirror read-only and check it is fit to read.
///
/// Fails CLOSED. A mirror that cannot attest its generation is exactly the state
/// that hid the dead-room reads, so "cannot tell" is treated as "do not trust",
/// never as "probably fine".
fn open_checked(db: &Path, room: &str) -> Result<Result<Connection, MirrorUnusable>> {
    // Read-only, and explicitly NOT creating: pointing this at a typo'd path
    // must be an error, not an empty database that silently yields no links.
    let conn = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("opening mirror at {}", db.display()))?;

    let version: i64 = conn
        .query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key='schema_version'",
            [],
            |r| r.get(0),
        )
        .context("mirror has no schema_version; is this a river-mirror database?")?;
    if version != 1 {
        return Ok(Err(MirrorUnusable(format!(
            "mirror schema version {version} is not 1; this build cannot read it"
        ))));
    }

    let row: Option<(i64, String)> = conn
        .query_row(
            "SELECT generation_ok, COALESCE(last_reconcile_ok_at,'') FROM mirror_state WHERE room=?1",
            [room],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok();
    let Some((generation_ok, last_ok)) = row else {
        return Ok(Err(MirrorUnusable(format!(
            "mirror has no state row for room {room}; it is not mirroring this room"
        ))));
    };
    if generation_ok != 1 {
        return Ok(Err(MirrorUnusable(
            "mirror reports generation_ok=0 -- it cannot attest that it read the \
             CURRENT room generation, so its contents may be from a superseded \
             contract"
                .into(),
        )));
    }
    let age = reconcile_age_secs(&last_ok);
    match age {
        // A FUTURE timestamp is not "very fresh", it is a corrupt or
        // clock-skewed one. Treating it as fresh would make a garbled value the
        // most trusted possible state.
        Some(a) if a < 0 => {
            return Ok(Err(MirrorUnusable(format!(
                "mirror's last_reconcile_ok_at is {}s in the FUTURE -- corrupt or \
                 clock-skewed",
                -a
            ))))
        }
        None => {
            return Ok(Err(MirrorUnusable(
                "mirror has never completed a reconcile".into(),
            )))
        }
        Some(a) if a > MAX_RECONCILE_AGE_SECS => {
            return Ok(Err(MirrorUnusable(format!(
                "mirror's last successful reconcile was {a}s ago (limit \
                 {MAX_RECONCILE_AGE_SECS}s) -- it may be wedged"
            ))))
        }
        Some(_) => {}
    }
    Ok(Ok(conn))
}

/// Seconds since an RFC3339 timestamp, or None if unparseable/absent.
///
/// Hand-rolled rather than pulling in `chrono`: the mirror writes
/// `to_rfc3339()`, and this crate has no other date needs.
fn reconcile_age_secs(ts: &str) -> Option<i64> {
    let ts = ts.trim();
    if ts.is_empty() {
        return None;
    }
    let epoch = rfc3339_to_epoch(ts)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs() as i64;
    Some(now - epoch)
}

/// Minimal RFC3339 -> unix seconds. Accepts the `+00:00` / `Z` forms the mirror
/// emits; anything else yields None, which the caller treats as unusable.
fn rfc3339_to_epoch(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let num = |a: usize, z: usize| -> Option<i64> { s.get(a..z)?.parse().ok() };
    let (y, mo, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    let (h, mi, sec) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    // Days from civil (Howard Hinnant's algorithm), valid for any Gregorian date.
    let y_adj = if mo <= 2 { y - 1 } else { y };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + h * 3600 + mi * 60 + sec)
}

/// Messages after `cursor`, oldest-first.
///
/// Ordered by `seq`, which is the mirror's own arrival order, so the FIRST
/// poster of a duplicate URL is seen first and is the one charged — the
/// earliest-poster property the spend accounting depends on. That property used
/// to come from sorting a whole room snapshot by the message's own `(time, id)`;
/// `seq` is strictly better for it, because a message's claimed time is
/// author-controlled and `seq` is not.
///
/// Deleted messages are excluded: a tombstoned message's links should not be
/// (re)captured.
pub fn messages_since(
    db: &Path,
    room: &str,
    cursor: i64,
    limit: usize,
) -> Result<Result<Vec<MirroredMessage>, MirrorUnusable>> {
    let conn = match open_checked(db, room)? {
        Ok(c) => c,
        Err(e) => return Ok(Err(e)),
    };
    let mut stmt = conn.prepare(
        "SELECT seq, author_id, content FROM messages
         WHERE room = ?1 AND seq > ?2 AND deleted = 0
         ORDER BY seq LIMIT ?3",
    )?;
    let rows = stmt
        .query_map(rusqlite::params![room, cursor, limit as i64], |r| {
            Ok(MirroredMessage {
                seq: r.get(0)?,
                author_id: r.get(1)?,
                content: r.get(2)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Ok(rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    /// Build a mirror-shaped database. Mirrors river-mirror's schema; if that
    /// schema changes incompatibly, `open_checked`'s schema_version guard is
    /// what stops this crawler reading it wrongly.
    fn fixture(
        generation_ok: i64,
        last_ok: &str,
        msgs: &[(i64, &str, &str, i64)],
    ) -> tempfile::NamedTempFile {
        let f = tempfile::NamedTempFile::new().unwrap();
        let c = Connection::open(f.path()).unwrap();
        c.execute_batch(
            "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO meta VALUES('schema_version','1');
             CREATE TABLE messages(seq INTEGER PRIMARY KEY, room TEXT, message_id TEXT,
                 author_id TEXT, content TEXT, deleted INTEGER DEFAULT 0);
             CREATE TABLE mirror_state(room TEXT PRIMARY KEY, last_reconcile_ok_at TEXT,
                 generation_ok INTEGER DEFAULT 0);",
        )
        .unwrap();
        c.execute(
            "INSERT INTO mirror_state(room,last_reconcile_ok_at,generation_ok) VALUES('room',?1,?2)",
            rusqlite::params![last_ok, generation_ok],
        )
        .unwrap();
        for (seq, author, content, deleted) in msgs {
            c.execute(
                "INSERT INTO messages(seq,room,message_id,author_id,content,deleted)
                 VALUES(?1,'room',?1,?2,?3,?4)",
                rusqlite::params![seq, author, content, deleted],
            )
            .unwrap();
        }
        f
    }

    fn now_rfc3339() -> String {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        // Round-trip through our own parser's inverse is unnecessary; build a
        // timestamp we know parses by reusing a fixed date offset from epoch.
        let days = secs / 86_400;
        let rem = secs % 86_400;
        // civil_from_days
        let z = days + 719_468;
        let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        format!(
            "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00",
            rem / 3600,
            (rem % 3600) / 60,
            rem % 60
        )
    }

    #[test]
    fn messages_come_back_in_seq_order_after_the_cursor() {
        let f = fixture(
            1,
            &now_rfc3339(),
            &[(1, "A", "one", 0), (2, "B", "two", 0), (3, "C", "three", 0)],
        );
        let got = messages_since(f.path(), "room", 1, 100)
            .unwrap()
            .ok()
            .unwrap();
        assert_eq!(
            got.iter().map(|m| m.seq).collect::<Vec<_>>(),
            vec![2, 3],
            "must return only messages AFTER the cursor, in seq order"
        );
        assert_eq!(got[0].author_id, "B");
    }

    #[test]
    fn deleted_messages_are_not_returned() {
        let f = fixture(
            1,
            &now_rfc3339(),
            &[(1, "A", "kept", 0), (2, "B", "gone", 1)],
        );
        let got = messages_since(f.path(), "room", 0, 100)
            .unwrap()
            .ok()
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].content, "kept");
    }

    /// The whole point of the migration: an unattested mirror must never be
    /// read. This is the check the old bundled-WASM path could not perform.
    #[test]
    fn an_unattested_generation_is_refused() {
        let f = fixture(0, &now_rfc3339(), &[(1, "A", "x", 0)]);
        let err = messages_since(f.path(), "room", 0, 100)
            .unwrap()
            .err()
            .unwrap();
        assert!(err.0.contains("generation_ok=0"), "{}", err.0);
    }

    #[test]
    fn a_stale_mirror_is_refused() {
        let f = fixture(1, "2020-01-01T00:00:00+00:00", &[(1, "A", "x", 0)]);
        let err = messages_since(f.path(), "room", 0, 100)
            .unwrap()
            .err()
            .unwrap();
        assert!(err.0.contains("wedged"), "{}", err.0);
    }

    #[test]
    fn a_mirror_that_never_reconciled_is_refused() {
        let f = fixture(1, "", &[(1, "A", "x", 0)]);
        let err = messages_since(f.path(), "room", 0, 100)
            .unwrap()
            .err()
            .unwrap();
        assert!(err.0.contains("never completed"), "{}", err.0);
    }

    /// The crawler/mirror compat-break detector. Untested until now, which is
    /// the exact gap this kind of guard is prone to.
    #[test]
    fn an_unknown_schema_version_is_refused() {
        let f = fixture(1, &now_rfc3339(), &[(1, "A", "x", 0)]);
        Connection::open(f.path())
            .unwrap()
            .execute("UPDATE meta SET value='2' WHERE key='schema_version'", [])
            .unwrap();
        let err = messages_since(f.path(), "room", 0, 100)
            .unwrap()
            .err()
            .unwrap();
        assert!(err.0.contains("schema version 2"), "{}", err.0);
    }

    #[test]
    fn a_future_reconcile_timestamp_is_refused_not_treated_as_fresh() {
        let f = fixture(1, "2099-01-01T00:00:00+00:00", &[(1, "A", "x", 0)]);
        let err = messages_since(f.path(), "room", 0, 100)
            .unwrap()
            .err()
            .unwrap();
        assert!(err.0.contains("FUTURE"), "{}", err.0);
    }

    #[test]
    fn a_room_the_mirror_does_not_track_is_refused() {
        let f = fixture(1, &now_rfc3339(), &[(1, "A", "x", 0)]);
        let err = messages_since(f.path(), "other-room", 0, 100)
            .unwrap()
            .err()
            .unwrap();
        assert!(err.0.contains("not mirroring"), "{}", err.0);
    }

    /// A typo'd path must be an ERROR, not an empty result set that reads as
    /// "the room had no new links".
    #[test]
    fn a_missing_database_is_an_error_not_an_empty_read() {
        assert!(messages_since(Path::new("/nonexistent/mirror.sqlite"), "room", 0, 100).is_err());
    }

    #[test]
    fn rfc3339_parses_the_shape_the_mirror_writes() {
        assert_eq!(rfc3339_to_epoch("1970-01-01T00:00:00+00:00"), Some(0));
        assert_eq!(
            rfc3339_to_epoch("2026-08-03T21:49:41.951297099+00:00"),
            Some(1_785_793_781)
        );
        assert_eq!(
            rfc3339_to_epoch("2026-08-03T21:49:41Z"),
            Some(1_785_793_781)
        );
        assert_eq!(rfc3339_to_epoch("garbage"), None);
        assert_eq!(rfc3339_to_epoch(""), None);
    }
}
