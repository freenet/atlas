//! Behavioural tests for room indexing. Room states are built with a local
//! mirror of River's serialisation (same CBOR shapes, same signing bytes); the
//! `golden` module pins that mirror against bytes produced by real `river-core`.

use atlas_common::{
    sign, Audience, Classification, IndexDelta, IndexParams, IndexState, KeyAuth, KeyAuthBody,
    Kind, Locator, RecordBody, SignedRecord, Tombstone, Verification, VerifyStatus, Volatility,
};
use atlas_rooms::eligibility::{
    evaluate, owner_b58, owner_of_locator, parse_owner, room_locator, Decision, Exclusion,
};
use atlas_rooms::entry::{
    next_entry, snippet_source, stored_config_version, Characterization, TextSource, ROOM_KIND,
};
use atlas_rooms::reconcile::{plan, Action, MintOptions};
use atlas_rooms::search::{find_room, search, Availability, RoomFilter};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde::Serialize;

// ---------------------------------------------------------------------------
// A mirror of River's room configuration, for building signed fixtures.
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone)]
enum Pm {
    Public,
    Private,
}

#[derive(Serialize, Clone)]
enum Sealed {
    Public {
        value: Vec<u8>,
    },
    Private {
        ciphertext: Vec<u8>,
        nonce: [u8; 12],
        secret_version: u32,
        declared_len_bytes: u32,
    },
}

#[derive(Serialize, Clone)]
struct Disp {
    name: Sealed,
    description: Option<Sealed>,
}

#[derive(Serialize, Clone)]
struct Policy {
    listed: bool,
}

#[derive(Serialize, Clone)]
struct Cfg {
    owner_member_id: u64,
    configuration_version: u32,
    privacy_mode: Pm,
    display: Disp,
    max_recent_messages: usize,
    max_user_bans: usize,
    max_message_size: usize,
    max_nickname_size: usize,
    max_members: usize,
    max_room_name: usize,
    max_room_description: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_direct_messages: Option<usize>,
    // Provisional river#974 shape; present only when a test sets it.
    #[serde(skip_serializing_if = "Option::is_none")]
    public_room_policy: Option<Policy>,
}

#[derive(Serialize)]
struct Auth {
    configuration: Cfg,
    signature: ed25519_dalek::Signature,
}

#[derive(Serialize)]
struct State {
    configuration: Auth,
    // Stand-in for the many other fields of ChatRoomStateV1, which must be
    // ignored by the reader.
    members: Vec<u8>,
    recent_messages: Vec<Vec<u8>>,
}

fn public_text(s: &str) -> Sealed {
    Sealed::Public {
        value: s.as_bytes().to_vec(),
    }
}

fn cfg(version: u32, mode: Pm, name: Sealed, desc: Option<Sealed>) -> Cfg {
    Cfg {
        owner_member_id: 42,
        configuration_version: version,
        privacy_mode: mode,
        display: Disp {
            name,
            description: desc,
        },
        max_recent_messages: 100,
        max_user_bans: 10,
        max_message_size: 1000,
        max_nickname_size: 50,
        max_members: 200,
        max_room_name: 128,
        max_room_description: 1024,
        max_direct_messages: None,
        public_room_policy: None,
    }
}

fn signed_state(sk: &SigningKey, c: Cfg) -> Vec<u8> {
    let mut signed = Vec::new();
    ciborium::ser::into_writer(&c, &mut signed).unwrap();
    let signature = sk.sign(&signed);
    let st = State {
        configuration: Auth {
            configuration: c,
            signature,
        },
        members: vec![1, 2, 3],
        recent_messages: vec![vec![9; 4]],
    };
    let mut out = Vec::new();
    ciborium::ser::into_writer(&st, &mut out).unwrap();
    out
}

fn public_room(sk: &SigningKey, v: u32, name: &str, desc: Option<&str>) -> Vec<u8> {
    signed_state(
        sk,
        cfg(v, Pm::Public, public_text(name), desc.map(public_text)),
    )
}

fn key() -> SigningKey {
    SigningKey::generate(&mut OsRng)
}

fn eligible(d: Decision) -> atlas_rooms::eligibility::RoomDescriptor {
    match d {
        Decision::Eligible(d) => d,
        other => panic!("expected eligible, got {other:?}"),
    }
}

fn excluded(d: Decision) -> (Exclusion, Option<u32>) {
    match d {
        Decision::Excluded(e, v) => (e, v),
        other => panic!("expected excluded, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Eligibility
// ---------------------------------------------------------------------------

#[test]
fn a_submitted_public_room_is_eligible_with_its_owner_text() {
    let sk = key();
    let st = public_room(&sk, 3, "Rust Chat", Some("All about Rust"));
    let d = eligible(evaluate(&sk.verifying_key(), &st, true));
    assert_eq!(d.name, "Rust Chat");
    assert_eq!(d.description.as_deref(), Some("All about Rust"));
    assert_eq!(d.configuration_version, 3);
}

#[test]
fn private_rooms_are_excluded_and_their_text_is_never_read() {
    let sk = key();
    // A private room whose plaintext name is a secret-looking string: it must
    // not appear anywhere in the outcome.
    let st = signed_state(&sk, cfg(1, Pm::Private, public_text("TOPSECRET"), None));
    let (why, v) = excluded(evaluate(&sk.verifying_key(), &st, true));
    assert_eq!(why, Exclusion::Private);
    assert!(!format!("{why:?}").contains("TOPSECRET"));
    assert_eq!(v, Some(1));
    assert!(why.delists());
}

#[test]
fn encrypted_names_are_excluded_even_in_public_mode() {
    let sk = key();
    let name = Sealed::Private {
        ciphertext: vec![1; 8],
        nonce: [0; 12],
        secret_version: 1,
        declared_len_bytes: 4,
    };
    let st = signed_state(&sk, cfg(1, Pm::Public, name, None));
    assert_eq!(
        excluded(evaluate(&sk.verifying_key(), &st, true)).0,
        Exclusion::EncryptedName
    );
}

#[test]
fn encrypted_description_costs_the_blurb_not_the_listing() {
    let sk = key();
    let desc = Sealed::Private {
        ciphertext: vec![1; 8],
        nonce: [0; 12],
        secret_version: 1,
        declared_len_bytes: 4,
    };
    let st = signed_state(&sk, cfg(1, Pm::Public, public_text("Room"), Some(desc)));
    let d = eligible(evaluate(&sk.verifying_key(), &st, true));
    assert_eq!(d.description, None);
}

#[test]
fn without_a_listed_flag_an_unsubmitted_room_is_not_indexed() {
    let sk = key();
    let st = public_room(&sk, 1, "Room", None);
    assert_eq!(
        excluded(evaluate(&sk.verifying_key(), &st, false)).0,
        Exclusion::NotListed
    );
}

#[test]
fn the_listed_hook_opts_in_and_an_opt_out_beats_a_submission() {
    let sk = key();
    let mut c = cfg(1, Pm::Public, public_text("Room"), None);
    c.public_room_policy = Some(Policy { listed: true });
    let st = signed_state(&sk, c.clone());
    // listed == true needs no submission.
    eligible(evaluate(&sk.verifying_key(), &st, false));
    // listed == false wins even over an explicit submission.
    c.public_room_policy = Some(Policy { listed: false });
    let st = signed_state(&sk, c);
    assert_eq!(
        excluded(evaluate(&sk.verifying_key(), &st, true)).0,
        Exclusion::NotListed
    );
}

#[test]
fn a_private_room_is_excluded_even_if_listed() {
    let sk = key();
    let mut c = cfg(1, Pm::Private, public_text("Room"), None);
    c.public_room_policy = Some(Policy { listed: true });
    let st = signed_state(&sk, c);
    assert_eq!(
        excluded(evaluate(&sk.verifying_key(), &st, true)).0,
        Exclusion::Private
    );
}

#[test]
fn a_configuration_not_signed_by_the_named_owner_proves_nothing() {
    let owner = key();
    let impostor = key();
    let st = public_room(&impostor, 1, "Fake", None);
    let (why, v) = excluded(evaluate(&owner.verifying_key(), &st, true));
    assert!(matches!(why, Exclusion::Unverifiable(_)));
    assert!(!why.delists(), "an unverifiable read must never delist");
    assert_eq!(v, None);
}

#[test]
fn tampering_with_a_field_the_reader_does_not_model_breaks_the_signature() {
    // The signature covers `max_members`, which the view never reads. Flipping
    // it in the bytes must be detected, which only holds because verification is
    // over the full generic value and not over a mirrored subset.
    let sk = key();
    let mut c = cfg(1, Pm::Public, public_text("Room"), None);
    let st = signed_state(&sk, c.clone());
    c.max_members = 201;
    // Re-encode with the OLD signature but the NEW field.
    let mut old_signed = Vec::new();
    ciborium::ser::into_writer(
        &cfg(1, Pm::Public, public_text("Room"), None),
        &mut old_signed,
    )
    .unwrap();
    let sig = sk.sign(&old_signed);
    let forged = State {
        configuration: Auth {
            configuration: c,
            signature: sig,
        },
        members: vec![],
        recent_messages: vec![],
    };
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(&forged, &mut bytes).unwrap();
    assert!(matches!(
        excluded(evaluate(&sk.verifying_key(), &bytes, true)).0,
        Exclusion::Unverifiable(_)
    ));
    // And the untampered one is fine, so the above is not vacuous.
    eligible(evaluate(&sk.verifying_key(), &st, true));
}

#[test]
fn garbage_state_is_unverifiable_not_a_panic() {
    let sk = key();
    for junk in [&b""[..], b"not cbor", &[0xff, 0xff, 0xff], &[0xa0]] {
        let (why, _) = excluded(evaluate(&sk.verifying_key(), junk, true));
        assert!(matches!(why, Exclusion::Unverifiable(_)), "{junk:?}");
    }
}

#[test]
fn hostile_text_is_neutralised_into_a_publishable_entry() {
    let sk = key();
    let name = "Ro\x1b[2Jom\u{202E}evil\u{200B}\r\n";
    let desc = "line1\nline2\x07\u{E0041} ".to_string() + &"x".repeat(2000);
    let st = public_room(&sk, 1, name, Some(&desc));
    let d = eligible(evaluate(&sk.verifying_key(), &st, true));
    for t in [d.name.as_str(), d.description.as_deref().unwrap()] {
        assert!(!t.chars().any(|c| c.is_control()), "{t:?}");
        assert!(!t.contains('\u{202E}') && !t.contains('\u{200B}') && !t.contains('\u{E0041}'));
    }
    assert!(d.description.as_ref().unwrap().len() <= atlas_common::MAX_SNIPPET);
    // The result passes the contract's own structural check.
    let e = atlas_rooms::entry::new_entry(&d, None, vec![], false, None, None, 10).unwrap();
    RecordBody::Live(e).check_structure().unwrap();
}

#[test]
fn an_empty_or_unprintable_or_non_utf8_name_is_excluded() {
    let sk = key();
    for name in [
        vec![],
        b" \x07 ".to_vec(),
        vec![0xff, 0xfe, 0xfd],
        "\u{202E}\u{200B}".as_bytes().to_vec(),
    ] {
        let st = signed_state(
            &sk,
            cfg(1, Pm::Public, Sealed::Public { value: name }, None),
        );
        assert_eq!(
            excluded(evaluate(&sk.verifying_key(), &st, true)).0,
            Exclusion::UnusableName
        );
    }
}

#[test]
fn a_non_utf8_description_is_dropped_not_fatal() {
    let sk = key();
    let st = signed_state(
        &sk,
        cfg(
            1,
            Pm::Public,
            public_text("Room"),
            Some(Sealed::Public {
                value: vec![0xff, 0xfe],
            }),
        ),
    );
    assert_eq!(
        eligible(evaluate(&sk.verifying_key(), &st, true)).description,
        None
    );
}

// ---------------------------------------------------------------------------
// Locator identity
// ---------------------------------------------------------------------------

#[test]
fn the_locator_is_the_full_owner_key_and_round_trips() {
    let vk = key().verifying_key();
    let loc = room_locator(&vk);
    assert_eq!(
        loc,
        Locator::AppResource {
            app: "river".into(),
            resource: owner_b58(&vk),
            path: String::new()
        }
    );
    loc.check().unwrap();
    assert_eq!(owner_of_locator(&loc), Some(vk));
    assert_eq!(parse_owner(&owner_b58(&vk)).unwrap(), vk);
    assert_eq!(loc.to_uri(), format!("app:river/{}", owner_b58(&vk)));
    // A different app's resource is never read as a room.
    let other = Locator::AppResource {
        app: "delta".into(),
        resource: owner_b58(&vk),
        path: String::new(),
    };
    assert_eq!(owner_of_locator(&other), None);
}

#[test]
fn short_or_malformed_owner_keys_are_refused() {
    assert!(parse_owner("abc").is_err());
    assert!(parse_owner("0OIl").is_err());
    let short = bs58::encode([1u8; 16]).into_string();
    assert!(parse_owner(&short).is_err());
}

// ---------------------------------------------------------------------------
// Index fixtures
// ---------------------------------------------------------------------------

struct Idx {
    root: SigningKey,
    online: SigningKey,
    params: IndexParams,
    state: IndexState,
}

fn index() -> Idx {
    let root = key();
    let online = key();
    let body = KeyAuthBody {
        version: 1,
        authorized: vec![online.verifying_key()],
    };
    let ka = KeyAuth {
        sig: sign(&body, &root),
        body,
    };
    Idx {
        params: IndexParams {
            root_vk: root.verifying_key(),
            slug: "main".into(),
        },
        state: IndexState::initialized(ka),
        root,
        online,
    }
}

fn sign_rec(body: RecordBody, online: &SigningKey) -> SignedRecord {
    body.check_structure()
        .expect("body must be structurally valid");
    SignedRecord {
        sig: sign(&body, online),
        by: online.verifying_key(),
        body,
    }
}

fn apply(idx: &mut Idx, p: atlas_rooms::reconcile::Plan) {
    let mut records = Vec::new();
    for a in p.actions {
        let body = match a {
            Action::Add(e) | Action::Update(e) => RecordBody::Live(*e),
            Action::Tombstone(t) => RecordBody::Tomb(t),
        };
        records.push(sign_rec(body, &idx.online));
    }
    let delta = IndexDelta {
        key_auth: None,
        records,
        apps: None,
    };
    idx.state
        .apply_delta(&delta, &idx.params)
        .expect("contract accepts the delta");
    idx.state.verify(&idx.params).expect("state verifies");
}

fn general() -> Classification {
    Classification {
        landing: Audience::new(Audience::GENERAL),
        has_adult_sections: false,
        volatility: Volatility::new(Volatility::FEED),
        classifier: 0,
        classified_at: 1,
    }
}

fn plan_for(
    idx: &Idx,
    sk: &SigningKey,
    state: &[u8],
    ch: Option<&Characterization>,
    now: u64,
) -> atlas_rooms::reconcile::Plan {
    let vk = sk.verifying_key();
    let d = evaluate(&vk, state, true);
    plan(
        &idx.state,
        &vk,
        &d,
        ch,
        &MintOptions {
            tags: vec!["rust".into()],
            featured: false,
            class: Some(general()),
        },
        now,
    )
    .unwrap()
}

fn live_rooms(idx: &Idx) -> Vec<atlas_common::IndexEntry> {
    idx.state
        .live_entries()
        .filter(|e| e.kind == Kind::new(ROOM_KIND))
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// Reconcile: add, duplicate, rename, stale, delist
// ---------------------------------------------------------------------------

#[test]
fn adding_a_room_mints_a_valid_signed_entry() {
    let mut idx = index();
    let sk = key();
    let st = public_room(&sk, 1, "Rust Chat", Some("All about Rust"));
    let p = plan_for(&idx, &sk, &st, None, 100);
    assert!(matches!(p.actions.as_slice(), [Action::Add(_)]));
    apply(&mut idx, p);
    let rooms = live_rooms(&idx);
    assert_eq!(rooms.len(), 1);
    let e = &rooms[0];
    assert_eq!(e.title, "Rust Chat");
    // The owner's description is NOT copied into the index.
    assert_eq!(e.snippet, "");
    assert_eq!(snippet_source(e), None);
    assert_eq!(stored_config_version(e), Some(1));
    assert_eq!(
        e.verified.as_ref().unwrap().status.as_str(),
        VerifyStatus::LIVE
    );
    assert_eq!(e.locator, room_locator(&sk.verifying_key()));
    assert!(e.version == 1 && !e.subject_id.as_str().is_empty());
}

#[test]
fn re_planning_an_unchanged_room_publishes_nothing() {
    let mut idx = index();
    let sk = key();
    let st = public_room(&sk, 1, "Rust Chat", None);
    let p = plan_for(&idx, &sk, &st, None, 100);
    apply(&mut idx, p);
    let p = plan_for(&idx, &sk, &st, None, 200);
    assert!(p.actions.is_empty(), "{:?}", p.notes);
}

#[test]
fn a_stale_verification_is_renewed_after_the_window_with_a_version_bump() {
    let mut idx = index();
    let sk = key();
    let st = public_room(&sk, 1, "Rust Chat", None);
    let p = plan_for(&idx, &sk, &st, None, 100);
    apply(&mut idx, p);
    let later = 100 + atlas_rooms::entry::REVERIFY_AFTER_SECS + 1;
    let p = plan_for(&idx, &sk, &st, None, later);
    assert!(matches!(p.actions.as_slice(), [Action::Update(e)] if e.version == 2));
}

#[test]
fn a_rename_updates_the_same_subject_and_keeps_its_identity() {
    let mut idx = index();
    let sk = key();
    let p = plan_for(&idx, &sk, &public_room(&sk, 1, "Old Name", None), None, 100);
    apply(&mut idx, p);
    let before = live_rooms(&idx)[0].clone();

    let p = plan_for(&idx, &sk, &public_room(&sk, 2, "New Name", None), None, 200);
    assert!(
        p.notes.iter().any(|n| n.contains("rename")),
        "{:?}",
        p.notes
    );
    apply(&mut idx, p);

    let rooms = live_rooms(&idx);
    assert_eq!(rooms.len(), 1, "a rename must not create a second listing");
    assert_eq!(rooms[0].subject_id, before.subject_id);
    assert_eq!(rooms[0].title, "New Name");
    assert_eq!(rooms[0].version, before.version + 1);
    assert_eq!(rooms[0].added_at, before.added_at);
    assert_eq!(rooms[0].locator, before.locator);
    // Curator classification and tags survive a refresh.
    assert_eq!(rooms[0].class, before.class);
    assert_eq!(rooms[0].tags, before.tags);
}

#[test]
fn a_read_older_than_the_entry_does_not_revert_a_rename() {
    let mut idx = index();
    let sk = key();
    let p = plan_for(&idx, &sk, &public_room(&sk, 5, "Current", None), None, 100);
    apply(&mut idx, p);
    // A lagging node answers with configuration version 3.
    let p = plan_for(&idx, &sk, &public_room(&sk, 3, "Ancient", None), None, 200);
    assert!(p.actions.is_empty(), "{:?}", p.notes);
    assert_eq!(live_rooms(&idx)[0].title, "Current");
}

#[test]
fn a_stale_private_read_does_not_delist_but_a_current_one_does() {
    let mut idx = index();
    let sk = key();
    let p = plan_for(&idx, &sk, &public_room(&sk, 5, "Room", None), None, 100);
    apply(&mut idx, p);

    let stale_private = signed_state(&sk, cfg(4, Pm::Private, public_text("Room"), None));
    let p = plan_for(&idx, &sk, &stale_private, None, 200);
    assert!(p.actions.is_empty(), "{:?}", p.notes);
    assert_eq!(live_rooms(&idx).len(), 1);

    let private = signed_state(&sk, cfg(6, Pm::Private, public_text("Room"), None));
    let p = plan_for(&idx, &sk, &private, None, 300);
    assert!(matches!(p.actions.as_slice(), [Action::Tombstone(_)]));
    apply(&mut idx, p);
    assert!(
        live_rooms(&idx).is_empty(),
        "a room gone private must be delisted"
    );
}

#[test]
fn a_room_that_opts_out_of_listing_is_delisted() {
    let mut idx = index();
    let sk = key();
    let p = plan_for(&idx, &sk, &public_room(&sk, 1, "Room", None), None, 100);
    apply(&mut idx, p);
    let mut c = cfg(2, Pm::Public, public_text("Room"), None);
    c.public_room_policy = Some(Policy { listed: false });
    let p = plan_for(&idx, &sk, &signed_state(&sk, c), None, 200);
    apply(&mut idx, p);
    assert!(live_rooms(&idx).is_empty());
}

#[test]
fn private_rooms_are_never_added() {
    let idx = index();
    let sk = key();
    let st = signed_state(&sk, cfg(1, Pm::Private, public_text("Hidden"), None));
    let p = plan_for(&idx, &sk, &st, None, 100);
    assert!(p.actions.is_empty());
    assert!(p.notes[0].contains("private"), "{:?}", p.notes);
}

#[test]
fn duplicates_collapse_to_one_canonical_subject() {
    let mut idx = index();
    let sk = key();
    let vk = sk.verifying_key();
    let d = eligible(evaluate(&vk, &public_room(&sk, 1, "Dup", None), true));
    // Two live entries for one owner key, minted independently (as an
    // `--allow-duplicate` add or a race could).
    let a =
        atlas_rooms::entry::new_entry(&d, None, vec![], false, Some(general()), None, 10).unwrap();
    let mut b =
        atlas_rooms::entry::new_entry(&d, None, vec![], false, Some(general()), None, 20).unwrap();
    b.version = 4;
    let delta = IndexDelta {
        key_auth: None,
        records: vec![
            sign_rec(RecordBody::Live(a.clone()), &idx.online),
            sign_rec(RecordBody::Live(b.clone()), &idx.online),
        ],
        apps: None,
    };
    idx.state.apply_delta(&delta, &idx.params).unwrap();
    assert_eq!(live_rooms(&idx).len(), 2);

    // Search already collapses them to the higher-versioned one.
    let hits = search(&idx.state, &RoomFilter::default());
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].subject_id, b.subject_id.as_str());

    let p = plan_for(&idx, &sk, &public_room(&sk, 1, "Dup", None), None, 300);
    let tombs: Vec<&Tombstone> = p
        .actions
        .iter()
        .filter_map(|a| {
            if let Action::Tombstone(t) = a {
                Some(t)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(tombs.len(), 1);
    assert_eq!(
        tombs[0].subject_id, a.subject_id,
        "the lower version is the duplicate"
    );
    apply(&mut idx, p);
    assert_eq!(live_rooms(&idx).len(), 1);
    assert_eq!(live_rooms(&idx)[0].subject_id, b.subject_id);
}

#[test]
fn an_unverifiable_read_marks_unreachable_once_and_never_delists() {
    let mut idx = index();
    let sk = key();
    let p = plan_for(&idx, &sk, &public_room(&sk, 1, "Room", None), None, 100);
    apply(&mut idx, p);

    let impostor = public_room(&key(), 9, "Evil", None);
    let p = plan_for(&idx, &sk, &impostor, None, 200);
    assert!(matches!(p.actions.as_slice(), [Action::Update(_)]));
    apply(&mut idx, p);
    let e = &live_rooms(&idx)[0];
    assert_eq!(
        e.verified.as_ref().unwrap().status.as_str(),
        VerifyStatus::UNREACHABLE
    );
    assert_eq!(e.title, "Room", "the entry itself is untouched");

    // A second failure does not churn another version.
    let p = plan_for(&idx, &sk, b"junk", None, 300);
    assert!(p.actions.is_empty(), "{:?}", p.notes);

    // Recovery: a good read flips it back to live with a version bump.
    let p = plan_for(&idx, &sk, &public_room(&sk, 1, "Room", None), None, 400);
    apply(&mut idx, p);
    assert_eq!(
        live_rooms(&idx)[0]
            .verified
            .as_ref()
            .unwrap()
            .status
            .as_str(),
        VerifyStatus::LIVE
    );
}

// ---------------------------------------------------------------------------
// Owner text stays in the room; the index carries Atlas's own text
// ---------------------------------------------------------------------------

#[test]
fn the_owner_description_is_never_stored_and_curator_text_is_attributed() {
    let mut idx = index();
    let sk = key();
    let ch =
        Characterization::new("A busy channel for Rust learners.", TextSource::Curator).unwrap();
    let st = public_room(&sk, 1, "Rust Chat", Some("OWNER WORDS"));
    let p = plan_for(&idx, &sk, &st, Some(&ch), 100);
    apply(&mut idx, p);
    let e = &live_rooms(&idx)[0];
    assert_eq!(e.snippet, "A busy channel for Rust learners.");
    assert_eq!(snippet_source(e), Some(TextSource::Curator));
    assert_eq!(
        e.ext.as_ref().unwrap().get("ts").map(String::as_str),
        Some("o")
    );
    // Nowhere in the signed entry is the owner's description.
    assert!(!format!("{e:?}").contains("OWNER WORDS"));

    // The owner editing their description changes nothing in the index, and a
    // refresh does not disturb the curator text.
    let st2 = public_room(&sk, 2, "Rust Chat", Some("NEW OWNER WORDS"));
    let p = plan_for(&idx, &sk, &st2, None, 200);
    apply(&mut idx, p);
    let e = &live_rooms(&idx)[0];
    assert_eq!(e.snippet, "A busy channel for Rust learners.");
    assert_eq!(snippet_source(e), Some(TextSource::Curator));
    assert!(!format!("{e:?}").contains("NEW OWNER WORDS"));

    // An LLM characterisation is attributed as such.
    let llm = Characterization::new("Chat about Rust.", TextSource::Llm).unwrap();
    let p = plan_for(&idx, &sk, &st2, Some(&llm), 300);
    apply(&mut idx, p);
    assert_eq!(snippet_source(&live_rooms(&idx)[0]), Some(TextSource::Llm));
}

#[test]
fn an_owner_cannot_be_the_source_of_a_characterization() {
    assert!(Characterization::new("x", TextSource::Owner).is_err());
    assert!(Characterization::new("\x07\u{200B}", TextSource::Curator).is_err());
}

#[test]
fn another_writers_ext_keys_survive_a_refresh() {
    let mut idx = index();
    let sk = key();
    let vk = sk.verifying_key();
    let d = eligible(evaluate(&vk, &public_room(&sk, 1, "R", None), true));
    let mut extra = std::collections::BTreeMap::new();
    extra.insert("crawl".to_string(), "x".to_string());
    let e =
        atlas_rooms::entry::new_entry(&d, None, vec![], false, Some(general()), Some(extra), 10)
            .unwrap();
    let next = next_entry(
        &e,
        &eligible(evaluate(&vk, &public_room(&sk, 2, "Renamed", None), true)),
        None,
        20,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        next.ext.as_ref().unwrap().get("crawl").map(String::as_str),
        Some("x")
    );
    drop(idx.state);
}

// ---------------------------------------------------------------------------
// Signed-record byte compatibility and migration
// ---------------------------------------------------------------------------

fn cbor<T: Serialize>(v: &T) -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::ser::into_writer(v, &mut b).unwrap();
    b
}

#[test]
fn room_entries_survive_a_cbor_round_trip_and_still_verify() {
    // `crate::verify` re-serialises the CURRENT in-memory struct to check a
    // signature, so a record that does not round-trip byte-identically would
    // fail verification on the node.
    let mut idx = index();
    let sk = key();
    let p = plan_for(
        &idx,
        &sk,
        &public_room(&sk, 1, "Rust Chat", Some("d")),
        None,
        100,
    );
    apply(&mut idx, p);
    let bytes = cbor(&idx.state);
    let back: IndexState = ciborium::de::from_reader(bytes.as_slice()).unwrap();
    assert_eq!(back, idx.state);
    assert_eq!(cbor(&back), bytes);
    back.verify(&idx.params).unwrap();
}

#[test]
fn legacy_entries_are_byte_identical_next_to_room_entries() {
    // An entry written before rooms existed (no ext, no class, no verified)
    // must still serialise to the same bytes, and a state mixing it with a room
    // entry must still verify. This is what "no re-key" means for stored data.
    let mut idx = index();
    let legacy = atlas_common::IndexEntry {
        subject_id: atlas_common::SubjectId::random(),
        version: 1,
        kind: Kind::new(Kind::SITE),
        title: "A site".into(),
        snippet: "s".into(),
        tags: vec![],
        locator: Locator::Freenet {
            contract_id: "1".repeat(44),
            path: String::new(),
        },
        featured: false,
        added_at: 5,
        class: None,
        verified: None,
        ext: None,
    };
    let before = cbor(&RecordBody::Live(legacy.clone()));
    assert!(!String::from_utf8_lossy(&before).contains("ext"));
    let rec = sign_rec(RecordBody::Live(legacy.clone()), &idx.online);
    idx.state
        .apply_delta(
            &IndexDelta {
                key_auth: None,
                records: vec![rec.clone()],
                apps: None,
            },
            &idx.params,
        )
        .unwrap();

    let sk = key();
    let p = plan_for(&idx, &sk, &public_room(&sk, 1, "R", None), None, 100);
    apply(&mut idx, p);

    // The legacy record is untouched, signature included.
    let kept = idx.state.records.get(&legacy.subject_id).unwrap();
    assert_eq!(kept, &rec);
    kept.verify_sig().unwrap();
    assert_eq!(cbor(&kept.body), before);
}

#[test]
fn an_index_migrates_with_its_room_entries_intact() {
    // The migration path merges a PRIOR generation's state into the new one
    // under the same params. Room entries are ordinary records, so they carry
    // across unchanged, and merging is idempotent and order-independent.
    let mut old = index();
    let a = key();
    let b = key();
    let p = plan_for(&old, &a, &public_room(&a, 1, "A", None), None, 100);
    apply(&mut old, p);
    let p = plan_for(&old, &b, &public_room(&b, 1, "B", None), None, 100);
    apply(&mut old, p);

    let mut new_gen = IndexState::initialized(old.state.key_auth.clone().unwrap());
    new_gen.merge(&old.state);
    new_gen.verify(&old.params).unwrap();
    assert_eq!(new_gen, old.state);

    let mut twice = new_gen.clone();
    twice.merge(&old.state);
    assert_eq!(twice, new_gen);

    let hits = search(&new_gen, &RoomFilter::default());
    assert_eq!(hits.len(), 2);
}

#[test]
fn the_ext_budget_is_respected() {
    let sk = key();
    let vk = sk.verifying_key();
    let d = eligible(evaluate(&vk, &public_room(&sk, u32::MAX, "R", None), true));
    let e = atlas_rooms::entry::new_entry(&d, None, vec![], false, None, None, 1).unwrap();
    let total: usize = e
        .ext
        .as_ref()
        .unwrap()
        .iter()
        .map(|(k, v)| k.len() + v.len())
        .sum();
    assert!(total <= 32, "room ext uses {total} B of the 64 B budget");
    RecordBody::Live(e).check_structure().unwrap();
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

fn add_room(
    idx: &mut Idx,
    name: &str,
    desc: Option<&str>,
    tags: &[&str],
    class: Option<Classification>,
    featured: bool,
    now: u64,
) -> VerifyingKey {
    let sk = key();
    let vk = sk.verifying_key();
    let d = evaluate(&vk, &public_room(&sk, 1, name, None), true);
    let ch = desc.map(|t| Characterization::new(t, TextSource::Curator).unwrap());
    let p = plan(
        &idx.state,
        &vk,
        &d,
        ch.as_ref(),
        &MintOptions {
            tags: tags.iter().map(|t| t.to_string()).collect(),
            featured,
            class,
        },
        now,
    )
    .unwrap();
    apply(idx, p);
    vk
}

#[test]
fn search_filters_by_text_tag_and_ranks_title_first() {
    let mut idx = index();
    add_room(
        &mut idx,
        "Gardening",
        Some("tomatoes and rust spots"),
        &["plants"],
        Some(general()),
        false,
        10,
    );
    let rc = add_room(
        &mut idx,
        "Rust Chat",
        Some("systems"),
        &["lang"],
        Some(general()),
        false,
        20,
    );
    add_room(
        &mut idx,
        "Cooking",
        None,
        &["rust"],
        Some(general()),
        false,
        30,
    );

    let f = RoomFilter {
        query: Some("RUST".into()),
        ..Default::default()
    };
    let hits = search(&idx.state, &f);
    let names: Vec<&str> = hits.iter().map(|h| h.title.as_str()).collect();
    // title match, then tag match, then snippet match.
    assert_eq!(names, ["Rust Chat", "Cooking", "Gardening"]);
    assert_eq!(hits[0].owner_key, owner_b58(&rc));

    let f = RoomFilter {
        tags: vec!["LANG".into()],
        ..Default::default()
    };
    assert_eq!(search(&idx.state, &f).len(), 1);
    let f = RoomFilter {
        query: Some("zzz".into()),
        ..Default::default()
    };
    assert!(search(&idx.state, &f).is_empty());
}

#[test]
fn safe_search_hides_unclassified_and_non_general_rooms() {
    let mut idx = index();
    add_room(&mut idx, "Open", None, &[], Some(general()), false, 10);
    add_room(&mut idx, "Unrated", None, &[], None, false, 20);
    let mut adult = general();
    adult.landing = Audience::new(Audience::ADULT);
    add_room(&mut idx, "Adult", None, &[], Some(adult), false, 30);
    let mut future = general();
    future.landing = Audience::new("explicit");
    add_room(&mut idx, "Future", None, &[], Some(future), false, 40);

    let safe: Vec<_> = search(&idx.state, &RoomFilter::default())
        .into_iter()
        .map(|h| h.title)
        .collect();
    assert_eq!(safe, ["Open"], "unknown landing values fail closed");

    let all = search(
        &idx.state,
        &RoomFilter {
            safe_search: false,
            ..Default::default()
        },
    );
    assert_eq!(all.len(), 4);
}

#[test]
fn search_orders_featured_then_newest_and_is_deterministic() {
    let mut idx = index();
    add_room(&mut idx, "Old", None, &[], Some(general()), false, 10);
    add_room(&mut idx, "New", None, &[], Some(general()), false, 50);
    add_room(&mut idx, "Star", None, &[], Some(general()), true, 5);
    let names: Vec<_> = search(&idx.state, &RoomFilter::default())
        .into_iter()
        .map(|h| h.title)
        .collect();
    assert_eq!(names, ["Star", "New", "Old"]);
}

#[test]
fn availability_reflects_verification_and_staleness() {
    let mut idx = index();
    let live = add_room(&mut idx, "Live", None, &[], Some(general()), false, 100);
    let f = RoomFilter {
        now: 100,
        stale_after_secs: Some(1000),
        ..Default::default()
    };
    assert_eq!(
        find_room(&idx.state, &owner_b58(&live), &f)
            .unwrap()
            .availability,
        Availability::Live
    );
    let f = RoomFilter {
        now: 5000,
        stale_after_secs: Some(1000),
        ..Default::default()
    };
    assert_eq!(
        find_room(&idx.state, &owner_b58(&live), &f)
            .unwrap()
            .availability,
        Availability::Stale
    );

    let f = RoomFilter {
        only_live: true,
        now: 5000,
        stale_after_secs: Some(1000),
        ..Default::default()
    };
    assert!(search(&idx.state, &f).is_empty());

    // An entry with no verification at all is Unknown, never Live.
    let sk = key();
    let d = eligible(evaluate(
        &sk.verifying_key(),
        &public_room(&sk, 1, "NV", None),
        true,
    ));
    let mut e =
        atlas_rooms::entry::new_entry(&d, None, vec![], false, Some(general()), None, 1).unwrap();
    e.verified = None::<Verification>;
    idx.state
        .apply_delta(
            &IndexDelta {
                key_auth: None,
                records: vec![sign_rec(RecordBody::Live(e), &idx.online)],
                apps: None,
            },
            &idx.params,
        )
        .unwrap();
    let h = find_room(
        &idx.state,
        &owner_b58(&sk.verifying_key()),
        &RoomFilter::default(),
    )
    .unwrap();
    assert_eq!(h.availability, Availability::Unknown);
}

#[test]
fn search_ignores_non_room_entries_and_malformed_room_locators() {
    let mut idx = index();
    // A Site entry and a Room-kind entry whose locator is not a River owner key.
    for (kind, loc) in [
        (
            Kind::SITE,
            Locator::Freenet {
                contract_id: "1".repeat(44),
                path: String::new(),
            },
        ),
        (
            ROOM_KIND,
            Locator::AppResource {
                app: "river".into(),
                resource: "abc".into(),
                path: String::new(),
            },
        ),
    ] {
        let e = atlas_common::IndexEntry {
            subject_id: atlas_common::SubjectId::random(),
            version: 1,
            kind: Kind::new(kind),
            title: "t".into(),
            snippet: String::new(),
            tags: vec![],
            locator: loc,
            featured: false,
            added_at: 1,
            class: Some(general()),
            verified: None,
            ext: None,
        };
        idx.state
            .apply_delta(
                &IndexDelta {
                    key_auth: None,
                    records: vec![sign_rec(RecordBody::Live(e), &idx.online)],
                    apps: None,
                },
                &idx.params,
            )
            .unwrap();
    }
    assert!(search(&idx.state, &RoomFilter::default()).is_empty());
    let _ = &idx.root;
}

// ---------------------------------------------------------------------------
// Golden: bytes produced by real river-core
// ---------------------------------------------------------------------------

mod golden {
    use super::*;

    include!("golden_fixture.rs");

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn real_river_core_states_decode_and_verify() {
        let vk = VerifyingKey::from_bytes(&unhex(OWNER_VK).try_into().unwrap()).unwrap();
        let d = eligible(evaluate(&vk, &unhex(PUBLIC), true));
        assert_eq!(d.name, "Rust Chat");
        assert_eq!(d.description.as_deref(), Some("All about Rust"));
        assert_eq!(d.configuration_version, 3);
        assert_eq!(
            excluded(evaluate(&vk, &unhex(PRIVATE), true)).0,
            Exclusion::Private
        );
        // Not signed by this key: rejected, which also proves the fixture is
        // being verified rather than skipped.
        let other = key().verifying_key();
        assert!(matches!(
            excluded(evaluate(&other, &unhex(PUBLIC), true)).0,
            Exclusion::Unverifiable(_)
        ));
    }
}
