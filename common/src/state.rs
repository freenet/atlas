use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::types::{
    AppRegistry, IndexEntry, IndexParams, KeyAuth, RecordBody, SignedRecord, SubjectId,
};
use crate::{MAX_AUTHORIZED, MAX_ENTRIES};

/// Total order used to resolve which app registry wins. Higher version wins;
/// ties break deterministically on signature bytes so merge is commutative.
///
/// Unlike `key_auth` (immutable in 0.1, see [`IndexState::merge`]), the registry
/// is genuinely mutable: the curator re-points an app whenever it republishes.
/// That is safe for associativity here because the registry does not gate
/// authorization of anything else in the state, so no record's admissibility
/// depends on which registry version is current. Records are authorized by
/// `key_auth` alone, and an `AppResource` entry naming an app the registry does
/// not (yet) know is structurally valid and merely unresolvable.
fn registry_order(r: &AppRegistry) -> (u64, [u8; 64]) {
    (r.body.version, r.sig.to_bytes())
}

/// The full index state: a root-signed authorization of online keys, plus one
/// current record per subject. Records merge as a commutative monoid (per-subject
/// last-write-wins by `(version, signature)`), so peers converge regardless of
/// the order updates arrive in.
///
/// ADDING A FIELD (the migration path constrains this). `atlasctl migrate` GETs
/// a PRIOR generation's state and deserializes it with the CURRENT types, so a
/// field the old encoding does not have must default rather than fail the whole
/// decode — a decode error there would silently strand the entire curated index.
/// Two verified facts about the encoding govern what is safe (both pinned by
/// `a_state_encoded_before_the_apps_field_still_decodes` and
/// `a_summary_encoded_before_apps_version_still_decodes`):
///
/// 1. ciborium encodes a struct as a CBOR **map keyed by field name**, not a
///    positional array, so a new field may be added anywhere and reordering
///    fields is not a wire change.
/// 2. A missing `Option<T>` field decodes to `None` on its own (serde's derive
///    special-cases it); a missing NON-`Option` field is a hard decode error.
///
/// So `#[serde(default)]` is REQUIRED on any new non-`Option` field, and is kept
/// on `Option` ones as documentation of the intent rather than for effect.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Default)]
pub struct IndexState {
    pub key_auth: Option<KeyAuth>,
    pub records: BTreeMap<SubjectId, SignedRecord>,
    /// Root-signed registry resolving `Locator::AppResource` to a live address.
    #[serde(default)]
    pub apps: Option<AppRegistry>,
}

/// Compact "what I have" for delta computation: per-subject versions plus the
/// key_auth and app-registry versions.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Default)]
pub struct IndexSummary {
    pub key_auth_version: u64,
    pub versions: BTreeMap<SubjectId, u64>,
    #[serde(default)]
    pub apps_version: u64,
}

/// The diff a peer needs to catch up: records newer than its summary, and a
/// newer key_auth or app registry if any.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Default)]
pub struct IndexDelta {
    pub key_auth: Option<KeyAuth>,
    pub records: Vec<SignedRecord>,
    #[serde(default)]
    pub apps: Option<AppRegistry>,
}

/// Total order used to resolve which record wins for a subject. Higher version
/// wins; at equal version a tombstone wins over a live entry (so a takedown is
/// never silently lost to a same-version entry); remaining ties break
/// deterministically on signature bytes so merge is commutative. Determinism of
/// the sig tie-break relies on signature non-malleability, which is why
/// verification uses `verify_strict` (see `crate::verify`).
fn record_order(r: &SignedRecord) -> (u64, bool, [u8; 64]) {
    let is_tomb = matches!(r.body, RecordBody::Tomb(_));
    (r.body.version(), is_tomb, r.sig.to_bytes())
}

impl IndexState {
    pub fn initialized(key_auth: KeyAuth) -> Self {
        IndexState {
            key_auth: Some(key_auth),
            records: BTreeMap::new(),
            apps: None,
        }
    }

    /// Full validity check (used by the contract's `validate_state`): key_auth is
    /// root-signed, the index is within bounds, and every record is correctly
    /// keyed, authorized, signed, and structurally valid.
    pub fn verify(&self, params: &IndexParams) -> Result<(), String> {
        let ka = self.key_auth.as_ref().ok_or("index has no key_auth")?;
        ka.verify_sig(&params.root_vk)?;
        if ka.body.authorized.len() > MAX_AUTHORIZED {
            return Err(format!(
                "key_auth authorizes too many keys: {}",
                ka.body.authorized.len()
            ));
        }
        if let Some(apps) = &self.apps {
            apps.verify_sig(&params.root_vk)?;
            apps.check_structure()?;
        }
        if self.records.len() > MAX_ENTRIES {
            return Err(format!("too many records: {}", self.records.len()));
        }
        for (sid, rec) in &self.records {
            if rec.body.subject_id() != sid {
                return Err("record stored under the wrong subject id".to_string());
            }
            if !ka.authorizes(&rec.by) {
                return Err("record signed by an unauthorized key".to_string());
            }
            rec.verify_sig()?;
            rec.body.check_structure()?;
        }
        Ok(())
    }

    /// Merge another already-sig-verified state into this one. Commutative,
    /// associative, idempotent.
    ///
    /// `key_auth` is immutable in 0.1: set once at init, never rotated, so every
    /// non-empty state for a contract carries the same root-signed `key_auth`.
    /// Records therefore merge as a plain per-subject max by `record_order` (a
    /// clean monoid). Authorization is enforced at the update boundary, not here:
    /// `apply_delta` only admits records authorized by the immutable `key_auth`,
    /// and `update_state` re-runs `verify` on the merged result. Making
    /// `key_auth` mutable here (rotation/revocation) would break associativity,
    /// because authorization is not monotone (a key can be de- then
    /// re-authorized), so dropping records against a non-final `key_auth` would
    /// depend on merge order. Rotation is deferred to a post-0.1 design.
    pub fn merge(&mut self, other: &IndexState) {
        // key_auth resolution is deterministic for ALL inputs, so convergence
        // does not rely on there being exactly one root-signed key_auth. Normally
        // there is (init runs once); but if the operator ever produced two
        // distinct root-signed key_auths, keeping the one with the smaller
        // signature bytes is a commutative+associative min (with None as
        // identity), so peers still converge instead of splitting. Records signed
        // only by the losing key_auth then fail the final verify in update_state,
        // surfacing the mistake rather than silently corrupting.
        match (&self.key_auth, &other.key_auth) {
            (None, Some(o)) => self.key_auth = Some(o.clone()),
            (Some(s), Some(o)) if o.sig.to_bytes() < s.sig.to_bytes() => {
                self.key_auth = Some(o.clone())
            }
            _ => {}
        }
        // App registry: plain max by (version, sig bytes) — commutative,
        // associative, idempotent, with None as the identity.
        if let Some(o) = &other.apps {
            let take = self
                .apps
                .as_ref()
                .is_none_or(|cur| registry_order(o) > registry_order(cur));
            if take {
                self.apps = Some(o.clone());
            }
        }
        for (sid, orec) in &other.records {
            let take = self
                .records
                .get(sid)
                .is_none_or(|cur| record_order(orec) > record_order(cur));
            if take {
                self.records.insert(sid.clone(), orec.clone());
            }
        }
    }

    /// Apply an untrusted delta (used by the contract's `update_state`): verify
    /// the key_auth and every record before merging it in.
    pub fn apply_delta(&mut self, delta: &IndexDelta, params: &IndexParams) -> Result<(), String> {
        if let Some(ka) = &delta.key_auth {
            ka.verify_sig(&params.root_vk)?;
            if ka.body.authorized.len() > MAX_AUTHORIZED {
                return Err(format!(
                    "key_auth authorizes too many keys: {}",
                    ka.body.authorized.len()
                ));
            }
            // key_auth is immutable in 0.1: adopt only to initialize from empty;
            // an identical re-send is idempotent; anything else (rotation) is
            // rejected. This keeps the merge a clean associative monoid.
            match &self.key_auth {
                None => self.key_auth = Some(ka.clone()),
                Some(cur) if cur == ka => {}
                Some(_) => return Err("key_auth is immutable in this version".to_string()),
            }
        }
        // The registry is root-signed and mutable, so a delta may legitimately
        // supersede it. Verify before adopting, and adopt only a strictly newer
        // one so replaying an old delta cannot roll the registry back.
        if let Some(apps) = &delta.apps {
            apps.verify_sig(&params.root_vk)?;
            apps.check_structure()?;
            let take = self
                .apps
                .as_ref()
                .is_none_or(|cur| registry_order(apps) > registry_order(cur));
            if take {
                self.apps = Some(apps.clone());
            }
        }
        let ka = self
            .key_auth
            .clone()
            .ok_or("no key_auth available to authorize records")?;
        for rec in &delta.records {
            if !ka.authorizes(&rec.by) {
                return Err("record signed by an unauthorized key".to_string());
            }
            rec.verify_sig()?;
            rec.body.check_structure()?;
            let sid = rec.body.subject_id().clone();
            let take = self
                .records
                .get(&sid)
                .is_none_or(|cur| record_order(rec) > record_order(cur));
            if take {
                self.records.insert(sid, rec.clone());
            }
        }
        if self.records.len() > MAX_ENTRIES {
            return Err(format!("too many records: {}", self.records.len()));
        }
        Ok(())
    }

    pub fn summarize(&self) -> IndexSummary {
        IndexSummary {
            key_auth_version: self.key_auth.as_ref().map_or(0, |k| k.body.version),
            versions: self
                .records
                .iter()
                .map(|(sid, rec)| (sid.clone(), rec.body.version()))
                .collect(),
            apps_version: self.apps.as_ref().map_or(0, |a| a.body.version),
        }
    }

    pub fn delta(&self, summary: &IndexSummary) -> IndexDelta {
        let key_auth = match &self.key_auth {
            Some(ka) if ka.body.version > summary.key_auth_version => Some(ka.clone()),
            _ => None,
        };
        let apps = match &self.apps {
            Some(a) if a.body.version > summary.apps_version => Some(a.clone()),
            _ => None,
        };
        let records = self
            .records
            .iter()
            .filter(|(sid, rec)| {
                summary
                    .versions
                    .get(*sid)
                    .is_none_or(|v| rec.body.version() > *v)
            })
            .map(|(_, rec)| rec.clone())
            .collect();
        IndexDelta {
            key_auth,
            records,
            apps,
        }
    }

    /// Resolve an entry's locator to a gateway-relative href, or `None` when it
    /// is an [`Locator::AppResource`] whose app the registry does not know yet.
    ///
    /// Lives here rather than in the UI so `atlasctl` and the UI agree on what a
    /// locator points at.
    pub fn resolve_href(&self, loc: &crate::types::Locator) -> Option<String> {
        use crate::types::Locator;
        match loc {
            Locator::External { url } => Some(url.clone()),
            Locator::Freenet { contract_id, path } => Some(contract_web_href(contract_id, path)),
            Locator::AppResource {
                app,
                resource,
                path,
            } => {
                let rec = self.apps.as_ref()?.get(app)?;
                Some(contract_web_href(
                    &rec.contract_id,
                    &rec.resolve(resource, path),
                ))
            }
        }
    }

    /// Live (non-tombstoned) entries, for clients rendering Discover/search.
    pub fn live_entries(&self) -> impl Iterator<Item = &IndexEntry> {
        self.records.values().filter_map(|r| match &r.body {
            RecordBody::Live(e) => Some(e),
            RecordBody::Tomb(_) => None,
        })
    }
}

/// Build `/v1/contract/web/<id><suffix>`, guaranteeing the slash the gateway's
/// cross-contract nav handler requires between the id and the suffix.
///
/// The gateway tolerates a slash-less form on a direct top-level load (it 308s),
/// but a click inside the sandboxed app iframe is validated against
/// `/v1/contract/web/<id>/` by the parent shell and is silently dropped without
/// it. A locator whose suffix is empty or starts with `#`/`?` would otherwise
/// produce exactly that.
pub fn contract_web_href(contract_id: &str, suffix: &str) -> String {
    if suffix.starts_with('/') {
        format!("/v1/contract/web/{contract_id}{suffix}")
    } else {
        format!("/v1/contract/web/{contract_id}/{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sign;
    use crate::types::{IndexEntry, KeyAuth, KeyAuthBody, Kind, Locator, Tombstone};
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn key() -> SigningKey {
        SigningKey::generate(&mut OsRng)
    }

    fn sid(n: u8) -> SubjectId {
        SubjectId::parse(&bs58::encode([n; 9]).into_string()).unwrap()
    }

    fn entry(sid: &SubjectId, version: u64, title: &str) -> IndexEntry {
        IndexEntry {
            subject_id: sid.clone(),
            version,
            kind: Kind::App,
            title: title.to_string(),
            snippet: "snippet".into(),
            tags: vec!["tag".into()],
            locator: Locator::External {
                url: "https://example.com".into(),
            },
            featured: false,
            added_at: 1,
        }
    }

    fn signed(body: RecordBody, online: &SigningKey) -> SignedRecord {
        let sig = sign(&body, online);
        SignedRecord {
            body,
            by: online.verifying_key(),
            sig,
        }
    }

    fn key_auth(root: &SigningKey, online: &SigningKey, version: u64) -> KeyAuth {
        let body = KeyAuthBody {
            version,
            authorized: vec![online.verifying_key()],
        };
        let sig = sign(&body, root);
        KeyAuth { body, sig }
    }

    fn params(root: &SigningKey) -> IndexParams {
        IndexParams {
            root_vk: root.verifying_key(),
            slug: "default".into(),
        }
    }

    #[test]
    fn empty_initialized_state_verifies() {
        let (root, online) = (key(), key());
        let s = IndexState::initialized(key_auth(&root, &online, 1));
        assert!(s.verify(&params(&root)).is_ok());
    }

    #[test]
    fn delta_application_and_verify() {
        let (root, online) = (key(), key());
        let p = params(&root);
        let mut s = IndexState::initialized(key_auth(&root, &online, 1));
        let id = sid(1);
        let rec = signed(RecordBody::Live(entry(&id, 1, "Hello")), &online);
        s.apply_delta(
            &IndexDelta {
                key_auth: None,
                records: vec![rec],
                apps: None,
            },
            &p,
        )
        .unwrap();
        assert!(s.verify(&p).is_ok());
        assert_eq!(s.live_entries().count(), 1);
    }

    #[test]
    fn merge_is_commutative() {
        let (root, online) = (key(), key());
        let ka = key_auth(&root, &online, 1);
        let a = signed(RecordBody::Live(entry(&sid(1), 1, "A")), &online);
        let b = signed(RecordBody::Live(entry(&sid(2), 1, "B")), &online);
        let c = signed(RecordBody::Live(entry(&sid(1), 2, "A2")), &online); // newer for sid 1

        let build = |recs: &[&SignedRecord]| {
            let mut s = IndexState::initialized(ka.clone());
            for r in recs {
                s.records.insert(r.body.subject_id().clone(), (*r).clone());
            }
            s
        };

        let mut left = build(&[&a, &b]);
        left.merge(&build(&[&c]));
        let mut right = build(&[&c]);
        right.merge(&build(&[&a, &b]));
        assert_eq!(left, right);
        // sid 1 resolves to the higher version regardless of order.
        let winner = left.records.get(&sid(1)).unwrap();
        assert_eq!(winner.body.version(), 2);
    }

    #[test]
    fn higher_version_wins() {
        let (root, online) = (key(), key());
        let p = params(&root);
        let mut s = IndexState::initialized(key_auth(&root, &online, 1));
        let id = sid(1);
        let v2 = signed(RecordBody::Live(entry(&id, 2, "v2")), &online);
        let v1 = signed(RecordBody::Live(entry(&id, 1, "v1")), &online);
        // apply newer then older; older must not win.
        s.apply_delta(
            &IndexDelta {
                key_auth: None,
                records: vec![v2],
                apps: None,
            },
            &p,
        )
        .unwrap();
        s.apply_delta(
            &IndexDelta {
                key_auth: None,
                records: vec![v1],
                apps: None,
            },
            &p,
        )
        .unwrap();
        assert_eq!(s.records.get(&id).unwrap().body.version(), 2);
    }

    #[test]
    fn tombstone_removes_from_live() {
        let (root, online) = (key(), key());
        let p = params(&root);
        let mut s = IndexState::initialized(key_auth(&root, &online, 1));
        let id = sid(1);
        let live = signed(RecordBody::Live(entry(&id, 1, "x")), &online);
        let tomb = signed(
            RecordBody::Tomb(Tombstone {
                subject_id: id.clone(),
                version: 2,
            }),
            &online,
        );
        s.apply_delta(
            &IndexDelta {
                key_auth: None,
                records: vec![live],
                apps: None,
            },
            &p,
        )
        .unwrap();
        assert_eq!(s.live_entries().count(), 1);
        s.apply_delta(
            &IndexDelta {
                key_auth: None,
                records: vec![tomb],
                apps: None,
            },
            &p,
        )
        .unwrap();
        assert_eq!(s.live_entries().count(), 0);
    }

    #[test]
    fn tampered_entry_is_rejected() {
        let (root, online) = (key(), key());
        let p = params(&root);
        let id = sid(1);
        let mut rec = signed(RecordBody::Live(entry(&id, 1, "original")), &online);
        // Tamper after signing.
        if let RecordBody::Live(e) = &mut rec.body {
            e.title = "tampered".into();
        }
        let mut s = IndexState::initialized(key_auth(&root, &online, 1));
        let err = s
            .apply_delta(
                &IndexDelta {
                    key_auth: None,
                    records: vec![rec],
                    apps: None,
                },
                &p,
            )
            .unwrap_err();
        assert!(err.contains("sig"), "expected sig error, got: {err}");
    }

    #[test]
    fn unauthorized_signer_is_rejected() {
        let (root, online, rogue) = (key(), key(), key());
        let p = params(&root);
        let id = sid(1);
        // rogue signs but is not in key_auth.authorized.
        let rec = signed(RecordBody::Live(entry(&id, 1, "x")), &rogue);
        let mut s = IndexState::initialized(key_auth(&root, &online, 1));
        let err = s
            .apply_delta(
                &IndexDelta {
                    key_auth: None,
                    records: vec![rec],
                    apps: None,
                },
                &p,
            )
            .unwrap_err();
        assert!(err.contains("unauthorized"), "got: {err}");
    }

    #[test]
    fn forged_key_auth_is_rejected() {
        let (root, online, attacker) = (key(), key(), key());
        let p = params(&root);
        let mut s = IndexState::initialized(key_auth(&root, &online, 1));
        // attacker tries to authorize their own key with a key_auth they signed.
        let forged = key_auth(&attacker, &attacker, 2);
        let err = s
            .apply_delta(
                &IndexDelta {
                    key_auth: Some(forged),
                    records: vec![],
                    apps: None,
                },
                &p,
            )
            .unwrap_err();
        assert!(err.contains("key_auth sig"), "got: {err}");
    }

    // --- app registry ---

    fn app_record(id: &str) -> crate::types::AppRecord {
        crate::types::AppRecord {
            contract_id: id.to_string(),
            name: "Delta".to_string(),
            link_template: "/#{resource}{path}".to_string(),
        }
    }

    fn registry(root: &SigningKey, version: u64, id: &str) -> AppRegistry {
        let mut body = crate::types::AppRegistryBody {
            version,
            ..Default::default()
        };
        body.apps.insert("delta".to_string(), app_record(id));
        let sig = sign(&body, root);
        AppRegistry { body, sig }
    }

    const ID_A: &str = "EqJ5YpEEV3XLqEvKWLQHFhGAac2qXzSUoE6k2zbdnXBr";
    const ID_B: &str = "771DvtPMwt2PumPyrFvsz7fpvU1gogcmb5qtS1yYEEH9";

    /// A re-pointed registry must win, and the merge must be order-independent:
    /// this is the whole reason an app republish is one edit and not N.
    #[test]
    fn newer_app_registry_wins_in_either_merge_order() {
        let (root, online) = (key(), key());
        let mut a = IndexState::initialized(key_auth(&root, &online, 1));
        a.apps = Some(registry(&root, 1, ID_A));
        let mut b = IndexState::initialized(key_auth(&root, &online, 1));
        b.apps = Some(registry(&root, 2, ID_B));

        let mut ab = a.clone();
        ab.merge(&b);
        let mut ba = b.clone();
        ba.merge(&a);
        assert_eq!(ab.apps, ba.apps, "registry merge must be commutative");
        assert_eq!(
            ab.apps.as_ref().unwrap().get("delta").unwrap().contract_id,
            ID_B,
            "the newer registry version must win"
        );
        // Idempotent.
        let mut again = ab.clone();
        again.merge(&b);
        assert_eq!(again.apps, ab.apps);
    }

    /// Replaying an OLD registry delta must not roll the registry back, or an
    /// app republish could be silently undone by stale traffic.
    #[test]
    fn an_older_registry_delta_does_not_roll_back() {
        let (root, online) = (key(), key());
        let params = params(&root);
        let mut st = IndexState::initialized(key_auth(&root, &online, 1));
        st.apps = Some(registry(&root, 5, ID_B));
        st.apply_delta(
            &IndexDelta {
                key_auth: None,
                records: vec![],
                apps: Some(registry(&root, 2, ID_A)),
            },
            &params,
        )
        .unwrap();
        assert_eq!(
            st.apps.as_ref().unwrap().get("delta").unwrap().contract_id,
            ID_B,
            "an older registry version must be ignored"
        );
    }

    /// The registry re-points every AppResource entry, so a non-root signature
    /// must be refused: an online-key compromise must not redirect Delta.
    #[test]
    fn a_registry_not_signed_by_root_is_rejected() {
        let (root, online) = (key(), key());
        let params = params(&root);
        let mut st = IndexState::initialized(key_auth(&root, &online, 1));
        // Signed by the ONLINE key, which is authorized for entries but not for
        // the registry.
        let forged = registry(&online, 1, ID_A);
        assert!(st
            .apply_delta(
                &IndexDelta {
                    key_auth: None,
                    records: vec![],
                    apps: Some(forged.clone()),
                },
                &params,
            )
            .is_err());
        // …and a state carrying it fails validate_state too.
        st.apps = Some(forged);
        assert!(st.verify(&params).is_err());
    }

    /// `atlasctl migrate` decodes a PRIOR generation's state with the CURRENT
    /// types. A state encoded before `apps` existed must still decode, or the
    /// migration silently loses the entire curated index.
    #[test]
    fn a_state_encoded_before_the_apps_field_still_decodes() {
        #[derive(Serialize)]
        struct LegacyIndexState {
            key_auth: Option<KeyAuth>,
            records: BTreeMap<SubjectId, SignedRecord>,
        }
        let (root, online) = (key(), key());
        let s = sid(1);
        let mut records = BTreeMap::new();
        records.insert(
            s.clone(),
            signed(RecordBody::Live(entry(&s, 1, "legacy")), &online),
        );
        let legacy = LegacyIndexState {
            key_auth: Some(key_auth(&root, &online, 1)),
            records,
        };
        let mut buf = Vec::new();
        ciborium::ser::into_writer(&legacy, &mut buf).unwrap();

        let decoded: IndexState = ciborium::de::from_reader(&buf[..]).unwrap();
        assert_eq!(decoded.records.len(), 1, "legacy records must survive");
        assert!(decoded.apps.is_none());
        assert!(decoded.verify(&params(&root)).is_ok());
    }

    /// Same for the summary: a peer running the older code sends a summary with
    /// no `apps_version`, and we must treat that as 0 (send it the registry)
    /// rather than failing to decode its summary at all.
    #[test]
    fn a_summary_encoded_before_apps_version_still_decodes() {
        #[derive(Serialize)]
        struct LegacySummary {
            key_auth_version: u64,
            versions: BTreeMap<SubjectId, u64>,
        }
        let legacy = LegacySummary {
            key_auth_version: 1,
            versions: BTreeMap::new(),
        };
        let mut buf = Vec::new();
        ciborium::ser::into_writer(&legacy, &mut buf).unwrap();
        let decoded: IndexSummary = ciborium::de::from_reader(&buf[..]).unwrap();
        assert_eq!(decoded.apps_version, 0);

        // …and the delta computed against it therefore carries the registry.
        let (root, online) = (key(), key());
        let mut st = IndexState::initialized(key_auth(&root, &online, 1));
        st.apps = Some(registry(&root, 3, ID_A));
        assert!(st.delta(&decoded).apps.is_some());
    }

    /// An entry pointing at an app the registry does not know is VALID but
    /// unresolvable. Rejecting it instead would make validity depend on merge
    /// order (entry before registry edit).
    #[test]
    fn an_unknown_app_is_valid_but_unresolvable() {
        let (root, online) = (key(), key());
        let params = params(&root);
        let s = sid(9);
        let mut e = entry(&s, 1, "a delta site");
        e.locator = Locator::AppResource {
            app: "delta".into(),
            resource: "AmcVD92D3U".into(),
            path: "/3/delta-sites".into(),
        };
        let loc = e.locator.clone();
        let mut st = IndexState::initialized(key_auth(&root, &online, 1));
        st.apply_delta(
            &IndexDelta {
                key_auth: None,
                records: vec![signed(RecordBody::Live(e), &online)],
                apps: None,
            },
            &params,
        )
        .expect("an entry naming an unregistered app must still be accepted");
        assert!(st.verify(&params).is_ok());
        assert!(st.resolve_href(&loc).is_none(), "not resolvable yet");

        // Registering the app makes every such entry resolve, with no entry edit.
        st.apps = Some(registry(&root, 1, ID_A));
        assert_eq!(
            st.resolve_href(&loc).unwrap(),
            format!("/v1/contract/web/{ID_A}/#AmcVD92D3U/3/delta-sites")
        );
    }

    /// The slash between the contract id and the suffix is load-bearing: without
    /// it a click inside the sandboxed iframe is silently dropped by the shell.
    #[test]
    fn resolved_hrefs_always_separate_the_id_from_the_suffix() {
        assert_eq!(
            contract_web_href(ID_A, "/#x"),
            format!("/v1/contract/web/{ID_A}/#x")
        );
        assert_eq!(
            contract_web_href(ID_A, "#x"),
            format!("/v1/contract/web/{ID_A}/#x")
        );
        assert_eq!(
            contract_web_href(ID_A, ""),
            format!("/v1/contract/web/{ID_A}/")
        );
    }

    #[test]
    fn params_roundtrip() {
        let root = key();
        let p = IndexParams {
            root_vk: root.verifying_key(),
            slug: "my-slug".into(),
        };
        let bytes = p.to_bytes();
        let back = IndexParams::from_bytes(&bytes).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn merge_is_associative() {
        let (root, online) = (key(), key());
        let ka = key_auth(&root, &online, 1);
        let mk = |recs: Vec<SignedRecord>| {
            let mut s = IndexState::initialized(ka.clone());
            for r in recs {
                s.records.insert(r.body.subject_id().clone(), r);
            }
            s
        };
        let a = mk(vec![signed(
            RecordBody::Live(entry(&sid(1), 1, "a1")),
            &online,
        )]);
        let b = mk(vec![
            signed(RecordBody::Live(entry(&sid(1), 2, "a2")), &online), // newer for sid 1
            signed(RecordBody::Live(entry(&sid(2), 1, "b")), &online),
        ]);
        let c = mk(vec![
            signed(
                RecordBody::Tomb(Tombstone {
                    subject_id: sid(2),
                    version: 2,
                }),
                &online,
            ),
            signed(RecordBody::Live(entry(&sid(3), 1, "c")), &online),
        ]);

        let mut left = a.clone();
        left.merge(&b);
        left.merge(&c);
        let mut bc = b.clone();
        bc.merge(&c);
        let mut right = a.clone();
        right.merge(&bc);
        assert_eq!(left, right, "merge must be associative");

        let mut ab = a.clone();
        ab.merge(&b);
        let mut ba = b.clone();
        ba.merge(&a);
        assert_eq!(ab, ba, "merge must be commutative");

        assert_eq!(left.records.get(&sid(1)).unwrap().body.version(), 2);
        assert_eq!(left.live_entries().count(), 2); // sid 1 + sid 3; sid 2 tombstoned
        assert!(left.verify(&params(&root)).is_ok());
    }

    #[test]
    fn key_auth_is_immutable() {
        let (root, k1, k2) = (key(), key(), key());
        let p = params(&root);
        let mut s = IndexState::initialized(key_auth(&root, &k1, 1));
        // A delta attempting to rotate to a different (even root-signed) key_auth
        // is rejected: key_auth is immutable in 0.1.
        let err = s
            .apply_delta(
                &IndexDelta {
                    key_auth: Some(key_auth(&root, &k2, 2)),
                    records: vec![],
                    apps: None,
                },
                &p,
            )
            .unwrap_err();
        assert!(err.contains("immutable"), "got: {err}");
        // Re-sending the identical key_auth is idempotent.
        s.apply_delta(
            &IndexDelta {
                key_auth: Some(key_auth(&root, &k1, 1)),
                records: vec![],
                apps: None,
            },
            &p,
        )
        .unwrap();
    }

    #[test]
    fn tombstone_wins_at_equal_version() {
        let (root, online) = (key(), key());
        let p = params(&root);
        let id = sid(1);
        let live = signed(RecordBody::Live(entry(&id, 5, "x")), &online);
        let tomb = signed(
            RecordBody::Tomb(Tombstone {
                subject_id: id.clone(),
                version: 5,
            }),
            &online,
        );
        for order in [[live.clone(), tomb.clone()], [tomb.clone(), live.clone()]] {
            let mut s = IndexState::initialized(key_auth(&root, &online, 1));
            for rec in order {
                s.apply_delta(
                    &IndexDelta {
                        key_auth: None,
                        records: vec![rec],
                        apps: None,
                    },
                    &p,
                )
                .unwrap();
            }
            assert_eq!(
                s.live_entries().count(),
                0,
                "tombstone wins at equal version regardless of arrival order"
            );
        }
    }

    #[test]
    fn two_keyauths_converge_deterministically() {
        // Operator-error case: two distinct root-signed key_auths. Even then,
        // empty->set adoption in merge must be order-independent so peers can't
        // split-brain.
        let (root, k1, k2) = (key(), key(), key());
        let a = IndexState::initialized(key_auth(&root, &k1, 1));
        let b = IndexState::initialized(key_auth(&root, &k2, 1));
        let mut left = IndexState::default();
        left.merge(&a);
        left.merge(&b);
        let mut right = IndexState::default();
        right.merge(&b);
        right.merge(&a);
        assert_eq!(
            left.key_auth, right.key_auth,
            "key_auth adoption must be order-independent even with two valid key_auths"
        );
    }
}
