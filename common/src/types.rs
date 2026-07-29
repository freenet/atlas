use std::collections::BTreeMap;

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::{MAX_SNIPPET, MAX_TAGS, MAX_TAG_LEN, MAX_TITLE};

/// Number of random bytes behind a subject id (~72 bits, ~12 base58 chars).
const SUBJECT_ID_BYTES: usize = 9;

/// Opaque, stable handle for a subject. Base58 over [`SUBJECT_ID_BYTES`] random
/// bytes. Deliberately not derived from any attribute, so it survives WASM
/// upgrades, URL changes, and owner re-keying.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SubjectId(String);

impl SubjectId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Parse and validate a base58 subject id (must decode to the right length).
    pub fn parse(s: &str) -> Option<Self> {
        let decoded = bs58::decode(s).into_vec().ok()?;
        if decoded.len() != SUBJECT_ID_BYTES {
            return None;
        }
        Some(SubjectId(s.to_string()))
    }

    /// Mint a fresh random subject id (native crates only).
    #[cfg(feature = "rng")]
    pub fn random() -> Self {
        use rand::RngCore;
        let mut bytes = [0u8; SUBJECT_ID_BYTES];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        SubjectId(bs58::encode(bytes).into_string())
    }

    fn is_well_formed(&self) -> bool {
        bs58::decode(&self.0)
            .into_vec()
            .map(|b| b.len() == SUBJECT_ID_BYTES)
            .unwrap_or(false)
    }
}

/// The 0.1 taxonomy. Constrained to things whose locator is directly openable;
/// richer kinds (Document, Media, Feed, Room) wait for per-kind Open semantics.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    App,
    Site,
    External,
}

/// Where "Open" navigates. An arbitrary URI; only the Freenet form has an
/// Atlas-defined shape. The path after the contract id is contract-defined and
/// opaque to Atlas.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub enum Locator {
    /// `contract_id` is the full 43-44 char base58 instance id; `path` is the
    /// suffix after it (leading `/`, query, `#fragment`), possibly empty.
    Freenet { contract_id: String, path: String },
    /// External web resource. Must be https.
    External { url: String },
    /// A resource *inside* an app rather than a contract of its own: a Delta
    /// site, a River room. `app` is a slug in the root-signed [`AppRegistry`],
    /// which supplies the app's current container contract id; `resource` is the
    /// app-defined handle for the thing itself (a Delta site's 10-char owner
    /// prefix); `path` is an optional deep link within it.
    ///
    /// This exists because a whole class of Freenet content is not
    /// contract-addressed. Every Delta site is served by the *same* Delta web
    /// container, so storing one as `Freenet { contract_id: <delta>, path:
    /// "/#<prefix>" }` makes the container id the identity, which collapses
    /// dedup across unrelated sites and re-points every entry whenever the app
    /// republishes. Keeping `(app, resource)` as the identity and resolving the
    /// address through the registry fixes both: a republish is one registry
    /// edit, not N entry rewrites.
    AppResource {
        app: String,
        resource: String,
        path: String,
    },
}

impl Locator {
    /// Structural validity. Mirrors the gateway's own retrieval facts: full id,
    /// no `..` path traversal, https-only externals.
    ///
    /// Note what is deliberately NOT checked for [`Locator::AppResource`]: that
    /// `app` is actually present in the registry. Requiring that would make
    /// validity depend on two independently-merging parts of the state, so an
    /// entry arriving before its registry edit would be rejected and the merge
    /// would stop being order-independent. An entry naming an unknown app is
    /// structurally valid and simply unresolvable until the registry catches up;
    /// the UI renders it as such.
    pub fn check(&self) -> Result<(), String> {
        match self {
            Locator::Freenet { contract_id, path } => {
                let n = contract_id.len();
                if n != 43 && n != 44 {
                    return Err(format!("contract id length {n} is not 43 or 44"));
                }
                if !contract_id.chars().all(is_base58_char) {
                    return Err("contract id has non-base58 chars".to_string());
                }
                check_path(path)
            }
            Locator::External { url } => {
                if !url.starts_with("https://") {
                    return Err("external locator must be https".to_string());
                }
                Ok(())
            }
            Locator::AppResource {
                app,
                resource,
                path,
            } => {
                check_app_slug(app)?;
                if resource.is_empty() || resource.len() > crate::MAX_RESOURCE {
                    return Err(format!("resource length {} out of range", resource.len()));
                }
                // Base58 keeps `{`, `}`, `/`, `#` and `?` out of a resource, so
                // substituting one into a link template can neither inject
                // another template placeholder nor escape the intended path
                // segment. `AppRecord::resolve` relies on that.
                if !resource.chars().all(is_base58_char) {
                    return Err("resource has non-base58 chars".to_string());
                }
                check_path(path)
            }
        }
    }

    /// Canonical string form (e.g. `freenet:<id><path>` or the external url).
    pub fn to_uri(&self) -> String {
        match self {
            Locator::Freenet { contract_id, path } => format!("freenet:{contract_id}{path}"),
            Locator::External { url } => url.clone(),
            Locator::AppResource {
                app,
                resource,
                path,
            } => format!("app:{app}/{resource}{path}"),
        }
    }

    /// The key two locators are considered "the same subject" under.
    ///
    /// For [`Locator::AppResource`] this deliberately DROPS `path`, so links to
    /// two different pages of one Delta site collapse to a single subject rather
    /// than becoming two listings. For the other variants it is the full URI,
    /// preserving existing behaviour (two paths under one web contract stay
    /// distinct, which is right when the contract is the publisher).
    pub fn dedup_key(&self) -> String {
        match self {
            Locator::AppResource { app, resource, .. } => format!("app:{app}/{resource}"),
            other => other.to_uri(),
        }
    }
}

/// Shared path rule for every locator variant: no `..` segment in the path,
/// query, or fragment, and a bounded length.
fn check_path(path: &str) -> Result<(), String> {
    if path.len() > crate::MAX_LOCATOR_PATH {
        return Err(format!("path length {} out of range", path.len()));
    }
    // Split off query/fragment first, then look for a traversal segment. The
    // fragment matters here as well as the path: an app-hosted locator routes
    // through the fragment, so a `..` there is a real traversal attempt on the
    // app's own router, not inert trailing text.
    if path.split(['?', '#']).any(|part| {
        part.split('/')
            .any(|seg| seg == ".." || seg.eq_ignore_ascii_case("%2e%2e"))
    }) {
        return Err("path contains a `..` segment".to_string());
    }
    Ok(())
}

/// An app slug is a short, lowercase, URL-safe identifier.
fn check_app_slug(app: &str) -> Result<(), String> {
    if app.is_empty() || app.len() > crate::MAX_APP_SLUG {
        return Err(format!("app slug length {} out of range", app.len()));
    }
    if !app
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err("app slug must be [a-z0-9-]".to_string());
    }
    Ok(())
}

/// Bitcoin-style base58 alphabet (excludes `0 O I l`).
fn is_base58_char(c: char) -> bool {
    matches!(c,
        '1'..='9' | 'A'..='H' | 'J'..='N' | 'P'..='Z' | 'a'..='k' | 'm'..='z')
}

/// A self-rendering entry: enough to draw a result card and detail view without
/// fetching anything else.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct IndexEntry {
    pub subject_id: SubjectId,
    pub version: u64,
    pub kind: Kind,
    pub title: String,
    pub snippet: String,
    pub tags: Vec<String>,
    pub locator: Locator,
    pub featured: bool,
    /// Unix seconds, set by the curator (contracts cannot read a clock).
    pub added_at: u64,
}

/// Removal marker. Wins over a live entry at the same subject once its version
/// is higher.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct Tombstone {
    pub subject_id: SubjectId,
    pub version: u64,
}

/// The signable body of a per-subject record.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub enum RecordBody {
    Live(IndexEntry),
    Tomb(Tombstone),
}

impl RecordBody {
    pub fn subject_id(&self) -> &SubjectId {
        match self {
            RecordBody::Live(e) => &e.subject_id,
            RecordBody::Tomb(t) => &t.subject_id,
        }
    }

    pub fn version(&self) -> u64 {
        match self {
            RecordBody::Live(e) => e.version,
            RecordBody::Tomb(t) => t.version,
        }
    }

    /// Structural checks independent of signatures.
    pub fn check_structure(&self) -> Result<(), String> {
        if self.version() == 0 {
            return Err("version must be >= 1".to_string());
        }
        if !self.subject_id().is_well_formed() {
            return Err("malformed subject id".to_string());
        }
        if let RecordBody::Live(e) = self {
            if e.title.is_empty() || e.title.len() > MAX_TITLE {
                return Err("title length out of range".to_string());
            }
            if e.snippet.len() > MAX_SNIPPET {
                return Err("snippet too long".to_string());
            }
            if e.tags.len() > MAX_TAGS || e.tags.iter().any(|t| t.len() > MAX_TAG_LEN) {
                return Err("too many tags or a tag is too long".to_string());
            }
            e.locator.check()?;
        }
        Ok(())
    }
}

/// A record signed by an online signing key (which must chain to the root key
/// via [`KeyAuth`]).
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct SignedRecord {
    pub body: RecordBody,
    pub by: VerifyingKey,
    pub sig: Signature,
}

impl SignedRecord {
    /// Verify the signature over the body. Does not check authorization; the
    /// caller checks `by` against the current [`KeyAuth`].
    pub fn verify_sig(&self) -> Result<(), String> {
        crate::verify(&self.body, &self.sig, &self.by).map_err(|e| format!("bad record sig: {e}"))
    }
}

/// Root-signed authorization of the online signing keys. Merges last-write-wins
/// by `version`, so the root can rotate or revoke online keys without changing
/// the contract address.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct KeyAuthBody {
    pub version: u64,
    pub authorized: Vec<VerifyingKey>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct KeyAuth {
    pub body: KeyAuthBody,
    /// Signature by `root_vk` (from the contract parameters) over the body.
    pub sig: Signature,
}

impl KeyAuth {
    pub fn verify_sig(&self, root_vk: &VerifyingKey) -> Result<(), String> {
        crate::verify(&self.body, &self.sig, root_vk).map_err(|e| format!("bad key_auth sig: {e}"))
    }

    pub fn authorizes(&self, key: &VerifyingKey) -> bool {
        self.body.authorized.iter().any(|k| k == key)
    }
}

/// One registered app: where its container currently lives, and how to turn a
/// `(resource, path)` pair into a URL under it.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct AppRecord {
    /// The app's CURRENT web-container instance id (43-44 char base58). This is
    /// the field that moves when the app republishes, and the reason the
    /// registry exists.
    pub contract_id: String,
    /// Display name, e.g. "Delta".
    pub name: String,
    /// Path suffix template appended after the contract id, with `{resource}`
    /// and `{path}` substituted. Delta is `/#{resource}{path}`.
    ///
    /// A template (rather than app-specific code in every client) keeps URL
    /// construction data-driven, so adding an app is a curator action rather
    /// than a UI release.
    pub link_template: String,
}

impl AppRecord {
    /// Structural checks. `link_template` is the security-relevant one: it is
    /// concatenated after `/v1/contract/web/<id>` in a client, so it must not be
    /// able to change the origin or escape the contract's web root.
    pub fn check(&self) -> Result<(), String> {
        let n = self.contract_id.len();
        if n != 43 && n != 44 {
            return Err(format!("app contract id length {n} is not 43 or 44"));
        }
        if !self.contract_id.chars().all(is_base58_char) {
            return Err("app contract id has non-base58 chars".to_string());
        }
        if self.name.is_empty() || self.name.len() > crate::MAX_APP_NAME {
            return Err(format!("app name length {} out of range", self.name.len()));
        }
        let t = &self.link_template;
        if t.is_empty() || t.len() > crate::MAX_LINK_TEMPLATE {
            return Err(format!("link template length {} out of range", t.len()));
        }
        if !t.starts_with('/') {
            return Err("link template must start with `/`".to_string());
        }
        // `//host` is protocol-relative: appended to a gateway origin it would
        // still be same-origin, but as a bare href it retargets to another host.
        // Refuse it rather than reason about every context it lands in.
        if t.starts_with("//") {
            return Err("link template must not start with `//`".to_string());
        }
        if !t.contains("{resource}") {
            return Err("link template must contain `{resource}`".to_string());
        }
        if t.contains(':') {
            return Err("link template must not contain `:`".to_string());
        }
        check_path(t)?;
        // Any placeholder other than the two we substitute would survive into
        // the emitted URL, so reject unknown ones instead of shipping a literal
        // `{foo}` to the gateway.
        let residue = t.replace("{resource}", "").replace("{path}", "");
        if residue.contains('{') || residue.contains('}') {
            return Err("link template has an unknown `{...}` placeholder".to_string());
        }
        Ok(())
    }

    /// Build the path suffix (everything after `/v1/contract/web/<contract_id>`)
    /// for a resource. Callers must have validated the locator, which guarantees
    /// `resource` is base58 and so cannot inject a placeholder or a separator.
    pub fn resolve(&self, resource: &str, path: &str) -> String {
        self.link_template
            .replace("{resource}", resource)
            .replace("{path}", path)
    }
}

/// The signable body of the app registry. Mutable, unlike [`KeyAuthBody`]:
/// merges last-write-wins by `version`, so the curator can re-point an app after
/// it republishes without changing the index's own address.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Default)]
pub struct AppRegistryBody {
    pub version: u64,
    /// Slug -> record. A `BTreeMap` so the CBOR signing payload is deterministic.
    pub apps: BTreeMap<String, AppRecord>,
}

/// Root-signed app registry.
///
/// Root-signed (not online-key-signed) on purpose: one registry row decides
/// where EVERY `AppResource` entry for that app navigates, so it carries more
/// authority than any single listing. Compromising an online key must not let an
/// attacker re-point Delta at a contract they control.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct AppRegistry {
    pub body: AppRegistryBody,
    /// Signature by `root_vk` (from the contract parameters) over the body.
    pub sig: Signature,
}

impl AppRegistry {
    pub fn verify_sig(&self, root_vk: &VerifyingKey) -> Result<(), String> {
        crate::verify(&self.body, &self.sig, root_vk)
            .map_err(|e| format!("bad app registry sig: {e}"))
    }

    pub fn check_structure(&self) -> Result<(), String> {
        if self.body.version == 0 {
            return Err("app registry version must be >= 1".to_string());
        }
        if self.body.apps.len() > crate::MAX_APPS {
            return Err(format!("too many apps: {}", self.body.apps.len()));
        }
        for (slug, rec) in &self.body.apps {
            check_app_slug(slug)?;
            rec.check()?;
        }
        Ok(())
    }

    pub fn get(&self, app: &str) -> Option<&AppRecord> {
        self.body.apps.get(app)
    }
}

/// Contract parameters: the index's identity. Fixed byte layout (not serde) so
/// a dependency bump can never silently re-key the live index.
/// Layout: `root_vk` (32 bytes) `||` `slug` (UTF-8).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IndexParams {
    pub root_vk: VerifyingKey,
    pub slug: String,
}

impl IndexParams {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.slug.len());
        out.extend_from_slice(self.root_vk.as_bytes());
        out.extend_from_slice(self.slug.as_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 32 {
            return None;
        }
        let vk_bytes: [u8; 32] = bytes[..32].try_into().ok()?;
        let root_vk = VerifyingKey::from_bytes(&vk_bytes).ok()?;
        let slug = String::from_utf8(bytes[32..].to_vec()).ok()?;
        Some(IndexParams { root_vk, slug })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "EqJ5YpEEV3XLqEvKWLQHFhGAac2qXzSUoE6k2zbdnXBr";

    fn app_res(app: &str, resource: &str, path: &str) -> Locator {
        Locator::AppResource {
            app: app.to_string(),
            resource: resource.to_string(),
            path: path.to_string(),
        }
    }

    fn record(template: &str) -> AppRecord {
        AppRecord {
            contract_id: ID.to_string(),
            name: "Delta".to_string(),
            link_template: template.to_string(),
        }
    }

    #[test]
    fn app_resource_accepts_a_delta_site_link() {
        assert!(app_res("delta", "AmcVD92D3U", "").check().is_ok());
        assert!(app_res("delta", "AmcVD92D3U", "/3/delta-sites")
            .check()
            .is_ok());
    }

    #[test]
    fn app_resource_rejects_a_bad_slug() {
        for bad in ["", "Delta", "del ta", "del_ta", "delta!"] {
            assert!(
                app_res(bad, "AmcVD92D3U", "").check().is_err(),
                "slug {bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn app_resource_rejects_a_non_base58_resource() {
        // `{`/`}` would let a resource inject a template placeholder, and `/#?`
        // would let it escape its path segment. Base58 excludes all of them, and
        // `AppRecord::resolve` relies on that, so pin it.
        for bad in ["", "has/slash", "has#frag", "has?q", "{resource}", "O0Il"] {
            assert!(
                app_res("delta", bad, "").check().is_err(),
                "resource {bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn a_traversal_in_an_app_path_or_fragment_is_rejected() {
        // The fragment is the routing surface for an app-hosted locator, so a
        // `..` there is a real traversal attempt, not inert trailing text.
        for bad in [
            "/../secret",
            "/a/../../b",
            "#/../x",
            "?next=/../x",
            "/%2e%2e/x",
        ] {
            assert!(
                app_res("delta", "AmcVD92D3U", bad).check().is_err(),
                "path {bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn dedup_key_collapses_pages_of_one_app_resource() {
        // Two links to different pages of ONE Delta site must be one subject.
        let a = app_res("delta", "AmcVD92D3U", "/1/home");
        let b = app_res("delta", "AmcVD92D3U", "/3/delta-sites");
        let c = app_res("delta", "AmcVD92D3U", "");
        assert_eq!(a.dedup_key(), b.dedup_key());
        assert_eq!(a.dedup_key(), c.dedup_key());
        // …but two DIFFERENT sites stay distinct, even though they share the
        // same container contract.
        let other = app_res("delta", "Fe5jaFmRnp", "/1/about");
        assert_ne!(a.dedup_key(), other.dedup_key());
    }

    #[test]
    fn dedup_key_keeps_paths_distinct_for_a_web_contract() {
        // Unchanged behaviour for contract-addressed locators: there the
        // contract IS the publisher, so two paths are two things.
        let a = Locator::Freenet {
            contract_id: ID.to_string(),
            path: "/a".to_string(),
        };
        let b = Locator::Freenet {
            contract_id: ID.to_string(),
            path: "/b".to_string(),
        };
        assert_ne!(a.dedup_key(), b.dedup_key());
    }

    #[test]
    fn app_resource_uri_round_trips_through_to_uri() {
        assert_eq!(
            app_res("delta", "AmcVD92D3U", "/3/delta-sites").to_uri(),
            "app:delta/AmcVD92D3U/3/delta-sites"
        );
        assert_eq!(
            app_res("delta", "AmcVD92D3U", "").to_uri(),
            "app:delta/AmcVD92D3U"
        );
    }

    #[test]
    fn link_template_must_be_a_same_origin_relative_path() {
        // Each of these could retarget or escape the app's web root once
        // concatenated after /v1/contract/web/<id>.
        for bad in [
            "#{resource}",               // no leading slash
            "//evil.example/{resource}", // protocol-relative
            "https://evil.example/{resource}",
            "/x:{resource}",      // scheme-ish
            "/{resource}/../..",  // traversal
            "/#{res}",            // missing {resource}
            "/#{resource}{oops}", // unknown placeholder survives into the URL
            "",
        ] {
            assert!(
                record(bad).check().is_err(),
                "template {bad:?} should be rejected"
            );
        }
        assert!(record("/#{resource}{path}").check().is_ok());
        assert!(record("/{resource}").check().is_ok());
    }

    #[test]
    fn resolve_substitutes_resource_and_path() {
        let r = record("/#{resource}{path}");
        assert_eq!(r.resolve("AmcVD92D3U", "/3/x"), "/#AmcVD92D3U/3/x");
        assert_eq!(r.resolve("AmcVD92D3U", ""), "/#AmcVD92D3U");
    }

    #[test]
    fn registry_rejects_a_zero_version_or_bad_member() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;
        let root = SigningKey::generate(&mut OsRng);
        let mut body = AppRegistryBody::default();
        body.apps
            .insert("delta".into(), record("/#{resource}{path}"));
        // version 0 is the "unset" sentinel and must never appear signed.
        let sig = crate::sign(&body, &root);
        let reg = AppRegistry { body, sig };
        assert!(reg.check_structure().is_err());

        // A bad slug inside an otherwise-valid registry is caught too.
        let mut body = AppRegistryBody {
            version: 1,
            ..Default::default()
        };
        body.apps.insert("BAD".into(), record("/#{resource}{path}"));
        let sig = crate::sign(&body, &root);
        let reg = AppRegistry { body, sig };
        assert!(reg.check_structure().is_err());
    }
}
