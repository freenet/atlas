//! Duplicate and clone collapsing (issue #68).
//!
//! A copy of a site should not be listed next to the original, or instead of it.
//! Three cases, and the signal that catches each:
//!
//! - **An author republishes their own site** under a second contract id, often
//!   by accident. Both contracts carry the SAME owner key, because a web
//!   container's parameters ARE its owner's verifying key. Merged
//!   unconditionally: one publisher, one listing.
//! - **A naive clone** republishes someone else's archive byte for byte under the
//!   cloner's own key. The owner keys differ but the archive bytes are identical.
//! - **A scam clone** copies a site and changes a detail, such as the Bitcoin
//!   address on a seller page. Exact matching cannot see this, so the rendered
//!   content text is compared as a set of word shingles and a near-identical
//!   pair is a duplicate.
//!
//! When a candidate matches an entry already in the live index, that entry is
//! CANONICAL (first seen wins) and the candidate is not listed. This is
//! deliberately the weakest claim that works: a clone that happened to be indexed
//! first keeps the listing. Outranking first-seen needs a publisher identity the
//! site itself declares (step 2 of #68), and the store keeps `first_seen` and the
//! owner key per entry precisely so that such a claim has something to outrank.
//!
//! What is NEVER compared, because the signals would be wrong rather than weak:
//!
//! - Two locators on the SAME contract id. `freenet:<id>/a.html` and
//!   `freenet:<id>/b.html` are different pages of one container: same owner and
//!   same archive by construction.
//! - Two `app:<slug>/…` resources of the SAME app. Every Delta site is served by
//!   the one Delta container, so they would all look like one publisher's one
//!   archive. Their text is not compared either: distinct sites sharing one host
//!   app must never be merged on that host's chrome.
//!
//! This module is pure: no network, no subprocess. `main.rs` fetches the owner
//! key and archive hash (`atlasctl contract-info`) and the rendered text, and
//! hands them here.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

/// Words per shingle.
///
/// Five is long enough that two unrelated pages rarely share a shingle by
/// accident (common short phrases like "on freenet" fall below it), and short
/// enough that a one-token edit, such as a swapped address, disturbs only five
/// shingles out of hundreds.
pub const SHINGLE_WORDS: usize = 5;

/// How many shingle hashes a sketch keeps: the numerically smallest `K`.
///
/// A page with fewer distinct shingles than this is kept whole, so the
/// similarity of two small pages is EXACT rather than estimated. For larger pages
/// the bottom-k estimator's standard error near the threshold is about
/// `sqrt(J(1-J)/K)`, roughly 0.02 at J = 0.9.
pub const SKETCH_K: usize = 256;

/// Fewer distinct shingles than this and the page carries no text signal at all.
///
/// Below it, two unrelated pages that share a boilerplate sentence or two (a
/// login prompt, a "loading" notice) could clear the threshold on that alone.
/// The owner and archive signals still apply to such a page.
pub const MIN_SHINGLES: usize = 30;

/// Estimated Jaccard similarity at or above which two pages are the same content.
///
/// Set from a measurement, not a guess: see the PR for #68, which reports the
/// highest similarity between any two DISTINCT live entries. A scam clone that
/// swaps one address in a few hundred words stays well above it.
pub const NEAR_DUP_JACCARD: f64 = 0.80;

/// A bottom-k sketch of a page's word-shingle set: the `SKETCH_K` smallest
/// shingle hashes, sorted ascending and distinct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sketch(Vec<u64>);

impl Sketch {
    /// Sketch `text`, or `None` if it has fewer than [`MIN_SHINGLES`] distinct
    /// shingles and so cannot support a similarity verdict.
    ///
    /// Words are maximal runs of alphanumeric characters, lowercased, so layout,
    /// punctuation and case do not count as differences. Digits are kept: an
    /// address or a price is part of what a page says.
    pub fn of(text: &str) -> Option<Self> {
        let words: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect();
        if words.len() < SHINGLE_WORDS {
            return None;
        }
        let mut hashes: Vec<u64> = words
            .windows(SHINGLE_WORDS)
            .map(|w| shingle_hash(&w.join(" ")))
            .collect();
        hashes.sort_unstable();
        hashes.dedup();
        if hashes.len() < MIN_SHINGLES {
            return None;
        }
        hashes.truncate(SKETCH_K);
        Some(Self(hashes))
    }

    /// Estimated Jaccard similarity of the two underlying shingle sets.
    ///
    /// The bottom-k estimator: take the `K` smallest hashes of the UNION of the
    /// two sketches and count how many of them are in both. That count is exact
    /// for sets smaller than `K`, because the sketches then hold the whole sets.
    pub fn jaccard(&self, other: &Self) -> f64 {
        let (a, b) = (&self.0, &other.0);
        let (mut i, mut j) = (0, 0);
        let (mut union, mut both) = (0usize, 0usize);
        while union < SKETCH_K && (i < a.len() || j < b.len()) {
            match (a.get(i), b.get(j)) {
                (Some(x), Some(y)) if x == y => {
                    both += 1;
                    i += 1;
                    j += 1;
                }
                (Some(x), Some(y)) if x < y => i += 1,
                (Some(_), Some(_)) => j += 1,
                // One sketch is exhausted. Its set holds nothing further at or
                // below its largest kept hash, but a SKETCH_K-full sketch may hold
                // more above it, so the union beyond this point is unknown and
                // counting further would bias the estimate low. Stop.
                (Some(_), None) if b.len() == SKETCH_K => break,
                (None, Some(_)) if a.len() == SKETCH_K => break,
                (Some(_), None) => i += 1,
                (None, Some(_)) => j += 1,
                (None, None) => unreachable!("loop condition"),
            }
            union += 1;
        }
        if union == 0 {
            0.0
        } else {
            both as f64 / union as f64
        }
    }

    fn encode(&self) -> String {
        self.0.iter().map(|h| format!("{h:016x}")).collect()
    }

    fn decode(s: &str) -> Option<Self> {
        if s.is_empty() || !s.len().is_multiple_of(16) || !s.is_ascii() {
            return None;
        }
        let v = (0..s.len())
            .step_by(16)
            .map(|i| u64::from_str_radix(&s[i..i + 16], 16).ok())
            .collect::<Option<Vec<u64>>>()?;
        // Re-establish the invariant rather than trust the file: `jaccard`
        // assumes sorted, distinct, at most SKETCH_K.
        let ok = v.len() <= SKETCH_K && v.windows(2).all(|w| w[0] < w[1]);
        ok.then_some(Self(v))
    }
}

/// FNV-1a with a SplitMix64 finalizer. Stable across Rust releases because it is
/// PERSISTED (see `fnv1a64` in `main.rs` for why `DefaultHasher` is not), and
/// finalized because the bottom-k sketch keeps the SMALLEST hashes, which is only
/// a uniform sample if the low values are well mixed.
fn shingle_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

/// What is known about one locator for duplicate detection. Every field is
/// optional: a missing one is "unknown", which never matches anything.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fingerprint {
    /// The contract's 32-byte owner verifying key, lowercase hex. Only set for a
    /// `freenet:` locator whose parameters are exactly 32 bytes (the web
    /// container's shape). Parameters of any other length encode something else
    /// (the Atlas index's are `root_vk || slug`), and treating equal bytes there
    /// as "same publisher" would be a guess.
    pub owner: Option<String>,
    /// BLAKE3 of the web archive, lowercase hex, excluding the signed metadata.
    pub archive: Option<String>,
    pub sketch: Option<Sketch>,
}

impl Fingerprint {
    /// Parse `atlasctl contract-info` output into the owner and archive fields.
    pub fn from_contract_info(json: &serde_json::Value) -> Self {
        let owner = json["params_hex"]
            .as_str()
            .filter(|p| p.len() == 64 && p.bytes().all(|b| b.is_ascii_hexdigit()))
            .map(str::to_ascii_lowercase);
        let archive = json["archive_blake3"]
            .as_str()
            .filter(|a| a.len() == 64 && a.bytes().all(|b| b.is_ascii_hexdigit()))
            .map(str::to_ascii_lowercase);
        Self {
            owner,
            archive,
            sketch: None,
        }
    }
}

/// One stored fingerprint: a locator that was indexed, and when.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stored {
    pub locator: String,
    /// When the locator was first indexed (the entry's `added_at` for entries
    /// backfilled from the live index). First seen is canonical.
    pub first_seen: u64,
    pub print: Fingerprint,
}

/// Which signal matched. Ordered strongest first, which is the order a match is
/// reported in when several agree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Signal {
    SameOwner,
    SameArchive,
    NearText(f64),
}

impl Signal {
    /// The decision-log reason fragment. Stable, like `Outcome::token`: an
    /// operator greps `duplicate-of` lines by it.
    pub fn reason(&self) -> String {
        match self {
            Self::SameOwner => "same-owner".to_string(),
            Self::SameArchive => "same-archive".to_string(),
            Self::NearText(j) => format!("near-text jaccard={j:.3}"),
        }
    }
}

/// The canonical entry a candidate duplicates, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct Match {
    pub canonical: String,
    pub signal: Signal,
}

/// The contract id a `freenet:` locator names, or `None`.
fn contract_of(loc: &str) -> Option<&str> {
    let rest = loc.strip_prefix("freenet:")?;
    Some(rest.split(['/', '?', '#']).next().unwrap_or(rest))
}

/// The app slug an `app:<slug>/<resource>` locator names, or `None`.
fn app_of(loc: &str) -> Option<&str> {
    loc.strip_prefix("app:")?.split('/').next()
}

/// Whether two locators may be compared at all. See the module doc.
pub fn comparable(a: &str, b: &str) -> bool {
    if a == b {
        return false;
    }
    if let (Some(x), Some(y)) = (contract_of(a), contract_of(b)) {
        if x == y {
            return false;
        }
    }
    if let (Some(x), Some(y)) = (app_of(a), app_of(b)) {
        if x == y {
            return false;
        }
    }
    true
}

/// The strongest signal on which `a` and `b` are the same site, if any. Assumes
/// [`comparable`] has already said yes.
pub fn signal(a: &Fingerprint, b: &Fingerprint) -> Option<Signal> {
    if a.owner.is_some() && a.owner == b.owner {
        return Some(Signal::SameOwner);
    }
    if a.archive.is_some() && a.archive == b.archive {
        return Some(Signal::SameArchive);
    }
    if let (Some(x), Some(y)) = (&a.sketch, &b.sketch) {
        let j = x.jaccard(y);
        if j >= NEAR_DUP_JACCARD {
            return Some(Signal::NearText(j));
        }
    }
    None
}

/// The live entry `candidate` duplicates, if any: the EARLIEST-indexed stored
/// entry that is still live, comparable, and matches on some signal.
///
/// `live` is the set of locators currently in the index. A stored entry that has
/// since left the index is not canonical: otherwise a removed original (a dead
/// site, a curator's removal) would block every later copy for ever, including
/// the author's own replacement.
pub fn find_canonical(
    loc: &str,
    candidate: &Fingerprint,
    store: &[Stored],
    live: &HashSet<String>,
) -> Option<Match> {
    store
        .iter()
        .filter(|s| live.contains(&s.locator) && comparable(loc, &s.locator))
        .filter_map(|s| signal(candidate, &s.print).map(|sig| (s, sig)))
        .min_by(|(a, _), (b, _)| (a.first_seen, &a.locator).cmp(&(b.first_seen, &b.locator)))
        .map(|(s, sig)| Match {
            canonical: s.locator.clone(),
            signal: sig,
        })
}

/// How many of the closest text pairs below the threshold `report` lists.
const REPORT_NEAREST: usize = 20;

/// A curator's view of `entries` as the duplicate check sees them: every pair it
/// would call the same site (with the canonical, earlier one first), then the
/// closest text pairs it would NOT, which is the false-positive margin.
pub fn report(entries: &[Stored]) -> String {
    let mut dups: Vec<(&Stored, &Stored, Signal)> = Vec::new();
    let mut near: Vec<(f64, &Stored, &Stored)> = Vec::new();
    let mut compared = 0usize;
    for (i, a) in entries.iter().enumerate() {
        for b in &entries[i + 1..] {
            if !comparable(&a.locator, &b.locator) {
                continue;
            }
            compared += 1;
            let (first, second) = if (a.first_seen, &a.locator) <= (b.first_seen, &b.locator) {
                (a, b)
            } else {
                (b, a)
            };
            match signal(&second.print, &first.print) {
                Some(sig) => dups.push((first, second, sig)),
                None => {
                    if let (Some(x), Some(y)) = (&a.print.sketch, &b.print.sketch) {
                        near.push((x.jaccard(y), first, second));
                    }
                }
            }
        }
    }
    let mut out = format!(
        "{compared} comparable pairs; {} would be duplicates (threshold jaccard >= {NEAR_DUP_JACCARD})\n",
        dups.len()
    );
    for (first, second, sig) in &dups {
        out.push_str(&format!(
            "  DUPLICATE {} of {} ({})\n",
            second.locator,
            first.locator,
            sig.reason()
        ));
    }
    near.sort_by(|x, y| y.0.total_cmp(&x.0));
    out.push_str(&format!(
        "closest {} text pairs below the threshold:\n",
        REPORT_NEAREST.min(near.len())
    ));
    for (j, a, b) in near.iter().take(REPORT_NEAREST) {
        out.push_str(&format!("  {j:.3}  {}  {}\n", a.locator, b.locator));
    }
    out
}

/// The fingerprint store: `crawler-fingerprints.txt`, one line per indexed
/// locator, `locator \t first_seen \t owner \t archive \t sketch`, `-` for an
/// unknown field.
///
/// Crawler-local, like the recheck schedule: never published. A malformed line
/// is skipped with a warning rather than failing the load, because losing one
/// fingerprint only means one fewer entry a clone can be caught against, while
/// refusing to run would stop the whole crawl.
pub struct Store {
    pub entries: Vec<Stored>,
}

impl Store {
    pub fn load(path: &Path) -> Self {
        let body = fs::read_to_string(path).unwrap_or_default();
        let mut entries: Vec<Stored> = Vec::new();
        for line in body.lines().filter(|l| !l.trim().is_empty()) {
            match parse_line(line) {
                Some(s) => {
                    // Later lines win: a re-fingerprint is appended, not edited in.
                    entries.retain(|e| e.locator != s.locator);
                    entries.push(s);
                }
                None => eprintln!(
                    "warn: skipping malformed fingerprint line in {}",
                    path.display()
                ),
            }
        }
        Self { entries }
    }

    /// Add or replace one locator's fingerprint and append it to `path`.
    pub fn record(&mut self, path: &Path, s: Stored) -> Result<()> {
        let line = format_line(&s);
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        use std::io::Write;
        writeln!(f, "{line}").with_context(|| format!("appending to {}", path.display()))?;
        self.entries.retain(|e| e.locator != s.locator);
        self.entries.push(s);
        Ok(())
    }

    /// Rewrite the whole file from `entries`, atomically.
    pub fn save(&self, path: &Path, tmp: &Path) -> Result<()> {
        let body: String = self
            .entries
            .iter()
            .map(|s| format!("{}\n", format_line(s)))
            .collect();
        fs::write(tmp, body).with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(tmp, path).with_context(|| format!("renaming onto {}", path.display()))
    }
}

fn opt(s: &Option<String>) -> &str {
    s.as_deref().unwrap_or("-")
}

fn format_line(s: &Stored) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}",
        s.locator,
        s.first_seen,
        opt(&s.print.owner),
        opt(&s.print.archive),
        s.print
            .sketch
            .as_ref()
            .map(Sketch::encode)
            .unwrap_or_else(|| "-".to_string())
    )
}

fn parse_line(line: &str) -> Option<Stored> {
    let f: Vec<&str> = line.split('\t').collect();
    let [locator, first_seen, owner, archive, sketch] = f[..] else {
        return None;
    };
    let field = |s: &str| (s != "-").then(|| s.to_string());
    Some(Stored {
        locator: locator.to_string(),
        first_seen: first_seen.parse().ok()?,
        print: Fingerprint {
            owner: field(owner),
            archive: field(archive),
            sketch: if sketch == "-" {
                None
            } else {
                Some(Sketch::decode(sketch)?)
            },
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A few hundred words of prose, standing in for a seller page.
    fn seller_page(address: &str) -> String {
        let mut s = String::from(
            "Welcome to Meadowbrook Farm on Freenet. We grow heirloom tomatoes, \
             sweet corn and winter squash without pesticides, and ship seeds to \
             anywhere the post office reaches. ",
        );
        for i in 0..40 {
            s.push_str(&format!(
                "Lot {i}: a packet of variety number {i} seeds, harvested this \
                 season and dried by hand in the barn loft. "
            ));
        }
        s.push_str(&format!(
            "To order, send payment in bitcoin to {address} and post your \
             delivery details in our River room. Thank you for supporting \
             small farms."
        ));
        s
    }

    fn fp_text(t: &str) -> Fingerprint {
        Fingerprint {
            sketch: Sketch::of(t),
            ..Default::default()
        }
    }

    fn live(locs: &[&str]) -> HashSet<String> {
        locs.iter().map(|s| s.to_string()).collect()
    }

    const A: &str = "freenet:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/";
    const B: &str = "freenet:BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB/";
    const C: &str = "freenet:CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC/";

    #[test]
    fn a_scam_clone_that_swaps_the_payment_address_is_a_near_duplicate() {
        let orig = fp_text(&seller_page("bc1qoriginalsellerxxxxxxxxxxxxxxxx"));
        let clone = fp_text(&seller_page("bc1qscammerzzzzzzzzzzzzzzzzzzzzzz"));
        let j = orig
            .sketch
            .as_ref()
            .unwrap()
            .jaccard(clone.sketch.as_ref().unwrap());
        assert!(j >= NEAR_DUP_JACCARD, "jaccard {j}");
        assert!(j < 1.0, "the address change must be visible, got {j}");
        let store = vec![Stored {
            locator: A.into(),
            first_seen: 100,
            print: orig,
        }];
        let m = find_canonical(B, &clone, &store, &live(&[A])).expect("clone caught");
        assert_eq!(m.canonical, A);
        assert!(matches!(m.signal, Signal::NearText(_)));
    }

    #[test]
    fn distinct_pages_with_shared_boilerplate_are_not_duplicates() {
        let chrome = "Home About Contact Powered by Freenet. This site is \
                      published on a decentralized network and cannot be \
                      censored. ";
        let a = format!(
            "{chrome}{}",
            "Recipes for sourdough bread, rye crackers and a slow-fermented \
             focaccia, with notes on hydration, flour choice and oven heat. \
             Each recipe lists weights in grams and timings for a cold kitchen. \
             The starter section explains feeding ratios and how to revive a \
             neglected jar after a month in the fridge."
        );
        let b = format!(
            "{chrome}{}",
            "Notes on running a Freenet gateway on a small VPS: firewall rules, \
             systemd units, log rotation and how to read the peer connection \
             table when the node will not bootstrap. The troubleshooting section \
             covers NAT, clock skew and running out of file descriptors."
        );
        let j = Sketch::of(&a).unwrap().jaccard(&Sketch::of(&b).unwrap());
        assert!(j < NEAR_DUP_JACCARD, "jaccard {j}");
    }

    #[test]
    fn same_owner_merges_even_when_the_content_differs() {
        let owner = Some("ab".repeat(32));
        let a = Fingerprint {
            owner: owner.clone(),
            ..fp_text(&seller_page("x"))
        };
        let b = Fingerprint {
            owner,
            sketch: None,
            archive: None,
        };
        let store = vec![Stored {
            locator: A.into(),
            first_seen: 1,
            print: a,
        }];
        let m = find_canonical(B, &b, &store, &live(&[A])).unwrap();
        assert_eq!(m.signal, Signal::SameOwner);
    }

    #[test]
    fn identical_archive_under_a_different_owner_is_a_clone() {
        let a = Fingerprint {
            owner: Some("aa".repeat(32)),
            archive: Some("11".repeat(32)),
            sketch: None,
        };
        let b = Fingerprint {
            owner: Some("bb".repeat(32)),
            archive: Some("11".repeat(32)),
            sketch: None,
        };
        let store = vec![Stored {
            locator: A.into(),
            first_seen: 1,
            print: a,
        }];
        let m = find_canonical(B, &b, &store, &live(&[A])).unwrap();
        assert_eq!(m.signal, Signal::SameArchive);
    }

    #[test]
    fn unknown_fields_never_match() {
        let empty = Fingerprint::default();
        assert_eq!(signal(&empty, &empty), None);
    }

    #[test]
    fn earliest_live_entry_is_canonical() {
        let text = seller_page("addr");
        let store = vec![
            Stored {
                locator: B.into(),
                first_seen: 200,
                print: fp_text(&text),
            },
            Stored {
                locator: A.into(),
                first_seen: 100,
                print: fp_text(&text),
            },
        ];
        let cand = fp_text(&text);
        let m = find_canonical(C, &cand, &store, &live(&[A, B])).unwrap();
        assert_eq!(m.canonical, A);
        // A has left the index: B, the earliest STILL LIVE, is canonical.
        let m = find_canonical(C, &cand, &store, &live(&[B])).unwrap();
        assert_eq!(m.canonical, B);
        // Nothing live: nothing to be a duplicate of.
        assert_eq!(find_canonical(C, &cand, &store, &live(&[])), None);
    }

    #[test]
    fn pages_of_one_container_and_resources_of_one_app_are_never_compared() {
        assert!(!comparable(
            "freenet:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/a.html",
            "freenet:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/b.html"
        ));
        assert!(!comparable(
            "freenet:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "freenet:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/"
        ));
        assert!(!comparable("app:delta/AWPjDQdKey", "app:delta/9CiJipmep5"));
        assert!(comparable("app:delta/AWPjDQdKey", "app:other/AWPjDQdKey"));
        assert!(comparable("app:delta/AWPjDQdKey", B));
        assert!(comparable(A, B));

        // And `find_canonical` honours it even when every signal agrees.
        let text = seller_page("addr");
        let same = Fingerprint {
            owner: Some("aa".repeat(32)),
            archive: Some("11".repeat(32)),
            sketch: Sketch::of(&text),
        };
        let store = vec![Stored {
            locator: "app:delta/AWPjDQdKey".into(),
            first_seen: 1,
            print: same.clone(),
        }];
        let l = live(&["app:delta/AWPjDQdKey"]);
        assert_eq!(
            find_canonical("app:delta/9CiJipmep5", &same, &store, &l),
            None
        );
    }

    #[test]
    fn too_little_text_has_no_sketch() {
        assert_eq!(Sketch::of("Enter your password to continue"), None);
        assert_eq!(Sketch::of(""), None);
    }

    #[test]
    fn jaccard_is_exact_for_small_sets_and_symmetric() {
        let a = Sketch::of(&seller_page("one")).unwrap();
        assert_eq!(a.jaccard(&a), 1.0);
        let b = Sketch::of(&seller_page("two")).unwrap();
        assert_eq!(a.jaccard(&b), b.jaccard(&a));
    }

    #[test]
    fn jaccard_estimate_tracks_the_true_value_on_large_pages() {
        // Two pages of 2000 distinct shingles sharing exactly half: true J = 1/3.
        let words =
            |r: std::ops::Range<usize>| r.map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let a = Sketch::of(&words(0..2004)).unwrap();
        let b = Sketch::of(&words(1000..3004)).unwrap();
        let j = a.jaccard(&b);
        assert!((j - 1.0 / 3.0).abs() < 0.08, "estimate {j}");
    }

    #[test]
    fn owner_needs_exactly_32_bytes_of_params() {
        let info = |p: &str| serde_json::json!({ "params_hex": p, "archive_blake3": null });
        assert_eq!(
            Fingerprint::from_contract_info(&info(&"Ab".repeat(32))).owner,
            Some("ab".repeat(32))
        );
        // The Atlas index's own params are root_vk || slug: not an owner key.
        assert_eq!(
            Fingerprint::from_contract_info(&info(&"ab".repeat(39))).owner,
            None
        );
        assert_eq!(Fingerprint::from_contract_info(&info("")).owner, None);
        assert_eq!(
            Fingerprint::from_contract_info(&serde_json::json!({})),
            Fingerprint::default()
        );
    }

    #[test]
    fn store_round_trips_and_later_lines_win() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fp.txt");
        let mut store = Store::load(&path);
        assert!(store.entries.is_empty());
        let s1 = Stored {
            locator: A.into(),
            first_seen: 5,
            print: Fingerprint {
                owner: Some("aa".repeat(32)),
                archive: None,
                sketch: Sketch::of(&seller_page("x")),
            },
        };
        store.record(&path, s1.clone()).unwrap();
        let s2 = Stored {
            first_seen: 5,
            print: Fingerprint::default(),
            ..s1.clone()
        };
        store.record(&path, s2.clone()).unwrap();
        assert_eq!(Store::load(&path).entries, vec![s2.clone()]);
        std::fs::write(&path, format!("{}\ngarbage line\n", format_line(&s1))).unwrap();
        assert_eq!(Store::load(&path).entries, vec![s1.clone()]);
        let tmp = dir.path().join("fp.tmp");
        Store {
            entries: vec![s1.clone(), s2.clone()],
        }
        .save(&path, &tmp)
        .unwrap();
        // Same locator twice: the later one wins on load.
        assert_eq!(Store::load(&path).entries, vec![s2]);
    }
}
