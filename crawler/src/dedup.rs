//! Duplicate and clone collapsing (issue #68).
//!
//! A copy of a site should not be listed next to the original, or instead of it.
//! Three cases, and the signal that catches each:
//!
//! - **An author republishes their own site** under a second contract id, often
//!   by accident. Both contracts carry the SAME owner key, because a web
//!   container's parameters ARE its owner's verifying key.
//! - **A naive clone** republishes someone else's archive byte for byte under the
//!   cloner's own key. The owner keys differ but the archive bytes are identical.
//! - **A scam clone** copies a site and changes a detail, such as the Bitcoin
//!   address on a seller page. Exact matching cannot see this, so the rendered
//!   text is compared as a set of word shingles. A near-identical pair is a
//!   duplicate, and so is a candidate that contains nearly all of an earlier
//!   page plus some padding.
//!
//! Owner and archive are corroboration, not proof (see [`signal`]). A same-owner
//! match also needs both pages' text to be comparable and not clearly different
//! (at least [`OWNER_TEXT_FLOOR`]), because a contract can name anybody's public
//! key as its parameters, and a key alone would let such a contract get its real
//! owner's sites held. A same-archive match stops only at clearly different text.
//!
//! When a candidate matches an entry already in the live index, that entry is
//! CANONICAL (first seen wins) and the candidate is held, not listed. This is
//! deliberately the weakest claim that works: a clone that happened to be indexed
//! first keeps the listing. Outranking first-seen needs a publisher identity the
//! site itself declares (step 2 of #68). The fingerprint store keeps `first_seen`
//! and the owner key per entry, and the held list keeps every refused duplicate
//! with its canonical, so such a claim has both sides to work with.
//!
//! A held duplicate is not final. When its canonical leaves the live index (a
//! curator removed a clone, a dead original was purged), or after
//! [`HELD_RECHECK_SECS`] in any case, it is queued again and judged afresh. See
//! `HeldStore`.
//!
//! What a text comparison does NOT catch, stated so nobody reads more into it:
//!
//! - A change made in the DOM rather than the text: a letter hidden with CSS
//!   inside every word, or a long hidden block, still reaches the text compared.
//!   Hidden blocks can be closed in the renderer (issue #70); hidden letters
//!   cannot.
//! - A clone padded to more than about twice the original (see
//!   [`CONTAINED_MIN_JACCARD`]), which looks the same as an aggregator quoting it.
//! - A clone that was indexed first. First seen is canonical.
//!
//! Those are what a publisher identity the site declares (step 2 of #68) is for.
//!
//! What is NEVER compared:
//!
//! - Two locators on the SAME contract id. `freenet:<id>/a.html` and
//!   `freenet:<id>/b.html` are different pages of one container: same owner and
//!   same archive by construction.
//! - Two locators of the same app RESOURCE (`app:delta/X` and a deep link into
//!   it). Different resources of one app ARE compared, on text only: every Delta
//!   site is served by the one Delta container, so their owner and archive are
//!   the app's and say nothing, but a copied Delta seller page is the cheapest
//!   scam clone there is. Their text is the content region, without the app's
//!   chrome (see `render.js`), so distinct sites on one host app do not look
//!   alike. The PR for #68 reports the measured margin between them.
//!
//! This module is pure apart from its two small state files: no network, no
//! subprocess. `main.rs` fetches the owner key and archive hash
//! (`atlasctl contract-info`) and the rendered text, and hands them here.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{BuildHasher, Hasher};
use std::io::Write;
use std::path::Path;

use anyhow::{bail, Context, Result};

/// Words per shingle.
///
/// Five is long enough that two unrelated pages rarely share a shingle by
/// accident (common short phrases like "on freenet" fall below it), and short
/// enough that a one-token edit, such as a swapped address, disturbs only five
/// shingles out of hundreds.
pub const SHINGLE_WORDS: usize = 5;

/// How many shingle hashes a sketch keeps: the numerically smallest `K`.
///
/// When the two pages' shingles together number fewer than this, the sketches
/// hold both whole sets and the similarity is EXACT. Otherwise the bottom-k
/// estimator's standard error is about `sqrt(J(1-J)/K)`, roughly 0.025 at
/// J = 0.8.
pub const SKETCH_K: usize = 256;

/// Fewer distinct shingles than this and a sketch cannot support a verdict on
/// its own (see [`Sketch::usable`]).
///
/// Below it, two unrelated pages that share a boilerplate sentence or two (a
/// login prompt, a "loading" notice) could clear the threshold on that alone.
/// A smaller sketch is still kept, because "is this short text contained in that
/// page" stays meaningful and gates the whole-page pair (see
/// `Fingerprint::pairs`). The owner and archive signals still apply.
pub const MIN_SHINGLES: usize = 30;

/// Estimated Jaccard similarity at or above which two pages are the same content.
///
/// Checked against the live index with `--dedup-backfill`, whose report lists
/// the closest pairs of DISTINCT live entries; the PR for #68 has the numbers. A
/// scam clone that swaps one address in a few hundred words stays far above it.
pub const NEAR_DUP_JACCARD: f64 = 0.80;

/// A candidate that contains at least this fraction of an earlier page's
/// shingles is that page plus padding, provided it also clears
/// [`CONTAINED_MIN_JACCARD`].
///
/// Jaccard alone misses padding: a clone that keeps the whole seller page and
/// appends a long FAQ falls below [`NEAR_DUP_JACCARD`] while still being the
/// seller page.
pub const CONTAINED: f64 = 0.90;

/// The Jaccard floor for a containment match. 0.5 means the candidate is at most
/// about twice the size of the page it contains, which keeps a directory or an
/// aggregator that quotes a small site in full from being merged into it. The
/// price, accepted: a clone padded beyond that is not caught.
pub const CONTAINED_MIN_JACCARD: f64 = 0.50;

/// Below this Jaccard, two pages with enough text to compare are different sites,
/// whatever their owner key or archive hash says. See the module doc.
pub const OWNER_TEXT_FLOOR: f64 = 0.30;

/// A bottom-k sketch of a page's word-shingle set.
#[derive(Clone)]
pub struct Sketch {
    /// How many distinct shingles the page had, which containment needs.
    n: u64,
    /// The `SKETCH_K` smallest salted shingle hashes, ascending and distinct.
    mins: Vec<u64>,
    /// EVERY shingle hash, ascending, for a sketch made this run from a page we
    /// just fetched. Never stored (`decode` leaves it `None`).
    ///
    /// It makes containment of a stored canonical in a fresh candidate exact,
    /// where the bottom-k estimate is worst: a short text inside a long page
    /// shares almost none of the long page's smallest hashes, so the estimate is
    /// usually 0 and occasionally far too high, by chance of the salt.
    all: Option<Vec<u64>>,
}

/// Equality is the stored identity: `all` is a run-time cache of the same set.
impl PartialEq for Sketch {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.mins == other.mins
    }
}

impl Eq for Sketch {}

impl std::fmt::Debug for Sketch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sketch(n={}, kept={})", self.n, self.mins.len())
    }
}

/// Characters a reader cannot see. Stripped before splitting into words, or a
/// clone could put a soft hyphen or a zero-width joiner inside every word and
/// change every shingle without changing what the page shows.
///
/// The Unicode `Default_Ignorable_Code_Point` ranges, minus a few that cannot
/// occur in rendered text anyway.
fn is_invisible(c: char) -> bool {
    matches!(c,
        '\u{00AD}' | '\u{034F}' | '\u{061C}' | '\u{115F}' | '\u{1160}'
        | '\u{17B4}' | '\u{17B5}' | '\u{180B}'..='\u{180F}'
        | '\u{200B}'..='\u{200F}' | '\u{202A}'..='\u{202E}'
        | '\u{2060}'..='\u{206F}' | '\u{3164}' | '\u{FE00}'..='\u{FE0F}'
        | '\u{FEFF}' | '\u{FFA0}' | '\u{1BCA0}'..='\u{1BCA3}'
        | '\u{1D173}'..='\u{1D17A}' | '\u{E0000}'..='\u{E0FFF}')
}

/// The words a page says, as compared: invisible characters dropped, each
/// character replaced by its Unicode confusable prototype (the UTS #39
/// skeleton), what remains folded to ASCII (`deunicode`), lowercased, and split
/// on anything not alphanumeric.
///
/// The skeleton is what makes a homoglyph swap useless: a Cyrillic `с`, `р` or
/// `у` in place of a Latin `c`, `p` or `y` maps to the Latin letter. `deunicode`
/// alone transliterates instead (Cyrillic `с` becomes `s`), which is why the
/// skeleton comes first. Both sides of every comparison go through the same
/// pipeline, so it does not matter that the skeleton also merges some genuinely
/// different characters (`0` and `O`, `1` and `l`). Digits are kept, because an
/// address or a price is part of what a page says.
///
/// What this cannot see is a change made in the DOM rather than the text: a
/// letter inserted inside a word but hidden with CSS still reaches the rendered
/// text. That residual is for step 2 of #68 (a publisher identity), not for a
/// text comparison.
fn words(text: &str) -> Vec<String> {
    let visible: String = text.chars().filter(|c| !is_invisible(*c)).collect();
    let skeleton: String = unicode_security::confusable_detection::skeleton(&visible).collect();
    deunicode::deunicode(&skeleton)
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

impl Sketch {
    /// Sketch `text` under `salt`, or `None` if it has no shingle at all.
    pub fn of(text: &str, salt: u64) -> Option<Self> {
        let words = words(text);
        if words.len() < SHINGLE_WORDS {
            return None;
        }
        let mut hashes: Vec<u64> = words
            .windows(SHINGLE_WORDS)
            .map(|w| shingle_hash(&w.join(" "), salt))
            .collect();
        hashes.sort_unstable();
        hashes.dedup();
        let n = hashes.len() as u64;
        let mins = hashes[..hashes.len().min(SKETCH_K)].to_vec();
        Some(Self {
            n,
            mins,
            all: Some(hashes),
        })
    }

    /// Drop the full hash list, keeping what is stored.
    pub fn without_full(mut self) -> Self {
        self.all = None;
        self
    }

    /// Whether this sketch has enough text ([`MIN_SHINGLES`]) to take part in a
    /// verdict, rather than only gate one.
    pub fn usable(&self) -> bool {
        self.n >= MIN_SHINGLES as u64
    }

    /// Estimated Jaccard similarity of the two underlying shingle sets.
    ///
    /// The bottom-k estimator: walk the union of the two sketches in ascending
    /// order, stop after `K` values, and count how many were in both. A value
    /// below the larger sketch's cut-off is in that set iff it is in its sketch,
    /// so membership is exact for every value walked.
    pub fn jaccard(&self, other: &Self) -> f64 {
        let (a, b) = (&self.mins, &other.mins);
        let (mut i, mut j) = (0, 0);
        let (mut union, mut both) = (0usize, 0usize);
        // Exhausting a FULL sketch takes K steps, which ends the loop first, so
        // reaching the end of one sketch means its whole set has been walked.
        while union < SKETCH_K && (i < a.len() || j < b.len()) {
            match (a.get(i), b.get(j)) {
                (Some(x), Some(y)) if x == y => {
                    both += 1;
                    i += 1;
                    j += 1;
                }
                (Some(x), Some(y)) if x < y => i += 1,
                (Some(_), Some(_)) | (None, Some(_)) => j += 1,
                (Some(_), None) => i += 1,
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

    /// Fraction of `self`'s shingles that also appear in `other`.
    ///
    /// When `other` was made this run (it has `all`): the fraction of `self`'s
    /// kept hashes found among ALL of `other`'s. That is exact when `self` has at
    /// most `SKETCH_K` shingles (it keeps them all), and otherwise an unbiased
    /// estimate from a uniform sample of `SKETCH_K` of them. This is the case
    /// that matters, a stored canonical against a fresh candidate.
    ///
    /// Between two stored sketches (the report), it falls back to the Jaccard
    /// estimate and the set sizes, `|A∩B| = J(|A|+|B|)/(1+J)`, which is poor when
    /// the sizes differ a lot.
    pub fn contained_in(&self, other: &Self) -> f64 {
        if let Some(all) = &other.all {
            let found = self
                .mins
                .iter()
                .filter(|h| all.binary_search(h).is_ok())
                .count();
            return found as f64 / self.mins.len() as f64;
        }
        let j = self.jaccard(other);
        let inter = j * (self.n + other.n) as f64 / (1.0 + j);
        (inter / self.n as f64).min(1.0)
    }

    fn encode(&self) -> String {
        let hex: String = self.mins.iter().map(|h| format!("{h:016x}")).collect();
        format!("{}:{hex}", self.n)
    }

    fn decode(s: &str) -> Option<Self> {
        let (n, hex) = s.split_once(':')?;
        let n: u64 = n.parse().ok()?;
        // ASCII first: `hex[i..i + 16]` below must not slice inside a character.
        if hex.is_empty() || !hex.is_ascii() || !hex.len().is_multiple_of(16) {
            return None;
        }
        let mins = (0..hex.len())
            .step_by(16)
            .map(|i| u64::from_str_radix(&hex[i..i + 16], 16).ok())
            .collect::<Option<Vec<u64>>>()?;
        // Re-establish the invariants rather than trust the file: `jaccard`
        // assumes sorted, distinct, at most SKETCH_K, and `contained_in` divides
        // by `n`.
        let ok = mins.len() <= SKETCH_K
            && mins.windows(2).all(|w| w[0] < w[1])
            && n >= mins.len() as u64
            && (mins.len() == SKETCH_K || n == mins.len() as u64);
        ok.then_some(Self { n, mins, all: None })
    }
}

/// FNV-1a over the shingle, then the installation's secret salt, then a
/// SplitMix64 finalizer.
///
/// Stable across Rust releases because it is PERSISTED (see `fnv1a64` in
/// `main.rs` for why `DefaultHasher` is not). Finalized because a bottom-k sketch
/// keeps the SMALLEST hashes, which is only a uniform sample if the low values
/// are well mixed. Salted because the sketch is otherwise steerable: with a
/// public hash, a clone can append words chosen so their shingles hash below the
/// original's, fill the union's bottom K with them, and read as unrelated. The
/// salt never leaves this machine (see `Store`).
fn shingle_hash(s: &str, salt: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= salt;
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

/// What is known about one locator for duplicate detection. Every field is
/// optional: a missing one is "unknown", which never matches anything.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fingerprint {
    /// The contract's 32-byte owner verifying key, lowercase hex. See
    /// [`Fingerprint::from_contract_info`] for when it is set.
    pub owner: Option<String>,
    /// BLAKE3 of the web archive, lowercase hex, excluding the signed metadata.
    pub archive: Option<String>,
    /// Sketch of the CONTENT text, the same text the describer reads.
    pub sketch: Option<Sketch>,
    /// Sketch of the whole entry page's visible text, for a contract of its own
    /// only. The content region is chosen by the page's own markup (the first
    /// `main` or `article`), so a clone can put a short blurb there and the
    /// copied page beside it; comparing the whole page as well closes that.
    pub page_sketch: Option<Sketch>,
}

impl Fingerprint {
    /// Whether this fingerprint was made this run (has the full hash lists), so
    /// containment of another page in it is exact rather than estimated.
    fn fresh(&self) -> bool {
        [&self.sketch, &self.page_sketch]
            .into_iter()
            .flatten()
            .any(|s| s.all.is_some())
    }

    /// This fingerprint as it is stored: without the run-time full hash lists.
    pub fn stored(self) -> Self {
        Self {
            sketch: self.sketch.map(Sketch::without_full),
            page_sketch: self.page_sketch.map(Sketch::without_full),
            ..self
        }
    }

    /// Parse `atlasctl contract-info` output into the owner and archive fields.
    ///
    /// The owner is set only when the parameters are exactly 32 bytes AND the
    /// state parsed as a web container (`archive_blake3` present). Parameters of
    /// another shape encode something else (the Atlas index's are
    /// `root_vk || slug`), and a contract whose state is not a web container is
    /// not a site whose params mean "owner". `contract-info` itself only reports
    /// params it has checked hash to the contract id.
    pub fn from_contract_info(json: &serde_json::Value) -> Self {
        let hex64 = |k: &str| {
            json[k]
                .as_str()
                .filter(|v| v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit()))
                .map(str::to_ascii_lowercase)
        };
        let archive = hex64("archive_blake3");
        let owner = archive.as_ref().and_then(|_| hex64("params_hex"));
        Self {
            owner,
            archive,
            sketch: None,
            page_sketch: None,
        }
    }

    /// `(jaccard, containment of the canonical side in the candidate side)` for
    /// each sketch pair that has a sketch on both sides, evaluated per pair so a
    /// verdict never mixes one pair's Jaccard with another's containment.
    ///
    /// Three pairs: content with content, whole page with whole page, and the
    /// candidate's whole page against the canonical's CONTENT. The last is the
    /// clone shape when the canonical has no page sketch (an `app:` resource) and
    /// the candidate hides the copy outside its own content region.
    ///
    /// The whole-page pair counts only when the canonical's CONTENT is also
    /// contained in the candidate's page. Two distinct sites built from one theme
    /// share a large nav and footer, so their whole pages can look alike while
    /// neither contains the other's content; a clone always contains it.
    fn pairs(&self, canon: &Self) -> Vec<(f64, f64)> {
        let sim = |a: Option<&Sketch>, b: Option<&Sketch>| {
            let (cand, canon) = (a?, b?);
            Some((cand.jaccard(canon), canon.contained_in(cand)))
        };
        fn usable(x: &Option<Sketch>) -> Option<&Sketch> {
            x.as_ref().filter(|s| s.usable())
        }
        // The gate reads the canonical's content however SHORT it is: a seller
        // page whose `<main>` is a one-line blurb, with everything else beside
        // it, still has that blurb inside any whole-page copy.
        let gate = sim(self.page_sketch.as_ref(), canon.sketch.as_ref())
            .is_some_and(|(_, c)| c >= CONTAINED);
        let page = sim(usable(&self.page_sketch), usable(&canon.page_sketch)).filter(|_| gate);
        [
            sim(usable(&self.sketch), usable(&canon.sketch)),
            page,
            sim(usable(&self.page_sketch), usable(&canon.sketch)),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// The Jaccard that decides whether two pages are "clearly different": the
    /// CONTENT pair when both have one, else the whole-page pair. Never the
    /// maximum over pairs, because the whole page includes chrome, and two
    /// distinct sites on one engine share their chrome.
    fn content_jaccard(&self, canon: &Self) -> Option<f64> {
        let both = |a: &Option<Sketch>, b: &Option<Sketch>| {
            let (a, b) = (a.as_ref()?, b.as_ref()?);
            (a.usable() && b.usable()).then(|| a.jaccard(b))
        };
        both(&self.sketch, &canon.sketch).or_else(|| both(&self.page_sketch, &canon.page_sketch))
    }

    /// Highest Jaccard and highest containment over the pairs, for the report.
    fn best_text(&self, canon: &Self) -> Option<(f64, f64)> {
        self.pairs(canon)
            .into_iter()
            .reduce(|x, y| (x.0.max(y.0), x.1.max(y.1)))
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
    /// The candidate contains this fraction of the canonical page.
    Contains(f64),
}

impl Signal {
    /// The decision-log reason fragment. Stable, like `Outcome::token`: an
    /// operator greps `duplicate-of` lines by it.
    pub fn reason(&self) -> String {
        match self {
            Self::SameOwner => "same-owner".to_string(),
            Self::SameArchive => "same-archive".to_string(),
            Self::NearText(j) => format!("near-text jaccard={j:.3}"),
            Self::Contains(c) => format!("contains containment={c:.3}"),
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
    rest.split(['/', '?', '#']).next()
}

/// The `(slug, resource)` an `app:<slug>/<resource>…` locator names, or `None`.
fn app_resource_of(loc: &str) -> Option<(&str, &str)> {
    let (slug, rest) = loc.strip_prefix("app:")?.split_once('/')?;
    Some((slug, rest.split(['/', '?', '#']).next()?))
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
    if let (Some(x), Some(y)) = (app_resource_of(a), app_resource_of(b)) {
        if x == y {
            return false;
        }
    }
    true
}

/// Whether two locators are served by one shared container (two resources of one
/// app), so that owner and archive describe the app, not either site.
fn share_container(a: &str, b: &str) -> bool {
    matches!((app_resource_of(a), app_resource_of(b)), (Some((x, _)), Some((y, _))) if x == y)
}

/// The strongest signal on which candidate `cand` (at `cand_loc`) duplicates
/// `canon` (at `canon_loc`), if any. Assumes [`comparable`] has already said yes.
///
/// Owner and archive are corroborating signals, not proofs:
///
/// - SAME OWNER also needs the text to agree: both pages have text, and it is
///   not clearly different. A contract can name anyone's public key as its
///   params, so a key alone would let an impostor with no text get every later
///   site of that key's real owner held.
/// - SAME ARCHIVE only needs the text not to disagree, because identical bytes
///   cannot be produced without copying them. It still stops at clearly
///   different text, which is what a generic shell container loading per-site
///   content looks like.
/// - Neither applies between resources of one app: their container is shared.
pub fn signal(
    cand_loc: &str,
    cand: &Fingerprint,
    canon_loc: &str,
    canon: &Fingerprint,
) -> Option<Signal> {
    if !share_container(cand_loc, canon_loc) {
        let cj = cand.content_jaccard(canon);
        if cand.owner.is_some()
            && cand.owner == canon.owner
            && cj.is_some_and(|j| j >= OWNER_TEXT_FLOOR)
        {
            return Some(Signal::SameOwner);
        }
        if cand.archive.is_some()
            && cand.archive == canon.archive
            && !cj.is_some_and(|j| j < OWNER_TEXT_FLOOR)
        {
            return Some(Signal::SameArchive);
        }
    }
    let pairs = cand.pairs(canon);
    let near = pairs
        .iter()
        .map(|&(j, _)| j)
        .filter(|&j| j >= NEAR_DUP_JACCARD)
        .reduce(f64::max);
    if let Some(j) = near {
        return Some(Signal::NearText(j));
    }
    pairs
        .iter()
        .filter(|&&(j, c)| c >= CONTAINED && j >= CONTAINED_MIN_JACCARD)
        .map(|&(_, c)| c)
        .reduce(f64::max)
        .map(Signal::Contains)
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
        .filter_map(|s| signal(loc, candidate, &s.locator, &s.print).map(|sig| (s, sig)))
        .min_by(|(a, _), (b, _)| (a.first_seen, &a.locator).cmp(&(b.first_seen, &b.locator)))
        .map(|(s, sig)| Match {
            canonical: s.locator.clone(),
            signal: sig,
        })
}

/// How many of the closest non-duplicate text pairs `report` lists.
const REPORT_NEAREST: usize = 25;

/// A curator's view of `entries` as the duplicate check sees them: every pair it
/// would call the same site (canonical, the earlier one, second), then the
/// closest text pairs it would NOT, which is the false-positive margin.
pub fn report(entries: &[Stored]) -> String {
    type Dup<'a> = (&'a Stored, &'a Stored, Signal, Option<(f64, f64)>);
    let mut dups: Vec<Dup> = Vec::new();
    let mut near: Vec<(f64, f64, &Stored, &Stored)> = Vec::new();
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
            let text = second.print.best_text(&first.print);
            match signal(&second.locator, &second.print, &first.locator, &first.print) {
                Some(sig) => dups.push((first, second, sig, text)),
                None => {
                    if let Some((j, c)) = text {
                        near.push((j, c, first, second));
                    }
                }
            }
        }
    }
    let mut out = format!(
        "{compared} comparable pairs; {} would be duplicates (near-text jaccard >= \
         {NEAR_DUP_JACCARD}, or containment >= {CONTAINED} with jaccard >= \
         {CONTAINED_MIN_JACCARD}; owner/archive unless jaccard < {OWNER_TEXT_FLOOR})\n\
         A line marked ~ compares two STORED sketches, so its containment (c=) is an \
         estimate that is poor for a short page inside a long one, and a containment \
         verdict the crawler would reach live can be missing from it.\n",
        dups.len()
    );
    // The later entry plays the candidate, so it decides whether `c` is exact.
    let mark = |second: &Stored| if second.print.fresh() { " " } else { "~" };
    for (first, second, sig, text) in &dups {
        let t = text
            .map(|(j, c)| format!(" text j={j:.3} c={c:.3}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "{} DUPLICATE {} of {} ({}){t}\n",
            mark(second),
            second.locator,
            first.locator,
            sig.reason()
        ));
    }
    near.sort_by(|x, y| y.0.total_cmp(&x.0).then(y.1.total_cmp(&x.1)));
    out.push_str(&format!(
        "closest {} text pairs that are NOT duplicates (by jaccard):\n",
        REPORT_NEAREST.min(near.len())
    ));
    for (j, c, a, b) in near.iter().take(REPORT_NEAREST) {
        out.push_str(&format!(
            "{} j={j:.3} c={c:.3}  {}  {}\n",
            mark(b),
            a.locator,
            b.locator
        ));
    }
    near.sort_by(|x, y| y.1.total_cmp(&x.1));
    out.push_str("closest 10 by containment:\n");
    for (j, c, a, b) in near.iter().take(10) {
        out.push_str(&format!(
            "{} c={c:.3} j={j:.3}  {}  {}\n",
            mark(b),
            a.locator,
            b.locator
        ));
    }
    out
}

/// Read a state file, treating ONLY a missing file as empty. Any other failure
/// is an error: reading an unreadable file as empty would silently switch
/// duplicate detection off while the daemon kept appending to it.
fn read_state(path: &Path) -> Result<String> {
    match fs::read_to_string(path) {
        Ok(s) => Ok(s),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

const SALT_HEADER: &str = "#salt\t";

/// Version of the text pipeline (`words`, `shingle_hash`, and `visible_text` in
/// `main.rs`, which makes the whole-page text) the stored sketches
/// were made with, written into the header next to the salt. A sketch made by a
/// different pipeline loads cleanly and never matches anything, so a store from
/// another version is refused, loudly, rather than silently switching detection
/// off. BUMP THIS whenever any of the three changes.
const PIPELINE: u32 = 1;

fn header(salt: u64) -> String {
    format!("{SALT_HEADER}{salt:016x}\tv{PIPELINE}\n")
}

/// The salt in a header line, if the line is a header of THIS pipeline version.
fn parse_header(line: &str) -> Option<u64> {
    let (salt, version) = line.strip_prefix(SALT_HEADER)?.trim().split_once('\t')?;
    (version == format!("v{PIPELINE}"))
        .then(|| u64::from_str_radix(salt, 16).ok())
        .flatten()
}

/// The fingerprint store: `crawler-fingerprints.txt`. A `#salt` header (salt and
/// [`PIPELINE`] version), then one
/// line per indexed locator:
/// `locator \t first_seen \t owner \t archive \t sketch \t page_sketch`, `-` for
/// an unknown field.
///
/// Crawler-local, like the recheck schedule: never published, which is what
/// keeps the salt secret. The salt is created with the file and every sketch in
/// it depends on it, so a file without one is refused rather than guessed at.
/// A malformed entry line is skipped with a warning: losing one fingerprint only
/// means one fewer entry a clone can be caught against.
pub struct Store {
    pub salt: u64,
    pub entries: Vec<Stored>,
}

impl Store {
    pub fn load(path: &Path) -> Result<Self> {
        let body = read_state(path)?;
        let mut lines = body.lines().filter(|l| !l.trim().is_empty());
        let Some(first) = lines.next() else {
            return Ok(Self::fresh());
        };
        let Some(salt) = parse_header(first) else {
            bail!(
                "{} has no #salt header for text pipeline v{PIPELINE}; move it aside \
                 and rebuild it with --dedup-backfill",
                path.display()
            );
        };
        // Later lines win: a re-fingerprint is appended, not edited in.
        let mut at: HashMap<String, usize> = HashMap::new();
        let mut entries: Vec<Stored> = Vec::new();
        for line in lines {
            match parse_line(line) {
                Some(s) => match at.get(&s.locator) {
                    Some(&i) => entries[i] = s,
                    None => {
                        at.insert(s.locator.clone(), entries.len());
                        entries.push(s);
                    }
                },
                None => eprintln!(
                    "warn: skipping malformed fingerprint line in {}",
                    path.display()
                ),
            }
        }
        Ok(Self { salt, entries })
    }

    /// An empty store with a new random salt.
    pub fn fresh() -> Self {
        // `RandomState` is seeded from the OS's randomness, which is all a
        // secret salt needs, without a dependency for it.
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(std::process::id() as u64);
        Self {
            salt: h.finish(),
            entries: Vec::new(),
        }
    }

    /// Add or replace one locator's fingerprint and append it to `path`, writing
    /// the salt header first if the file is new.
    ///
    /// One `write_all` per call, so a crash cannot leave a line without its
    /// newline for the next append to run into.
    ///
    /// Refuses to append under a DIFFERENT salt's header (another process created
    /// the file meanwhile): those lines would load cleanly and never match.
    pub fn record(&mut self, path: &Path, mut s: Stored) -> Result<()> {
        s.print = s.print.stored();
        let mut out = String::new();
        match read_state(path)?.lines().find(|l| !l.trim().is_empty()) {
            None => out.push_str(&header(self.salt)),
            Some(first) => {
                if parse_header(first) != Some(self.salt) {
                    bail!(
                        "{} was written under a different salt; not appending",
                        path.display()
                    );
                }
            }
        }
        out.push_str(&format_line(&s));
        out.push('\n');
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut f| f.write_all(out.as_bytes()))
            .with_context(|| format!("appending to {}", path.display()))?;
        self.entries.retain(|e| e.locator != s.locator);
        self.entries.push(s);
        Ok(())
    }

    /// Rewrite the whole file, atomically.
    pub fn save(&self, path: &Path, tmp: &Path) -> Result<()> {
        let mut body = header(self.salt);
        for s in &self.entries {
            body.push_str(&format_line(s));
            body.push('\n');
        }
        fs::write(tmp, body).with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(tmp, path).with_context(|| format!("renaming onto {}", path.display()))
    }
}

fn opt(s: &Option<String>) -> &str {
    s.as_deref().unwrap_or("-")
}

fn opt_sketch(s: &Option<Sketch>) -> String {
    s.as_ref().map(Sketch::encode).unwrap_or_else(|| "-".into())
}

fn format_line(s: &Stored) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}",
        s.locator,
        s.first_seen,
        opt(&s.print.owner),
        opt(&s.print.archive),
        opt_sketch(&s.print.sketch),
        opt_sketch(&s.print.page_sketch),
    )
}

fn parse_line(line: &str) -> Option<Stored> {
    let f: Vec<&str> = line.split('\t').collect();
    let [locator, first_seen, owner, archive, sketch, page_sketch] = f[..] else {
        return None;
    };
    let field = |s: &str| (s != "-").then(|| s.to_string());
    let sk = |s: &str| -> Option<Option<Sketch>> {
        if s == "-" {
            Some(None)
        } else {
            Some(Some(Sketch::decode(s)?))
        }
    };
    Some(Stored {
        locator: locator.to_string(),
        first_seen: first_seen.parse().ok()?,
        print: Fingerprint {
            owner: field(owner),
            archive: field(archive),
            sketch: sk(sketch)?,
            page_sketch: sk(page_sketch)?,
        },
    })
}

/// How long a duplicate stays held before it is judged again even though its
/// canonical is still live.
///
/// A verdict about a page's TEXT goes stale when the page changes. A site held
/// while it was still an unedited template, or the placeholder copy an author
/// put up before writing their own, must not stay held for as long as the page
/// it matched lives. Re-judging costs a render and a contract GET, never tokens
/// (a duplicate is decided before the describer), so a month is cheap.
pub const HELD_RECHECK_SECS: u64 = 30 * 86_400;

/// Upper bound on the held list. One clone contract can be posted under any
/// number of paths, each of which is held separately, and an unbounded list is
/// a disk- and time-filling bug. Over the bound the OLDEST entry is dropped,
/// which only means it can be discovered and judged again.
pub const MAX_HELD: usize = 5_000;

/// A candidate refused as a duplicate, held until its canonical leaves the live
/// index or [`HELD_RECHECK_SECS`] pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Held {
    pub canonical: String,
    /// Who posted it, so it rejoins the queue under its original author's share.
    pub author: String,
    /// When it was held (unix seconds).
    pub at: u64,
    pub locator: String,
}

/// The held duplicates: `crawler-duplicates.txt`, one
/// `canonical \t author \t held_at \t locator` line each.
///
/// Why a duplicate is HELD rather than marked seen: `crawler-seen.txt` is
/// terminal, and a duplicate verdict is only as good as its canonical and as
/// current as the text it compared. If a clone was indexed first and the real
/// site was refused against it, removing the clone must let the real site back
/// in, and nothing re-reads the seen file for that. So every run re-queues each
/// held locator whose canonical is no longer live, or that has been held for
/// [`HELD_RECHECK_SECS`].
pub struct HeldStore {
    pub entries: Vec<Held>,
}

impl HeldStore {
    pub fn load(path: &Path) -> Result<Self> {
        let body = read_state(path)?;
        let mut at: HashMap<String, usize> = HashMap::new();
        let mut entries: Vec<Held> = Vec::new();
        for line in body.lines().filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            let [canonical, author, held_at, locator] = f[..] else {
                eprintln!("warn: skipping malformed line in {}", path.display());
                continue;
            };
            let Ok(held_at) = held_at.parse() else {
                eprintln!("warn: skipping malformed line in {}", path.display());
                continue;
            };
            let h = Held {
                canonical: canonical.to_string(),
                author: author.to_string(),
                at: held_at,
                locator: locator.to_string(),
            };
            match at.get(locator) {
                Some(&i) => entries[i] = h,
                None => {
                    at.insert(locator.to_string(), entries.len());
                    entries.push(h);
                }
            }
        }
        Ok(Self { entries })
    }

    pub fn save(&self, path: &Path, tmp: &Path) -> Result<()> {
        let body: String = self
            .entries
            .iter()
            .map(|h| {
                // An author id is taken from room data: strip what would forge a
                // column or a line.
                let author = h.author.replace(['\t', '\n', '\r'], " ");
                format!("{}\t{author}\t{}\t{}\n", h.canonical, h.at, h.locator)
            })
            .collect();
        fs::write(tmp, body).with_context(|| format!("writing {}", tmp.display()))?;
        fs::rename(tmp, path).with_context(|| format!("renaming onto {}", path.display()))
    }

    /// Hold (or re-hold) `h`, replacing any earlier entry for its locator.
    /// Returns the entries dropped to stay within [`MAX_HELD`], oldest first.
    pub fn hold(&mut self, h: Held) -> Vec<Held> {
        self.entries.retain(|e| e.locator != h.locator);
        self.entries.push(h);
        let mut dropped = Vec::new();
        if self.entries.len() > MAX_HELD {
            self.entries.sort_by_key(|e| e.at);
            let excess = self.entries.len() - MAX_HELD;
            dropped = self.entries.drain(..excess).collect();
        }
        dropped
    }

    /// Remove and return every entry whose canonical is not in `live`, or that
    /// has been held for [`HELD_RECHECK_SECS`].
    pub fn release(&mut self, live: &HashSet<String>, now: u64) -> Vec<Held> {
        let (gone, kept) = std::mem::take(&mut self.entries)
            .into_iter()
            .partition(|h| {
                !live.contains(&h.canonical) || now.saturating_sub(h.at) >= HELD_RECHECK_SECS
            });
        self.entries = kept;
        gone
    }

    pub fn remove(&mut self, loc: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|h| h.locator != loc);
        before != self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SALT: u64 = 0x5eed;
    const GOLDEN_SALTED: u64 = 0xe814_5495_2b96_7bf4;

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

    fn sk(t: &str) -> Sketch {
        Sketch::of(t, SALT).expect("enough text")
    }

    fn fp_text(t: &str) -> Fingerprint {
        Fingerprint {
            sketch: Sketch::of(t, SALT),
            ..Default::default()
        }
    }

    /// `n` distinct words `w{start}..`, which make `n - 4` distinct shingles.
    fn run(start: usize, n: usize) -> String {
        (start..start + n)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn live(locs: &[&str]) -> HashSet<String> {
        locs.iter().map(|s| s.to_string()).collect()
    }

    fn stored(loc: &str, first_seen: u64, print: Fingerprint) -> Stored {
        Stored {
            locator: loc.into(),
            first_seen,
            print,
        }
    }

    const A: &str = "freenet:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/";
    const B: &str = "freenet:BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB/";
    const C: &str = "freenet:CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC/";

    #[test]
    fn a_scam_clone_that_swaps_the_payment_address_is_a_near_duplicate() {
        let orig = fp_text(&seller_page("bc1qoriginalsellerxxxxxxxxxxxxxxxx"));
        let clone = fp_text(&seller_page("bc1qscammerzzzzzzzzzzzzzzzzzzzzzz"));
        let j = clone
            .sketch
            .as_ref()
            .unwrap()
            .jaccard(orig.sketch.as_ref().unwrap());
        assert!(j >= NEAR_DUP_JACCARD, "jaccard {j}");
        assert!(j < 1.0, "the address change must be visible, got {j}");
        let store = vec![stored(A, 100, orig)];
        let m = find_canonical(B, &clone, &store, &live(&[A])).expect("clone caught");
        assert_eq!(m.canonical, A);
        assert!(matches!(m.signal, Signal::NearText(_)));
    }

    #[test]
    fn a_padded_clone_is_caught_by_containment() {
        let orig = seller_page("bc1qoriginal");
        let mut clone = seller_page("bc1qscammer");
        // About 40% more text of its own: Jaccard alone no longer clears 0.8.
        for i in 0..16 {
            clone.push_str(&format!(
                " Frequently asked question {i}: how long does shipping take to \
                 region {i}, and can orders be combined with a friend's order?"
            ));
        }
        let (o, c) = (fp_text(&orig), fp_text(&clone));
        let j = c
            .sketch
            .as_ref()
            .unwrap()
            .jaccard(o.sketch.as_ref().unwrap());
        assert!(
            j < NEAR_DUP_JACCARD,
            "test needs padding past the jaccard bar, got {j}"
        );
        let m = find_canonical(B, &c, &[stored(A, 1, o)], &live(&[A])).expect("padded clone");
        assert!(matches!(m.signal, Signal::Contains(_)), "{m:?}");
    }

    #[test]
    fn a_page_quoted_whole_inside_a_much_larger_one_is_not_merged_into_it() {
        // A directory that quotes a small site in full is not that site.
        let small = seller_page("addr");
        let big = format!("{small} {}", run(0, 2000));
        let m = find_canonical(
            B,
            &fp_text(&big),
            &[stored(A, 1, fp_text(&small))],
            &live(&[A]),
        );
        assert_eq!(m, None);
    }

    #[test]
    fn invisible_characters_and_homoglyphs_do_not_hide_a_clone() {
        let orig = seller_page("addr");
        // A soft hyphen inside every word, and Cyrillic а/е/о for Latin a/e/o.
        let disguised: String = orig
            .split(' ')
            .map(|w| {
                let mut s: String = w
                    .chars()
                    .map(|c| match c {
                        'a' => 'а',
                        'e' => 'е',
                        'o' => 'о',
                        c => c,
                    })
                    .collect();
                if s.chars().count() > 2 {
                    let mid = s.char_indices().nth(1).unwrap().0;
                    s.insert(mid, '\u{00AD}');
                }
                s
            })
            .collect::<Vec<_>>()
            .join(" ");
        assert_ne!(disguised, orig);
        assert_eq!(sk(&disguised).jaccard(&sk(&orig)), 1.0);
    }

    /// The lookalikes `deunicode` alone transliterates to a DIFFERENT letter.
    #[test]
    fn confusables_fold_to_the_letter_they_imitate() {
        let orig = seller_page("addr");
        let disguised: String = orig
            .chars()
            .map(|c| match c {
                'c' => 'с', // Cyrillic es
                'p' => 'р', // Cyrillic er
                'y' => 'у', // Cyrillic u
                'x' => 'х', // Cyrillic ha
                c => c,
            })
            .collect();
        assert_ne!(disguised, orig);
        assert_eq!(sk(&disguised).jaccard(&sk(&orig)), 1.0);
    }

    /// Invisible characters that `deunicode` would turn into a space or a
    /// placeholder rather than drop, inside every word.
    #[test]
    fn invisible_characters_inside_words_are_dropped() {
        let orig = seller_page("addr");
        let disguised: String = orig
            .split(' ')
            .map(|w| {
                let mut s = String::new();
                for (i, c) in w.chars().enumerate() {
                    if i == 1 {
                        s.push('\u{200B}');
                        s.push('\u{E0041}');
                    }
                    s.push(c);
                }
                s
            })
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(sk(&disguised).jaccard(&sk(&orig)), 1.0);
    }

    #[test]
    fn the_threshold_is_inclusive_and_exact_below_k() {
        // 94 words = 90 shingles each. Offset 10 shares 80 of a 100-shingle union:
        // J = 0.8 exactly. Offset 11 shares 79 of 101: J = 0.782.
        let a = fp_text(&run(0, 94));
        let at = fp_text(&run(10, 94));
        let below = fp_text(&run(11, 94));
        assert_eq!(
            at.sketch
                .as_ref()
                .unwrap()
                .jaccard(a.sketch.as_ref().unwrap()),
            0.8
        );
        let store = [stored(A, 1, a)];
        let m = find_canonical(B, &at, &store, &live(&[A])).expect("J = 0.8 matches");
        assert_eq!(m.signal, Signal::NearText(0.8));
        assert_eq!(find_canonical(B, &below, &store, &live(&[A])), None);
    }

    #[test]
    fn distinct_pages_with_shared_boilerplate_are_not_duplicates() {
        let chrome = "Home About Contact Powered by Freenet. This site is \
                      published on a decentralized network and cannot be \
                      censored. ";
        let a = format!(
            "{chrome}Recipes for sourdough bread, rye crackers and a \
             slow-fermented focaccia, with notes on hydration, flour choice and \
             oven heat. Each recipe lists weights in grams and timings for a cold \
             kitchen. The starter section explains feeding ratios and how to \
             revive a neglected jar after a month in the fridge."
        );
        let b = format!(
            "{chrome}Notes on running a Freenet gateway on a small VPS: firewall \
             rules, systemd units, log rotation and how to read the peer \
             connection table when the node will not bootstrap. The \
             troubleshooting section covers NAT, clock skew and running out of \
             file descriptors."
        );
        assert_eq!(signal(B, &fp_text(&a), A, &fp_text(&b)), None);
    }

    #[test]
    fn same_owner_needs_the_text_to_agree() {
        let with = |owner: &str, t: Option<&str>| Fingerprint {
            owner: Some(owner.repeat(32)),
            sketch: t.and_then(|t| Sketch::of(t, SALT)),
            ..Default::default()
        };
        let page = seller_page("x");
        // A republish of the same site.
        assert_eq!(
            signal(B, &with("ab", Some(&page)), A, &with("ab", Some(&page))),
            Some(Signal::SameOwner)
        );
        // An impostor that names the owner's key but has no text to compare
        // cannot get the owner's real site held.
        assert_eq!(
            signal(B, &with("ab", Some(&page)), A, &with("ab", None)),
            None
        );
        assert_eq!(
            signal(B, &with("ab", None), A, &with("ab", Some(&page))),
            None
        );
        // A different site under the same key is not merged.
        assert_eq!(
            signal(
                B,
                &with("ab", Some(&run(0, 300))),
                A,
                &with("ab", Some(&page))
            ),
            None
        );
        // What the owner signal adds over text alone: J about 0.5 is not a text
        // match, but with the same owner it is the same publisher's site.
        let (x, y) = (run(0, 104), run(33, 104));
        assert_eq!(
            signal(B, &with("ab", Some(&x)), A, &with("ab", Some(&y))),
            Some(Signal::SameOwner)
        );
        assert_eq!(
            signal(B, &with("ab", Some(&x)), A, &with("cd", Some(&y))),
            None
        );
    }

    #[test]
    fn owner_and_archive_say_nothing_between_resources_of_one_app() {
        let shared = Fingerprint {
            owner: Some("ab".repeat(32)),
            archive: Some("11".repeat(32)),
            sketch: Some(sk(&run(0, 104))),
            page_sketch: None,
        };
        let other = Fingerprint {
            sketch: Some(sk(&run(33, 104))),
            ..shared.clone()
        };
        let (d1, d2) = ("app:delta/AWPjDQdKey", "app:delta/9CiJipmep5");
        assert_eq!(signal(d2, &other, d1, &shared), None);
        assert_eq!(signal(B, &other, A, &shared), Some(Signal::SameOwner));
    }

    #[test]
    fn identical_archive_merges_unless_the_text_is_clearly_different() {
        let with = |owner: &str, t: &str| Fingerprint {
            owner: Some(owner.repeat(32)),
            archive: Some("11".repeat(32)),
            sketch: Sketch::of(t, SALT),
            page_sketch: None,
        };
        let page = seller_page("x");
        assert_eq!(
            signal(B, &with("bb", &page), A, &with("aa", &page)),
            Some(Signal::SameArchive)
        );
        // A generic container shell that loads different content per site.
        assert_eq!(
            signal(B, &with("bb", &run(0, 300)), A, &with("aa", &page)),
            None
        );
    }

    #[test]
    fn the_whole_page_sketch_catches_a_clone_with_a_decoy_content_region() {
        let copied = seller_page("bc1qscammer");
        let orig = Fingerprint {
            sketch: Sketch::of(&seller_page("bc1qreal"), SALT),
            page_sketch: Sketch::of(&seller_page("bc1qreal"), SALT),
            ..Default::default()
        };
        let clone = Fingerprint {
            // `<main>` holds an unrelated blurb; the copy sits beside it.
            sketch: Sketch::of(&run(0, 60), SALT),
            page_sketch: Sketch::of(&format!("{} {copied}", run(0, 60)), SALT),
            ..Default::default()
        };
        assert!(find_canonical(B, &clone, &[stored(A, 1, orig)], &live(&[A])).is_some());
    }

    /// The canonical is an app resource, which has a content sketch only, and the
    /// clone is a contract of its own that hides the copy outside its content
    /// region. Only the cross pair sees it.
    #[test]
    fn a_decoy_clone_of_an_app_resource_is_caught_by_the_cross_pair() {
        let orig = fp_text(&seller_page("bc1qreal"));
        let clone = Fingerprint {
            sketch: Sketch::of(&run(0, 60), SALT),
            page_sketch: Sketch::of(&format!("{} {}", run(0, 60), seller_page("bc1qscam")), SALT),
            ..Default::default()
        };
        let canon = "app:delta/AWPjDQdKey";
        assert!(find_canonical(B, &clone, &[stored(canon, 1, orig)], &live(&[canon])).is_some());
    }

    /// Two distinct sites from one theme: big shared chrome, different content.
    #[test]
    fn sites_sharing_a_theme_are_not_merged_on_their_chrome() {
        let chrome = run(0, 1500);
        let site = |start: usize| Fingerprint {
            sketch: Sketch::of(&run(start, 60), SALT),
            page_sketch: Sketch::of(&format!("{chrome} {}", run(start, 60)), SALT),
            ..Default::default()
        };
        let (a, b) = (site(10_000), site(20_000));
        assert!(
            a.page_sketch
                .as_ref()
                .unwrap()
                .jaccard(b.page_sketch.as_ref().unwrap())
                >= NEAR_DUP_JACCARD,
            "the test needs whole pages that look alike"
        );
        assert_eq!(signal(B, &b, A, &a), None);
    }

    #[test]
    fn containment_needs_jaccard_one_half_and_is_exact_below_k() {
        // 90 canonical shingles, all inside the candidate.
        let canon = fp_text(&run(0, 94));
        let store = [stored(A, 1, canon)];
        // 180 shingles: J = 90/180 = 0.5 exactly.
        let m = find_canonical(B, &fp_text(&run(0, 184)), &store, &live(&[A]));
        assert_eq!(m.map(|m| m.signal), Some(Signal::Contains(1.0)));
        // 181 shingles: J just under 0.5.
        assert_eq!(
            find_canonical(B, &fp_text(&run(0, 185)), &store, &live(&[A])),
            None
        );
    }

    #[test]
    fn unknown_fields_never_match() {
        let empty = Fingerprint::default();
        assert_eq!(signal(B, &empty, A, &empty), None);
    }

    #[test]
    fn earliest_live_entry_is_canonical() {
        let text = seller_page("addr");
        let store = vec![
            stored(B, 200, fp_text(&text)),
            stored(A, 100, fp_text(&text)),
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
    fn what_is_and_is_not_compared() {
        assert!(!comparable(A, A));
        assert!(!comparable(
            "freenet:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/a.html",
            "freenet:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/b.html"
        ));
        assert!(!comparable(
            "freenet:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "freenet:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/"
        ));
        // One app resource and a deep link into it.
        assert!(!comparable(
            "app:delta/AWPjDQdKey",
            "app:delta/AWPjDQdKey/page2"
        ));
        // Different resources of one app ARE compared, on text: a copied Delta
        // seller page is the cheapest clone there is.
        assert!(comparable("app:delta/AWPjDQdKey", "app:delta/9CiJipmep5"));
        assert!(comparable("app:delta/AWPjDQdKey", B));
        assert!(comparable(A, B));

        let text = seller_page("addr");
        let store = vec![stored("app:delta/AWPjDQdKey", 1, fp_text(&text))];
        let l = live(&["app:delta/AWPjDQdKey"]);
        assert!(find_canonical("app:delta/9CiJipmep5", &fp_text(&text), &store, &l).is_some());
        assert_eq!(
            find_canonical("app:delta/AWPjDQdKey/2", &fp_text(&text), &store, &l),
            None
        );
    }

    #[test]
    fn too_little_text_cannot_decide_on_its_own() {
        assert!(!sk("Enter your password to continue").usable());
        assert_eq!(Sketch::of("", SALT), None);
        assert_eq!(Sketch::of("four words only here", SALT), None);
        assert!(!sk(&run(0, 33)).usable(), "29 shingles");
        assert!(sk(&run(0, 34)).usable(), "30 shingles");
        // Two short pages that happen to be identical are not a verdict.
        let short = fp_text("Enter your password to continue to the site");
        assert_eq!(signal(B, &short, A, &short), None);
    }

    /// The clone shapes that rest on containment must be caught under EVERY salt,
    /// not under the one a test happens to pin: the salt is random per
    /// installation, and an estimate that depends on it is a coin toss.
    #[test]
    fn containment_verdicts_do_not_depend_on_the_salt() {
        let blurb = "Meadowbrook Farm heirloom seeds, shipped anywhere.";
        let page = |addr: &str| format!("{blurb} {} {}", seller_page(addr), run(0, 2000));
        for salt in 0..200u64 {
            let s = |t: &str| Sketch::of(t, salt);
            // Thin content region, whole-page copy.
            let orig = Fingerprint {
                sketch: s(blurb).map(Sketch::without_full),
                page_sketch: s(&page("bc1qreal")).map(Sketch::without_full),
                ..Default::default()
            };
            let clone = Fingerprint {
                sketch: s(blurb),
                page_sketch: s(&page("bc1qscam")),
                ..Default::default()
            };
            assert!(
                signal(B, &clone, A, &orig).is_some(),
                "thin-content copy, salt {salt}"
            );
            // Decoy content region in front of a copy of an app resource.
            let app = fp_text(&seller_page("bc1qreal"));
            let app = Fingerprint {
                sketch: app
                    .sketch
                    .map(|_| s(&seller_page("bc1qreal")).unwrap().without_full()),
                ..app
            };
            let decoy = Fingerprint {
                sketch: s(&run(5000, 60)),
                page_sketch: s(&format!("{} {}", run(5000, 60), seller_page("bc1qscam"))),
                ..Default::default()
            };
            assert!(
                signal(B, &decoy, "app:delta/AWPjDQdKey", &app).is_some(),
                "decoy clone, salt {salt}"
            );
            // Theme siblings sharing chrome, distinct thin content.
            let sib = |start: usize| Fingerprint {
                sketch: s(&run(start, 8)),
                page_sketch: s(&format!("{} {}", run(0, 1500), run(start, 8))),
                ..Default::default()
            };
            let stored_sib = sib(10_000).stored();
            assert_eq!(
                signal(B, &sib(20_000), A, &stored_sib),
                None,
                "theme, salt {salt}"
            );
        }
    }

    /// The canonical's content region is a short blurb (too short to decide on)
    /// and everything else is beside it. A whole-page copy still contains the
    /// blurb, which lets the whole-page pair decide.
    #[test]
    fn a_whole_page_copy_of_a_site_with_a_thin_content_region_is_caught() {
        let blurb = "Meadowbrook Farm heirloom seeds, shipped anywhere.";
        let page = |addr: &str| format!("{blurb} {}", seller_page(addr));
        let orig = Fingerprint {
            sketch: Sketch::of(blurb, SALT),
            page_sketch: Sketch::of(&page("bc1qreal"), SALT),
            ..Default::default()
        };
        assert!(!orig.sketch.as_ref().unwrap().usable());
        let clone = Fingerprint {
            sketch: Sketch::of(blurb, SALT),
            page_sketch: Sketch::of(&page("bc1qscam"), SALT),
            ..Default::default()
        };
        assert!(matches!(
            signal(B, &clone, A, &orig),
            Some(Signal::NearText(_))
        ));
    }

    #[test]
    fn jaccard_is_symmetric_and_one_on_itself() {
        let a = sk(&seller_page("one"));
        let b = sk(&seller_page("two"));
        assert_eq!(a.jaccard(&a), 1.0);
        assert_eq!(a.jaccard(&b), b.jaccard(&a));
        assert_eq!(a.contained_in(&a), 1.0);
    }

    #[test]
    fn jaccard_estimate_tracks_the_true_value_on_large_pages() {
        // 2000 shingles each, sharing 1000: true J = 1/3.
        let a = sk(&run(0, 2004));
        let b = sk(&run(1000, 2004));
        let j = a.jaccard(&b);
        assert!((j - 1.0 / 3.0).abs() < 0.08, "estimate {j}");
    }

    #[test]
    fn a_full_sketch_against_a_partial_one() {
        // 1000 shingles against a 100-shingle subset of them: J = 0.1, and the
        // small page is wholly contained in the large one.
        let big = sk(&run(0, 1004));
        let small = sk(&run(0, 104));
        assert_eq!(big.mins.len(), SKETCH_K);
        assert!(small.mins.len() < SKETCH_K);
        let j = small.jaccard(&big);
        assert!((j - 0.1).abs() < 0.05, "estimate {j}");
        let c = small.contained_in(&big);
        assert!(c > 0.85, "containment {c}");
        assert!(big.contained_in(&small) < 0.2);
    }

    #[test]
    fn the_salt_changes_the_hash_and_the_hash_is_stable() {
        // Pinned values: these are persisted, so a change to the tokenizer or
        // the hash silently breaks every stored sketch.
        assert_eq!(shingle_hash("a b c d e", 0), 0x2d79_2dac_f53d_be19);
        // And at a nonzero salt, which pins WHERE the salt enters the hash.
        assert_eq!(shingle_hash("a b c d e", 0x5eed), GOLDEN_SALTED);
        assert_ne!(shingle_hash("a b c d e", 0), shingle_hash("a b c d e", 1));
        assert_eq!(words("Ｈéllo,\u{200B}Wörld"), vec!["hello", "world"]);
    }

    #[test]
    fn owner_needs_32_bytes_of_params_and_a_web_container_state() {
        let info =
            |p: &str, a: Option<&str>| serde_json::json!({ "params_hex": p, "archive_blake3": a });
        let arch = "11".repeat(32);
        assert_eq!(
            Fingerprint::from_contract_info(&info(&"Ab".repeat(32), Some(&arch))).owner,
            Some("ab".repeat(32))
        );
        // Not a web container: params are not known to mean "owner".
        assert_eq!(
            Fingerprint::from_contract_info(&info(&"ab".repeat(32), None)).owner,
            None
        );
        // The Atlas index's own params are root_vk || slug: not an owner key.
        assert_eq!(
            Fingerprint::from_contract_info(&info(&"ab".repeat(39), Some(&arch))).owner,
            None
        );
        assert_eq!(
            Fingerprint::from_contract_info(&serde_json::json!({})),
            Fingerprint::default()
        );
    }

    /// The keys `atlasctl contract-info` prints, and this module reads, must
    /// agree, or the owner and archive signals switch off silently.
    #[test]
    fn contract_info_keys_match_the_cli() {
        let cli = include_str!("../../cli/src/main.rs");
        for key in ["\"params_hex\"", "\"archive_blake3\""] {
            assert!(cli.contains(key), "cli no longer prints {key}");
        }
    }

    #[test]
    fn store_round_trips_keeps_its_salt_and_later_lines_win() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fp.txt");
        let mut store = Store::load(&path).unwrap();
        assert!(store.entries.is_empty());
        let salt = store.salt;
        let s1 = stored(
            A,
            5,
            Fingerprint {
                owner: Some("aa".repeat(32)),
                archive: None,
                sketch: Sketch::of(&seller_page("x"), salt),
                page_sketch: Sketch::of(&run(0, 50), salt),
            },
        );
        store.record(&path, s1.clone()).unwrap();
        let s2 = stored(A, 5, Fingerprint::default());
        store.record(&path, s2.clone()).unwrap();
        let back = Store::load(&path).unwrap();
        assert_eq!((back.salt, back.entries), (salt, vec![s2.clone()]));

        let tmp = dir.path().join("fp.tmp");
        Store {
            salt,
            entries: vec![s1.clone()],
        }
        .save(&path, &tmp)
        .unwrap();
        assert_eq!(Store::load(&path).unwrap().entries, vec![s1]);

        std::fs::write(&path, format_line(&s2)).unwrap();
        assert!(Store::load(&path).is_err(), "no salt header");
        std::fs::write(&path, format!("{SALT_HEADER}{salt:016x}\tv0\n")).unwrap();
        assert!(Store::load(&path).is_err(), "another text pipeline");
        std::fs::write(&path, format!("{SALT_HEADER}{salt:016x}\n")).unwrap();
        assert!(Store::load(&path).is_err(), "no pipeline version");
        std::fs::create_dir(dir.path().join("dir")).unwrap();
        assert!(
            Store::load(&dir.path().join("dir")).is_err(),
            "unreadable is not empty"
        );
    }

    #[test]
    fn corrupt_sketches_are_skipped_not_trusted() {
        let good = sk(&seller_page("x")).encode();
        let (n, hex) = good.split_once(':').unwrap();
        let first = &hex[..16];
        let second = &hex[16..32];
        let bad = [
            format!("{n}:{second}{first}"),       // unsorted
            format!("{n}:{first}{first}"),        // duplicate
            format!("{n}:{}", first.repeat(257)), // > K (and duplicate)
            format!("{n}:{}", "zz".repeat(8)),    // not hex
            format!("{n}:{}", &hex[..15]),        // odd length
            format!("{n}:{}é", &hex[..14]),       // multi-byte
            format!("x:{hex}"),                   // bad count
            format!("3:{hex}"),                   // count below sketch size
            hex.to_string(),                      // no count
        ];
        for b in bad {
            assert_eq!(Sketch::decode(&b), None, "{b:.40}");
        }
        assert_eq!(Sketch::decode(&good), Some(sk(&seller_page("x"))));
    }

    #[test]
    fn held_duplicates_are_released_when_their_canonical_leaves() {
        let dir = tempfile::tempdir().unwrap();
        let (path, tmp) = (dir.path().join("held.txt"), dir.path().join("held.tmp"));
        let mut held = HeldStore::load(&path).unwrap();
        held.hold(Held {
            canonical: A.into(),
            author: "bob\tforged".into(),
            at: 100,
            locator: B.into(),
        });
        held.hold(Held {
            canonical: B.into(),
            author: "carol".into(),
            at: 100,
            locator: C.into(),
        });
        held.save(&path, &tmp).unwrap();
        let mut held = HeldStore::load(&path).unwrap();
        assert_eq!(held.entries.len(), 2);
        assert_eq!(held.entries[0].author, "bob forged");
        assert_eq!(held.entries[0].at, 100);
        assert!(held.entries.iter().any(|h| h.locator == B));
        let gone = held.release(&live(&[B]), 101);
        assert_eq!(
            gone.iter().map(|h| h.locator.as_str()).collect::<Vec<_>>(),
            vec![B]
        );
        assert_eq!(held.entries.len(), 1);
        assert!(held.remove(C));
        assert!(!held.remove(C));
    }

    #[test]
    fn a_held_duplicate_is_judged_again_after_the_recheck_period() {
        let mut held = HeldStore {
            entries: Vec::new(),
        };
        held.hold(Held {
            canonical: A.into(),
            author: "x".into(),
            at: 1_000,
            locator: B.into(),
        });
        let l = live(&[A]);
        assert!(held.release(&l, 1_000 + HELD_RECHECK_SECS - 1).is_empty());
        assert_eq!(held.release(&l, 1_000 + HELD_RECHECK_SECS).len(), 1);
    }

    #[test]
    fn the_held_list_is_bounded_dropping_the_oldest() {
        let mut held = HeldStore {
            entries: Vec::new(),
        };
        for i in 0..MAX_HELD {
            assert!(held
                .hold(Held {
                    canonical: A.into(),
                    author: "x".into(),
                    at: 10 + i as u64,
                    locator: format!("{B}{i}"),
                })
                .is_empty());
        }
        let dropped = held.hold(Held {
            canonical: A.into(),
            author: "x".into(),
            at: 5,
            locator: C.into(),
        });
        // The new entry is the oldest by `at`, so it is the one dropped.
        assert_eq!(dropped.iter().map(|h| h.at).collect::<Vec<_>>(), vec![5]);
        assert_eq!(held.entries.len(), MAX_HELD);
    }

    #[test]
    fn record_refuses_to_append_under_another_salt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fp.txt");
        let mut mine = Store::fresh();
        let mut theirs = Store::fresh();
        theirs.salt = mine.salt ^ 1;
        theirs
            .record(&path, stored(A, 1, Fingerprint::default()))
            .unwrap();
        assert!(mine
            .record(&path, stored(B, 1, Fingerprint::default()))
            .is_err());
        assert_eq!(Store::load(&path).unwrap().entries.len(), 1);
    }

    #[test]
    fn report_lists_duplicates_canonical_first_and_the_margin() {
        let text = seller_page("addr");
        let r = report(&[
            stored(B, 2, fp_text(&text)),
            stored(A, 1, fp_text(&text)),
            stored(C, 3, fp_text(&run(0, 200))),
        ]);
        assert!(r.contains(&format!("DUPLICATE {B} of {A}")), "{r}");
        assert!(
            r.contains("3 comparable pairs; 1 would be duplicates"),
            "{r}"
        );
        assert!(r.contains(C), "{r}");
    }
}
