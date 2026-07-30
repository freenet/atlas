//! Path guards for locator validation.
//!
//! These are enforced by the index contract on untrusted peers, so they are the
//! authoritative rules, not a best-effort filter. Three independent escape
//! primitives matter, and no one of them finds the others:
//!
//! - **Traversal** (`..`), which walks out of the contract root.
//! - **Absolute-path escape**, which contains no dots at all: a leading `//`, or
//!   a Windows drive prefix (`/C:/...`). The node splits `<key>/<path>` and hands
//!   the remainder to `Path::join`, and `join` with an absolute path DISCARDS THE
//!   BASE, so `//home/user/.ssh/id_ed25519` reads that file rather than something
//!   under the contract root.
//! - **Overlong UTF-8** (`%c0%ae` for `.`), which is a dot segment to a parser
//!   that decodes it and ordinary bytes to a guard that compares them.
//!
//! All are checked after decoding percent-escapes, because `%2e%2e`, `..%2f`,
//! `%252e%252e` and friends are the same attack wearing a coat. One decode pass
//! is not enough: `%252e` decodes to `%2e`, which decodes to `.`.
//!
//! The crawler carries its own copies of these guards (`has_dot_segment`,
//! `is_absolute_contract_path`), which predate this module. They are NOT
//! equivalent and have already drifted in both directions: the crawler catches
//! the Windows drive-prefix escape that this module originally missed, and folds
//! its control-character check into a differently-shaped function. This module is
//! the canonical one, because it is the one the CONTRACT enforces. Deduplicating
//! them is tracked separately; meanwhile the crawler carries a differential test
//! (`common_path_guards_are_at_least_as_strict_as_the_crawler_copies`) pinning the
//! property that matters: THIS module must never be weaker than the crawler copy,
//! or a path the crawler refuses could still be signed into the index by hand.

/// Max percent-decoding passes before an input is treated as hostile.
///
/// Two layers (`%252e` -> `%2e` -> `.`) is already an attack rather than a
/// mistake, and nothing legitimate needs more. The cap is also what keeps these
/// guards LINEAR: decoding to a true fixed point is O(n²) in the worst case, and
/// they run per-record inside `validate_state`, so an index near `MAX_ENTRIES`
/// full of deeply-nested escapes would otherwise be a cheap way to burn contract
/// CPU on every full-state validation.
const MAX_DECODE_PASSES: usize = 8;

/// True if any path segment is `.` or `..` after percent-decoding.
///
/// Splits on `\` as well as `/`: a backslash is a separator on Windows hosts and
/// some URL parsers normalise it to `/`, so treating it as an ordinary character
/// would leave `/..\x` unguarded.
///
/// Fails CLOSED on an input that will not converge (see [`decode_bounded`]).
pub fn has_dot_segment(path: &str) -> bool {
    match decode_bounded(path.as_bytes()) {
        None => true,
        Some(d) => d
            .split(|b| *b == b'/' || *b == b'\\')
            .any(|seg| seg == b"." || seg == b".."),
    }
}

/// True if the path escapes the contract root by being ABSOLUTE rather than by
/// traversing. Two forms, neither containing a dot:
///
/// - A second separator immediately following the first (`//etc/passwd`). An
///   interior `//` (`/a//b`) is harmless because it stays under the base, and a
///   lone trailing slash is the ordinary root form, so only the leading case
///   counts.
/// - A Windows drive prefix (`/C:/Windows/win.ini`, and even `/C:foo`). On a
///   Windows host `Path::join` discards the base for both, so this is the same
///   file-read primitive. Checked because a locator validated here may be served
///   by a node on any platform, and because the crawler's older copy of this
///   guard already checked it — this module claims to be canonical, so it has to
///   be at least as strict.
///
/// Fails CLOSED on an input that will not converge.
pub fn is_absolute_escape(path: &str) -> bool {
    match decode_bounded(path.as_bytes()) {
        None => true,
        Some(d) => {
            let sep = |b: u8| b == b'/' || b == b'\\';
            if matches!((d.first(), d.get(1)), (Some(&a), Some(&b)) if sep(a) && sep(b)) {
                return true;
            }
            // Strip a single leading separator, then look for `<letter>:`.
            let rest = match d.first() {
                Some(&b) if sep(b) => &d[1..],
                _ => &d[..],
            };
            matches!(rest, [d0, b':', ..] if d0.is_ascii_alphabetic())
        }
    }
}

/// True if the decoded path is not valid UTF-8.
///
/// This is what closes the overlong-UTF-8 class: `%c0%ae` is an overlong encoding
/// of `.`, and `%c1%9c` of `\`, so a parser that decodes them sees a dot segment
/// while a byte-compare guard does not. Rather than enumerate the encodings,
/// reject anything that is not well-formed UTF-8 — overlong forms never are, and a
/// legitimate locator path has no reason to be invalid UTF-8 either.
pub fn has_invalid_utf8(path: &str) -> bool {
    match decode_bounded(path.as_bytes()) {
        None => true,
        Some(d) => core::str::from_utf8(&d).is_err(),
    }
}

/// True if the string contains an ASCII control character (including CR, LF and
/// NUL) after percent-decoding. Such characters have no place in a locator and
/// are how header/log injection and JSON-corruption tricks get carried.
///
/// Fails CLOSED on an input that will not converge.
pub fn has_control_char(s: &str) -> bool {
    match decode_bounded(s.as_bytes()) {
        None => true,
        Some(d) => d.iter().any(|b| b.is_ascii_control()),
    }
}

/// Decode `%XX` escapes until the result stops changing, giving up after
/// [`MAX_DECODE_PASSES`].
///
/// Returns `None` when the input had not converged by the cap, which every caller
/// treats as hostile. That is deliberate: an input still shedding escape layers
/// after eight passes is not worth reasoning about further, and refusing it is
/// both cheaper and safer than deciding what it "really" says.
///
/// Note this decodes bytes, so an overlong UTF-8 form such as `%c0%ae` becomes
/// `c0 ae` rather than `.` and is invisible to a byte-compare guard.
/// [`has_invalid_utf8`] is what closes that class; `check_path` calls both.
pub fn decode_bounded(input: &[u8]) -> Option<Vec<u8>> {
    let mut cur = percent_decode_once(input);
    for _ in 0..MAX_DECODE_PASSES {
        let next = percent_decode_once(&cur);
        if next == cur {
            return Some(cur);
        }
        cur = next;
    }
    None
}

/// One pass of `%XX` decoding (either case). Invalid escapes are left as-is;
/// this is a guard, not a general-purpose decoder.
///
/// Works on bytes end to end and never slices a `&str`. Indexing a `&str` by the
/// byte offsets of a `%XX` triple panics when the `%` is followed by a multi-byte
/// character (`%aé` puts the end of the triple inside `é`), and the input here is
/// untrusted, so that would be a remote panic inside the contract.
fn percent_decode_once(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(hi), Some(lo)) = (hex_val(b[i + 1]), hex_val(b[i + 2])) {
                out.push(hi * 16 + lo);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traversal_is_caught_through_every_encoding() {
        for bad in [
            "/../x",
            "/a/../../b",
            "#/../x",
            "?next=/../x",
            "/%2e%2e/x",
            "/..%2fx",
            "/%2e%2e%2fx",
            "/%252e%252e/x",
            "/.%2e/x",
            "/%2e./x",
            "/..\\x",
            "/%2e/x",
            "/a/./b",
            "/A/%2E%2E/b",
        ] {
            assert!(has_dot_segment(bad), "{bad:?} should be caught");
        }
    }

    #[test]
    fn ordinary_paths_are_not_caught() {
        for ok in [
            "",
            "/",
            "/index.html",
            "/#AmcVD92D3U/2/links",
            "/a..b/c",
            "/...",
            "/a/b?q=1#frag",
            "/%20space",
            "/assets/delta-ui-dxh9e39964992bde68f.js",
        ] {
            assert!(!has_dot_segment(ok), "{ok:?} should be allowed");
            assert!(!is_absolute_escape(ok), "{ok:?} should be allowed");
            assert!(!has_control_char(ok), "{ok:?} should be allowed");
        }
    }

    #[test]
    fn absolute_escape_is_caught_but_interior_double_slash_is_not() {
        for bad in [
            "//etc/passwd",
            "/%2fetc",
            "/%252fetc",
            "//",
            "/\\x",
            "\\\\x",
            // Windows drive prefixes. `Path::join` discards the base for both the
            // rooted and the relative form, so each is the same file-read
            // primitive as `//`, with no dots involved.
            "/C:/Windows/win.ini",
            "/C:foo",
            "C:/x",
            "/%43:/x",
            "/c:/x",
            // A single letter followed by `:` IS a drive spec, even mid-looking
            // like an ordinary segment: `join("a:b/c")` discards the base too.
            "/a:b/c",
        ] {
            assert!(is_absolute_escape(bad), "{bad:?} should be caught");
        }
        // A colon that cannot be a drive prefix must still be allowed: more than
        // one leading letter, or a colon with no letter before it.
        for ok in ["/a//b", "/", "/a/", "/#x//y", "/ab:/c", "/:x", "/1:/x"] {
            assert!(!is_absolute_escape(ok), "{ok:?} should be allowed");
        }
    }

    /// Overlong UTF-8 is how a dot segment gets past a byte-compare guard on a
    /// parser that decodes it (`%c0%ae` is an overlong `.`, `%c1%9c` an overlong
    /// `\`). Rejecting all invalid UTF-8 kills the class without enumerating it.
    #[test]
    fn overlong_utf8_encodings_are_rejected() {
        for bad in [
            "/%c0%ae%c0%ae/x",
            "/%e0%80%ae/x",
            "/%c1%9c..%c1%9c",
            "/%25c0%25ae/x",
            "/%f0%80%80%ae/x",
        ] {
            assert!(has_invalid_utf8(bad), "{bad:?} should be caught");
        }
        // Legitimate multi-byte UTF-8 must survive.
        for ok in [
            "/ordinary",
            "/caf%c3%a9",
            "/#AmcVD92D3U/2/links",
            "/%20",
            "/café",
        ] {
            assert!(!has_invalid_utf8(ok), "{ok:?} should be allowed");
        }
    }

    #[test]
    fn control_chars_are_caught_through_encoding() {
        for bad in ["/a\nb", "/a%0Ab", "/a%250Ab", "/a\rb", "/a\0b", "/a\tb"] {
            assert!(has_control_char(bad), "{bad:?} should be caught");
        }
        assert!(!has_control_char("/ordinary/path?q=1#f"));
    }

    /// A `%` followed by a multi-byte char must not panic (it would be a remote
    /// panic inside the contract).
    #[test]
    fn decoding_never_panics_on_hostile_input() {
        for s in ["%aé", "%", "%2", "%%%", "é%2e%", "%e9%", "%zz", "%%2e"] {
            let _ = has_dot_segment(s);
            let _ = is_absolute_escape(s);
            let _ = has_control_char(s);
            let _ = has_invalid_utf8(s);
        }
        // Behavioural assertions too, so a decoder that returns garbage on
        // hostile input is not merely "did not panic".
        assert!(has_dot_segment("%2e"), "a valid escape must still decode");
        assert!(!has_dot_segment("%zz"), "an invalid escape is literal text");
        assert!(!has_dot_segment("%2"), "a truncated escape is literal text");
    }

    /// Decoding must keep going past one pass, not stop at the first layer.
    #[test]
    fn decoding_peels_more_than_one_layer() {
        assert_eq!(decode_bounded(b"%252e").unwrap(), b".");
        assert_eq!(decode_bounded(b"%25252e").unwrap(), b".");
    }

    /// An input that will not converge within the cap must be REFUSED rather than
    /// decoded further, and every guard must fail closed on it. The cap is what
    /// keeps these linear, so this is load-bearing for contract CPU too.
    #[test]
    fn a_non_converging_input_is_refused_by_every_guard() {
        // Nesting is `%` + "25"*(k-1) + "2e": each pass peels exactly one layer,
        // so level k needs k passes. (`"%25".repeat(n)` does NOT nest — it
        // collapses to a run of literal `%` within two passes.)
        let nest = |k: usize| format!("%{}2e", "25".repeat(k - 1));
        assert_eq!(decode_bounded(nest(3).as_bytes()).unwrap(), b".");

        let over = nest(MAX_DECODE_PASSES + 2);
        assert!(
            decode_bounded(over.as_bytes()).is_none(),
            "an input needing more than {MAX_DECODE_PASSES} passes must be refused"
        );
        assert!(has_dot_segment(&over), "must fail closed");
        assert!(is_absolute_escape(&over), "must fail closed");
        assert!(has_control_char(&over), "must fail closed");

        // Just inside the cap it still converges and is judged on its content.
        let under = nest(MAX_DECODE_PASSES - 1);
        assert_eq!(decode_bounded(under.as_bytes()).unwrap(), b".");
        assert!(has_dot_segment(&under));
    }
}
