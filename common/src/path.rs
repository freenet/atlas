//! Path guards for locator validation.
//!
//! These are enforced by the index contract on untrusted peers, so they are the
//! authoritative rules, not a best-effort filter. Two separate escape primitives
//! matter, and neither finds the other:
//!
//! - **Traversal** (`..`), which walks out of the contract root.
//! - **Absolute-path escape** (a leading `//`), which contains no dots at all.
//!   The node splits `<key>/<path>` and hands the remainder to `Path::join`, and
//!   `join` with an absolute path DISCARDS THE BASE, so a locator path of
//!   `//home/user/.ssh/id_ed25519` reads that file rather than something under
//!   the contract root.
//!
//! Both are checked after decoding percent-escapes, because `%2e%2e`, `..%2f`,
//! `%252e%252e` and friends are the same attack wearing a coat. One decode pass
//! is not enough: `%252e` decodes to `%2e`, which decodes to `.`.
//!
//! The crawler carries its own copies of these guards (`has_dot_segment`,
//! `is_absolute_contract_path`) which predate this module and are equivalent.
//! They are being switched over to these so the two cannot drift; until then,
//! treat THIS module as canonical, since it is the one the contract enforces.

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
/// traversing, i.e. a second separator immediately follows the first.
///
/// An interior `//` (`/a//b`) is harmless: it stays under the base. A lone
/// trailing slash is the ordinary root form. Only the leading case escapes.
///
/// Fails CLOSED on an input that will not converge.
pub fn is_absolute_escape(path: &str) -> bool {
    match decode_bounded(path.as_bytes()) {
        None => true,
        Some(d) => {
            let sep = |b: u8| b == b'/' || b == b'\\';
            matches!((d.first(), d.get(1)), (Some(&a), Some(&b)) if sep(a) && sep(b))
        }
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
/// NOTE on overlong UTF-8 (`%c0%ae` for `.`): this decodes to the bytes `c0 ae`,
/// which is not `.`, so it is not treated as a dot segment. That is correct for
/// our consumers, since browsers and the Rust gateway reject overlong sequences
/// rather than folding them to ASCII. It WOULD be a bypass against a consumer
/// that folds them (classically IIS), so if a locator is ever handed to one,
/// reject non-ASCII decoded bytes here too.
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
        ] {
            assert!(is_absolute_escape(bad), "{bad:?} should be caught");
        }
        for ok in ["/a//b", "/", "/a/", "/#x//y"] {
            assert!(!is_absolute_escape(ok), "{ok:?} should be allowed");
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
        }
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

    /// Overlong UTF-8 is deliberately NOT folded to `.` (see `decode_bounded`).
    /// Pin the actual behaviour so the limitation is explicit rather than assumed.
    #[test]
    fn overlong_utf8_is_not_treated_as_a_dot_segment() {
        assert!(!has_dot_segment("/%c0%ae%c0%ae/x"));
    }
}
