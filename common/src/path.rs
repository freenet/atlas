//! Path guards for locator validation.
//!
//! These are enforced by the index contract on untrusted peers, so they are the
//! authoritative rules — not a best-effort filter. Two separate escape
//! primitives matter, and neither finds the other:
//!
//! - **Traversal** (`..`), which walks out of the contract root.
//! - **Absolute-path escape** (a leading `//`), which contains no dots at all.
//!   The node splits `<key>/<path>` and hands the remainder to `Path::join`, and
//!   `join` with an absolute path DISCARDS THE BASE, so a locator path of
//!   `//home/user/.ssh/id_ed25519` reads that file rather than something under
//!   the contract root.
//!
//! Both are checked after decoding percent-escapes **to a fixed point**, because
//! `%2e%2e`, `..%2f`, `%252e%252e` and friends are the same attack wearing a
//! coat. A single decode pass is not enough: `%252e` decodes to `%2e`, which
//! decodes to `.`.
//!
//! The crawler carries its own copies of these guards (`has_dot_segment`,
//! `is_absolute_contract_path`) which predate this module and are equivalent.
//! They are being switched over to these so the two cannot drift; until then,
//! treat THIS module as canonical, since it is the one the contract enforces.

/// True if any path segment is `.` or `..` after full percent-decoding.
/// Splits on `\` as well as `/`: a backslash is a separator on Windows hosts and
/// some URL parsers normalise it to `/`, so treating it as an ordinary character
/// would leave `/..\x` unguarded.
pub fn has_dot_segment(path: &str) -> bool {
    percent_decode_fully(path.as_bytes())
        .split(|b| *b == b'/' || *b == b'\\')
        .any(|seg| seg == b"." || seg == b"..")
}

/// True if the path escapes the contract root by being ABSOLUTE rather than by
/// traversing, i.e. a second separator immediately follows the first.
///
/// An interior `//` (`/a//b`) is harmless: it stays under the base. A lone
/// trailing slash is the ordinary root form. Only the leading case escapes.
pub fn is_absolute_escape(path: &str) -> bool {
    let decoded = percent_decode_fully(path.as_bytes());
    let sep = |b: u8| b == b'/' || b == b'\\';
    matches!((decoded.first(), decoded.get(1)), (Some(&a), Some(&b)) if sep(a) && sep(b))
}

/// True if the string contains an ASCII control character (including CR, LF and
/// NUL) after full percent-decoding. Such characters have no place in a locator
/// and are how header/log injection and JSON-corruption tricks get carried.
pub fn has_control_char(s: &str) -> bool {
    percent_decode_fully(s.as_bytes())
        .iter()
        .any(|b| b.is_ascii_control())
}

/// Decode `%XX` escapes repeatedly until the result stops changing, bounded by
/// the input length so it always terminates.
pub fn percent_decode_fully(input: &[u8]) -> Vec<u8> {
    let mut cur = percent_decode_once(input);
    for _ in 0..input.len() {
        let next = percent_decode_once(&cur);
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

/// One pass of `%XX` decoding (either case). Invalid escapes are left as-is;
/// this is a guard, not a general-purpose decoder.
///
/// Works on bytes end to end and never slices a `&str`. Indexing a `&str` by the
/// byte offsets of a `%XX` triple panics when the `%` is followed by a
/// multi-byte character (`%aé` puts the end of the triple inside `é`), and the
/// input here is untrusted, so that would be a remote panic inside the contract.
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
            "/..\\x",
            "/%2e/x",
            "/a/./b",
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
        ] {
            assert!(!has_dot_segment(ok), "{ok:?} should be allowed");
            assert!(!is_absolute_escape(ok), "{ok:?} should be allowed");
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
        for s in ["%aé", "%", "%2", "%%%", "é%2e%", "%e9%", "%zz"] {
            let _ = has_dot_segment(s);
            let _ = is_absolute_escape(s);
            let _ = has_control_char(s);
        }
    }

    /// Decoding must reach a fixed point, not stop after one pass.
    #[test]
    fn decoding_reaches_a_fixed_point() {
        assert_eq!(percent_decode_fully(b"%252e"), b".");
        assert_eq!(percent_decode_fully(b"%25252e"), b".");
    }
}
