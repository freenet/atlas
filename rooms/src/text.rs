//! Sanitising owner-supplied room text before it reaches an index entry.
//!
//! Room names and descriptions are written by whoever owns the room, so they are
//! untrusted input that ends up in a DOM text node, a terminal (`atlasctl show`)
//! and `--json` output. The contract's own `check_structure` rejects raw control
//! characters, which would make an unsanitised hostile name an entry nobody can
//! publish; stripping here means the entry is publishable AND harmless.
//!
//! What is removed, and why each class matters:
//! - C0/C1 controls (ANSI escape, CR/LF): terminal injection.
//! - Every Unicode format character (category Cf): bidi overrides and isolates,
//!   zero-width characters, soft hyphen, ALM, tag characters. These make a name
//!   render as something other than what it is, or let two rooms look identical.
//! - Variation selectors and the invisible fillers (CGJ, Hangul and Khmer
//!   fillers), which Cf does not cover.
//! - Line/paragraph separators (U+2028/2029): treated as whitespace.
//!
//! Whitespace of any kind collapses to one space and the ends are trimmed, so a
//! card is always a single line.

fn is_invisible(c: char) -> bool {
    use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};
    // Every format character (Cf) covers the bidi controls, zero-width
    // characters, ALM (U+061C), soft hyphen, interlinear annotations
    // (U+FFF9-FFFB), U+180E and the tag block, including ones added by later
    // Unicode versions. The ranges below are what Cf does not cover.
    c.general_category() == GeneralCategory::Format
        || matches!(c,
            // variation selectors, which change how a neighbour renders
            '\u{FE00}'..='\u{FE0F}' | '\u{E0100}'..='\u{E01EF}'
            | '\u{180B}'..='\u{180D}' | '\u{180F}'
            // combining grapheme joiner and the invisible Hangul/Khmer fillers
            | '\u{034F}' | '\u{115F}' | '\u{1160}' | '\u{17B4}' | '\u{17B5}'
            | '\u{3164}' | '\u{FFA0}')
}

/// Clean `s` to a single printable line of at most `max_bytes` UTF-8 bytes,
/// cut on a character boundary. `None` when nothing printable is left.
pub fn sanitize_line(s: &str, max_bytes: usize) -> Option<String> {
    let mut out = String::with_capacity(s.len().min(max_bytes));
    let mut pending_space = false;
    for c in s.chars() {
        if is_invisible(c) {
            continue;
        }
        if c.is_whitespace() || c == '\u{2028}' || c == '\u{2029}' {
            pending_space = !out.is_empty();
            continue;
        }
        if c.is_control() {
            continue;
        }
        let need = c.len_utf8() + usize::from(pending_space);
        if out.len() + need > max_bytes {
            break;
        }
        if pending_space {
            out.push(' ');
            pending_space = false;
        }
        out.push(c);
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_passes_unchanged() {
        assert_eq!(
            sanitize_line("Rust Chat", 200).as_deref(),
            Some("Rust Chat")
        );
        assert_eq!(
            sanitize_line("Café 日本", 200).as_deref(),
            Some("Café 日本")
        );
    }

    #[test]
    fn terminal_escapes_and_newlines_are_neutralised() {
        // ESC is dropped and the rest of the sequence is then just text, which is
        // harmless without the escape; CR/LF collapse to a space.
        let got = sanitize_line("a\x1b[31mred\r\nline", 200).unwrap();
        assert!(!got.contains('\x1b') && !got.contains('\r') && !got.contains('\n'));
        assert_eq!(got, "a[31mred line");
    }

    #[test]
    fn bidi_and_zero_width_are_removed() {
        let got = sanitize_line("ad\u{202E}min\u{200B}\u{FEFF}!", 200).unwrap();
        assert_eq!(got, "admin!");
    }

    #[test]
    fn category_format_characters_and_fillers_are_removed() {
        // ALM, CGJ, Hangul fillers, Mongolian vowel separator, variation
        // selectors, interlinear annotation anchors.
        let got = sanitize_line(
            "a\u{061C}b\u{034F}c\u{115F}d\u{1160}e\u{3164}f\u{180E}g\u{FE0F}h\u{E0100}i\u{FFF9}j",
            200,
        )
        .unwrap();
        assert_eq!(got, "abcdefghij");
    }

    #[test]
    fn tag_characters_are_removed() {
        assert_eq!(
            sanitize_line("hi\u{E0041}\u{E0042}", 200).as_deref(),
            Some("hi")
        );
    }

    #[test]
    fn whitespace_collapses_and_trims() {
        assert_eq!(
            sanitize_line("  a \t\n  b\u{2028}c  ", 200).as_deref(),
            Some("a b c")
        );
    }

    #[test]
    fn nothing_printable_is_none() {
        assert_eq!(sanitize_line("", 10), None);
        assert_eq!(sanitize_line(" \u{200B}\x07\n", 10), None);
    }

    #[test]
    fn truncates_on_a_char_boundary_within_the_byte_budget() {
        // Each of these is 3 bytes; a budget of 7 fits two, never splits a third.
        let got = sanitize_line("日本語", 7).unwrap();
        assert_eq!(got, "日本");
        assert!(got.len() <= 7);
        // A trailing separator is never emitted.
        let got = sanitize_line("ab cd", 3).unwrap();
        assert_eq!(got, "ab");
    }
}
