use unicode_xid::UnicodeXID;

use crate::Span;

pub struct Tokenizer<'s> {
    /// The full source string
    source: &'s str,
    /// The remainder of the input which has not been tokenized yet
    remainder: &'s str,
}

impl<'s> Tokenizer<'s> {
    pub fn new(source: &'s str) -> Self {
        Tokenizer {
            source,
            remainder: source,
        }
    }

    pub fn next_token(&mut self) -> Option<Token<'s>> {
        // skip whitespace
        if self.next_if(char::is_whitespace).is_some() {
            self.advance_while(char::is_whitespace);
        }
        let start = self.next_index();
        let ch = self.next()?;
        let kind = match ch.into() {
            CharKind::Digit => self.lex_number(),
            CharKind::Singleton => TokenKind::Tag,
            CharKind::DoubleQuote => {
                self.lex_string();
                TokenKind::String
            }
            kind => {
                self.advance_while(|ch| kind.can_be_followed_by(ch));
                TokenKind::Tag
            }
        };
        let end = self.next_index();
        Some(Token {
            span: Span { start, end },
            source: self.source,
            kind,
        })
    }

    fn lex_number(&mut self) -> TokenKind {
        let mut kind = TokenKind::Integer;
        self.advance_while(is_number_char);
        if self.next_if(|ch| ch == '.').is_some() && self.next_if(is_number_start_char).is_some() {
            kind = TokenKind::Float;
            self.advance_while(is_number_char);
        }
        let mut it = self.remainder.chars();
        if let Some('e' | 'E') = it.next() {
            // try to parse an exponent -- optional `+` or `-` followed by one or more digits
            match (it.next(), it.next()) {
                (Some('+' | '-'), Some(c)) if is_number_char(c) => {
                    kind = TokenKind::Float;
                    // e/E
                    self.next();
                    // +/-
                    self.next();
                    self.advance_while(is_number_char);
                }
                (Some(c), _) if is_number_char(c) => {
                    kind = TokenKind::Float;
                    // e/E
                    self.next();
                    self.advance_while(is_number_char);
                }
                _ => {}
            }
        }
        kind
    }

    fn lex_string(&mut self) {
        for (idx, _) in self.remainder.match_indices('"') {
            // includes the string up to the `"` we just found
            let string_match = &self.remainder[..idx];
            // count backslashes
            let num_backslash = string_match
                .chars()
                .rev()
                .take_while(|ch| *ch == '\\')
                .count();
            if num_backslash % 2 == 0 {
                // even number of backslashes => the `"` is unescaped
                self.remainder = &self.remainder[idx..];
                self.next();
                break;
            }
        }
        // never found the closing `"`, we must have hit the end of the input
        self.remainder = "";
    }

    fn peek(&self) -> Option<char> {
        self.remainder.chars().next()
    }

    fn next(&mut self) -> Option<char> {
        let mut it = self.remainder.chars();
        let val = it.next();
        self.remainder = it.as_str();
        val
    }

    /// Advances by one character if that character matches the predicate
    fn next_if(&mut self, predicate: impl FnOnce(char) -> bool) -> Option<char> {
        let mut it = self.remainder.chars();
        let ch = it.next()?;
        if predicate(ch) {
            self.remainder = it.as_str();
            Some(ch)
        } else {
            None
        }
    }

    fn advance_while(&mut self, mut predicate: impl FnMut(char) -> bool) {
        let offset = self
            .remainder
            .char_indices()
            .skip_while(|(_, c)| predicate(*c))
            .map(|(idx, _)| idx)
            .next()
            .unwrap_or(self.remainder.len());
        self.remainder = &self.remainder[offset..];
    }

    /// Returns the byte position of the next character, or the length of the underlying string if
    /// there are no more characters.
    fn next_index(&self) -> usize {
        self.source.len() - self.remainder.len()
    }
}

pub struct Token<'s> {
    span: Span,
    source: &'s str,
    kind: TokenKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenKind {
    Tag,
    Integer,
    Float,
    String,
}

/// Characters allowed at the start of a token
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CharKind {
    /// ASCII digits 0-9
    Digit,
    /// Any other character with the Unicode `XID_Start` property
    Identifier,
    /// `"`
    DoubleQuote,
    /// A character which forms a token on its own
    Singleton,
    /// `<`, `=`, `>`
    Comparison,
    /// Whitespace isn't part of any token
    Whitespace,
    /// Any character not covered by the above categories
    Other,
}

impl CharKind {
    /// Whether the specified character can follow in a token with this start kind.
    fn can_be_followed_by(self, ch: char) -> bool {
        match self {
            CharKind::Digit => is_number_char(ch),
            CharKind::Identifier => is_ident_char(ch),
            CharKind::DoubleQuote => true,
            CharKind::Singleton => false,
            CharKind::Comparison => is_comparison_char(ch),
            CharKind::Whitespace => ch.is_whitespace(),
            CharKind::Other => matches!(ch.into(), CharKind::Comparison | CharKind::Other),
        }
    }
}

impl From<char> for CharKind {
    fn from(ch: char) -> CharKind {
        match ch {
            '"' => CharKind::DoubleQuote,
            ch if is_singleton_char(ch) => CharKind::Singleton,
            ch if is_comparison_char(ch) => CharKind::Comparison,
            ch if is_number_start_char(ch) => CharKind::Digit,
            ch if is_ident_start_char(ch) => CharKind::Identifier,
            ch if ch.is_whitespace() => CharKind::Whitespace,
            _ => CharKind::Other,
        }
    }
}

#[inline]
fn is_number_start_char(ch: char) -> bool {
    ch.is_ascii_digit()
}

#[inline]
fn is_number_char(ch: char) -> bool {
    ch == '_' || ch.is_ascii_digit()
}

#[inline]
fn is_ident_start_char(ch: char) -> bool {
    ch == '_' || UnicodeXID::is_xid_start(ch)
}

#[inline]
fn is_ident_char(ch: char) -> bool {
    UnicodeXID::is_xid_continue(ch)
}

#[inline]
fn is_comparison_char(ch: char) -> bool {
    ch == '<' || ch == '=' || ch == '>'
}

#[inline]
fn is_singleton_char(ch: char) -> bool {
    ch == '(' || ch == ')'
}

#[cfg(test)]
mod tests {
    use test_case::test_case;

    use super::{Span, TokenKind, Tokenizer};

    #[test_case("abc", TokenKind::Tag, 0, 3 ; "tag abc")]
    #[test_case("a\u{0300}bc", TokenKind::Tag, 0, 5 ; "tag with combining char")]
    #[test_case("_0", TokenKind::Tag, 0, 2 ; "tag _0")]
    #[test_case("abc+", TokenKind::Tag, 0, 3 ; "tag followed by other char")]
    #[test_case("   \n\t\rabc", TokenKind::Tag, 6, 9 ; "leading whitespace")]
    #[test_case("<>", TokenKind::Tag, 0, 2 ; "comparison tag")]
    #[test_case("=-", TokenKind::Tag, 0, 1 ; "comparison followed by other char")]
    #[test_case("-=", TokenKind::Tag, 0, 2 ; "other char followed by comparison")]
    #[test_case("123", TokenKind::Integer, 0, 3 ; "integer")]
    #[test_case("1_234", TokenKind::Integer, 0, 5 ; "integer with underscores")]
    #[test_case("1.234", TokenKind::Float, 0, 5 ; "simple float")]
    #[test_case("1e1", TokenKind::Float, 0, 3 ; "float exponential")]
    #[test_case("1e1.", TokenKind::Float, 0, 3 ; "float exponential followed by dot")]
    #[test_case("1e", TokenKind::Integer, 0, 1 ; "float incomplete exponential")]
    #[test_case("1e-", TokenKind::Integer, 0, 1 ; "float incomplete exponential with sign")]
    #[test_case("1.3e-10", TokenKind::Float, 0, 7 ; "float with")]
    #[test_case(r#""abc\"\\\"\\""#, TokenKind::String, 0, 13 ; "string")]
    #[test_case(r#""abc\"\\\"abc"#, TokenKind::String, 0, 13 ; "string unterminated")]
    #[test_case("(((", TokenKind::Tag, 0, 1 ; "singleton")]
    fn lex_one(source: &str, kind: TokenKind, start: usize, end: usize) {
        let actual = Tokenizer::new(source).next_token().unwrap();
        assert_eq!(actual.kind, kind);
        let expected_span = Span { start, end };
        let actual_span = actual.span;
        assert_eq!(
            actual.span,
            expected_span,
            "\nexpected: {}\n{}\nactual: {}\n{}\n",
            expected_span,
            &source[start..end],
            actual_span,
            &source[actual_span.start..actual_span.end],
        );
    }
}
