use std::fmt;

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

    pub fn source(&self) -> &'s str {
        self.source
    }

    /// Returns whether the remainder of the input is empty (i.e. the tokenizer has run to
    /// completion)
    pub fn is_empty(&self) -> bool {
        self.remainder.is_empty()
    }

    pub fn next_token(&mut self) -> Option<Token<'s>> {
        // skip whitespace
        if self.next_if(char::is_whitespace).is_some() {
            self.advance_while(char::is_whitespace);
        }
        let start = self.next_index();
        let ch = self.next()?;
        let kind = match ch.into() {
            CharKind::Digit => self.lex_number(false),
            CharKind::Singleton => TokenKind::Tag,
            CharKind::DoubleQuote => self.lex_string(),
            CharKind::Dot if self.next_if(is_number_start_char).is_some() => self.lex_number(true),
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

    fn lex_number(&mut self, has_dot: bool) -> TokenKind {
        let mut kind = if has_dot {
            TokenKind::Float
        } else {
            TokenKind::Integer
        };
        self.advance_while(is_number_char);
        if !has_dot
            && self.next_if(|ch| ch == '.').is_some()
            && self.next_if(is_number_start_char).is_some()
        {
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

    fn lex_string(&mut self) -> TokenKind {
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
                return TokenKind::String;
            }
        }
        // never found the closing `"`, we must have hit the end of the input
        self.remainder = "";
        TokenKind::UnterminatedString
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Token<'s> {
    span: Span,
    source: &'s str,
    kind: TokenKind,
}

impl<'s> Token<'s> {
    pub fn span(&self) -> Span {
        self.span
    }

    pub fn source(&self) -> &'s str {
        self.source
    }

    pub fn as_str(&self) -> &'s str {
        &self.source[self.span.into_range()]
    }

    pub fn kind(&self) -> TokenKind {
        self.kind
    }
}

impl<'s> fmt::Display for Token<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenKind {
    Tag,
    Integer,
    Float,
    String,
    UnterminatedString,
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
    /// `.`
    Dot,
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
            CharKind::Dot | CharKind::Other => is_other_continuation_char(ch),
        }
    }
}

impl From<char> for CharKind {
    fn from(ch: char) -> CharKind {
        match ch {
            '"' => CharKind::DoubleQuote,
            '.' => CharKind::Dot,
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

#[inline]
fn is_other_continuation_char(ch: char) -> bool {
    matches!(
        ch.into(),
        CharKind::Comparison | CharKind::Dot | CharKind::Other
    )
}

#[cfg(test)]
mod tests {
    use test_case::test_case;

    use super::{TokenKind, Tokenizer};

    #[test_case("abc", TokenKind::Tag, "abc" ; "tag abc")]
    #[test_case("a\u{0300}bc", TokenKind::Tag, "a\u{0300}bc" ; "tag with combining char")]
    #[test_case("_0", TokenKind::Tag, "_0" ; "tag _0")]
    #[test_case("abc+", TokenKind::Tag, "abc" ; "tag followed by other char")]
    #[test_case("   \n\t\rabc", TokenKind::Tag, "abc" ; "leading whitespace")]
    #[test_case("<>", TokenKind::Tag, "<>" ; "comparison tag")]
    #[test_case("=-", TokenKind::Tag, "=" ; "comparison followed by other char")]
    #[test_case("-=", TokenKind::Tag, "-=" ; "other char followed by comparison")]
    #[test_case("..", TokenKind::Tag, ".." ; "tag starting with dot")]
    #[test_case("..123", TokenKind::Tag, ".." ; "tag starting with dot followed by number")]
    #[test_case("123", TokenKind::Integer, "123" ; "integer")]
    #[test_case("1_234", TokenKind::Integer, "1_234" ; "integer with underscores")]
    #[test_case("1.234", TokenKind::Float, "1.234" ; "simple float")]
    #[test_case(".234", TokenKind::Float, ".234" ; "float with no integer")]
    #[test_case("1.234.5", TokenKind::Float, "1.234" ; "float with extra dot")]
    #[test_case(".234.5", TokenKind::Float, ".234" ; "float with no integer and extra dot")]
    #[test_case("1e1", TokenKind::Float, "1e1" ; "float exponential")]
    #[test_case("1e1.", TokenKind::Float, "1e1" ; "float exponential followed by dot")]
    #[test_case("1e", TokenKind::Integer, "1" ; "float incomplete exponential")]
    #[test_case("1e-", TokenKind::Integer, "1" ; "float incomplete exponential with sign")]
    #[test_case("1.3e-10", TokenKind::Float, "1.3e-10" ; "float with all")]
    #[test_case(r#""abc\"\\\"\\""#, TokenKind::String, r#""abc\"\\\"\\""# ; "string")]
    #[test_case(r#""abc\"\\\"abc"#, TokenKind::UnterminatedString, r#""abc\"\\\"abc"# ; "string unterminated")]
    #[test_case("(((", TokenKind::Tag, "(" ; "singleton")]
    fn lex_one(source: &str, kind: TokenKind, as_str: &str) {
        let actual = Tokenizer::new(source).next_token().unwrap();
        assert_eq!(actual.kind, kind);
        assert_eq!(actual.as_str(), as_str);
    }
}
