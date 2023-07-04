use std::{fmt, marker::PhantomData, ops::ControlFlow};

use unicode_xid::UnicodeXID;

use crate::Span;

pub trait Tokenizer<'s> {
    /// Returns the full source string
    fn source(&self) -> &'s str;

    /// Returns whether the remainder of the input is empty (i.e. the tokenizer has run to
    /// completion)
    fn is_empty(&self) -> bool;

    /// Returns the next token in the input, or `None` if there is no more input.
    fn next_token(&mut self) -> Option<(Token<'s>, TokenKind)>;
}

/// A tokenizer which tokenizes characters by grouping them into sets.
pub struct CharSetTokenizer<'s, C> {
    /// The full source string
    source: &'s str,
    /// The remainder of the input which has not been tokenized yet
    remainder: &'s str,
    _marker: PhantomData<fn() -> C>,
}

pub trait CharSet {
    /// Categorize a character at the start of a potential token.
    fn categorize(c: char) -> Self;

    /// Categorize a charcter while continuing a potential token.
    ///
    /// # Return value
    /// - `Continue`: accepts the character as part of this potential token.
    /// - `Break(Some(_))`: rejects the character and produces a token with the specified
    ///   `TokenKind`.
    /// - `Break(None)`: rejects the character and does not produce a token.
    fn next_char(&mut self, c: char) -> ControlFlow<Option<TokenKind>>;

    /// What token kind (if any) to return if end of input is reached.
    fn end_of_input(self) -> Option<TokenKind>;
}

impl<'s, C: CharSet> CharSetTokenizer<'s, C> {
    pub fn new(source: &'s str) -> Self {
        Self {
            source,
            remainder: source,
            _marker: PhantomData,
        }
    }

    /// Advance to the next input character.
    fn next(&mut self) -> Option<char> {
        let mut it = self.remainder.chars();
        let val = it.next();
        self.remainder = it.as_str();
        val
    }

    /// Advances in the input as long as the character matches the character set.
    fn advance_while(&mut self, mut state: C) -> Option<TokenKind> {
        let kind = &mut None;
        let mut predicate = |c| match state.next_char(c) {
            ControlFlow::Continue(()) => true,
            ControlFlow::Break(new_kind) => {
                *kind = new_kind;
                false
            }
        };
        let offset = self
            .remainder
            .char_indices()
            .skip_while(|(_, c)| predicate(*c))
            .map(|(idx, _)| idx)
            .next()
            .unwrap_or(self.remainder.len());
        self.remainder = &self.remainder[offset..];
        if self.remainder.is_empty() {
            state.end_of_input()
        } else {
            *kind
        }
    }

    /// Returns the byte position of the next character, or the length of the underlying string if
    /// there are no more characters.
    fn next_index(&self) -> usize {
        self.source.len() - self.remainder.len()
    }
}

impl<'s, C: CharSet> Tokenizer<'s> for CharSetTokenizer<'s, C> {
    fn source(&self) -> &'s str {
        self.source
    }

    fn is_empty(&self) -> bool {
        self.remainder.is_empty()
    }

    fn next_token(&mut self) -> Option<(Token<'s>, TokenKind)> {
        loop {
            let start = self.next_index();
            let ch = self.next()?;
            let state = C::categorize(ch);
            if let Some(kind) = self.advance_while(state) {
                let end = self.next_index();
                return Some((
                    Token {
                        span: Span { start, end },
                        source: self.source,
                    },
                    kind,
                ));
            }
        }
    }
}

pub type SimpleTokenizer<'s> = CharSetTokenizer<'s, SimpleCharSet>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimpleCharSet {
    /// A number, which can be an integer, or a floating-point number
    Number(NumberState),
    /// An identifier consists of a character with the Unicode `XID_Start` property, followed by a
    /// sequence of characters with the Unicode `XID_continue` property
    Identifier,
    /// A string delimited by double quotes (`"`). The boolean indicates whether an odd number of
    /// backslashes have been matched (and thus the next `"` is escaped).
    String(bool),
    /// A character which forms a token on its own
    Singleton,
    /// Tokens starting with `<`, `=`, `>` can only contain other characters from that set.
    Comparison,
    /// `.` is part of a number if followed by a digit, or part of a punctuation tag otherwise.
    Dot,
    /// Whitespace isn't part of any token
    Whitespace,
    /// Any character not covered by the above categories
    Other,
    /// The next character will not be in this token
    BreakNext(Option<TokenKind>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumberState {
    /// The first digit of the integer part has been matched
    Integer,
    /// The dot separating the integer and fractional parts has been matched
    Dot,
    /// The first digit of the fractional part has been matched
    Fractional,
}

impl CharSet for SimpleCharSet {
    fn categorize(ch: char) -> Self {
        match ch {
            '"' => Self::String(false),
            '.' => Self::Dot,
            ch if is_singleton_char(ch) => Self::Singleton,
            ch if is_comparison_char(ch) => Self::Comparison,
            ch if is_number_start_char(ch) => Self::Number(NumberState::Integer),
            ch if is_ident_start_char(ch) => Self::Identifier,
            ch if ch.is_whitespace() => Self::Whitespace,
            _ => Self::Other,
        }
    }

    fn next_char(&mut self, ch: char) -> ControlFlow<Option<TokenKind>> {
        match (*self, ch) {
            (Self::Number(mut state), ch) => {
                let res = state.next_char(ch);
                *self = Self::Number(state);
                res
            }
            (Self::Identifier, ch) if is_ident_char(ch) => ControlFlow::Continue(()),
            (Self::String(false), '"') => {
                *self = Self::BreakNext(Some(TokenKind::String));
                ControlFlow::Continue(())
            }
            (Self::String(escaped), '\\') => {
                *self = Self::String(!escaped);
                ControlFlow::Continue(())
            }
            (Self::String(_), _) => {
                *self = Self::String(false);
                ControlFlow::Continue(())
            }
            (Self::Comparison, ch) if is_comparison_char(ch) => ControlFlow::Continue(()),
            (Self::Dot, ch) if is_number_start_char(ch) => {
                *self = Self::Number(NumberState::Dot);
                ControlFlow::Continue(())
            }
            (Self::Dot, ch) if is_other_continuation_char(ch) => {
                *self = Self::Other;
                ControlFlow::Continue(())
            }
            (Self::Other, ch) if is_other_continuation_char(ch) => ControlFlow::Continue(()),
            (
                Self::Identifier | Self::Singleton | Self::Comparison | Self::Other | Self::Dot,
                _,
            ) => ControlFlow::Break(Some(TokenKind::Tag)),
            (Self::Whitespace, ch) if ch.is_whitespace() => ControlFlow::Continue(()),
            (Self::Whitespace, _) => ControlFlow::Break(None),
            (Self::BreakNext(kind), _) => ControlFlow::Break(kind),
        }
    }

    fn end_of_input(self) -> Option<TokenKind> {
        match self {
            Self::Number(state) => state.end_of_input(),
            Self::Identifier | Self::Singleton | Self::Comparison | Self::Dot | Self::Other => {
                Some(TokenKind::Tag)
            }
            Self::String(_) => Some(TokenKind::UnterminatedString),
            Self::BreakNext(kind) => kind,
            Self::Whitespace => None,
        }
    }
}

impl NumberState {
    fn next_char(&mut self, ch: char) -> ControlFlow<Option<TokenKind>> {
        match (*self, ch) {
            (Self::Integer, '.') => {
                *self = Self::Dot;
                ControlFlow::Continue(())
            }
            (Self::Dot, ch) if is_number_start_char(ch) => {
                *self = Self::Fractional;
                ControlFlow::Continue(())
            }
            (Self::Integer | Self::Fractional, ch) if is_number_char(ch) => {
                ControlFlow::Continue(())
            }
            (Self::Integer, _) => ControlFlow::Break(Some(TokenKind::Integer)),
            (Self::Dot | Self::Fractional, _) => ControlFlow::Break(Some(TokenKind::Float)),
        }
    }

    fn end_of_input(self) -> Option<TokenKind> {
        match self {
            Self::Integer => Some(TokenKind::Integer),
            Self::Dot | Self::Fractional => Some(TokenKind::Float),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Token<'s> {
    span: Span,
    source: &'s str,
}

impl<'s> Token<'s> {
    pub fn new(span: Span, source: &'s str) -> Self {
        Token { span, source }
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn source(&self) -> &'s str {
        self.source
    }

    pub fn as_str(&self) -> &'s str {
        &self.source[self.span.into_range()]
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
        SimpleCharSet::categorize(ch),
        SimpleCharSet::Comparison | SimpleCharSet::Dot | SimpleCharSet::Other
    )
}

#[cfg(test)]
mod tests {
    use test_case::test_case;

    use super::{SimpleTokenizer, TokenKind, Tokenizer};

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
    #[test_case("1.", TokenKind::Float, "1." ; "integer followed by dot")]
    #[test_case("1.234.5", TokenKind::Float, "1.234" ; "float with extra dot")]
    #[test_case(".234.5", TokenKind::Float, ".234" ; "float with no integer and extra dot")]
    #[test_case(r#""abc\"\\\"\\""#, TokenKind::String, r#""abc\"\\\"\\""# ; "string")]
    #[test_case(r#""abc\"\\\"abc"#, TokenKind::UnterminatedString, r#""abc\"\\\"abc"# ; "string unterminated")]
    #[test_case("(((", TokenKind::Tag, "(" ; "singleton")]
    fn lex_one(source: &str, kind: TokenKind, as_str: &str) {
        let actual = SimpleTokenizer::new(source).next_token().unwrap();
        assert_eq!((actual.0.as_str(), actual.1), (as_str, kind));
    }
}
