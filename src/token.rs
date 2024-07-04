use std::{
    convert::Infallible,
    fmt,
    io::{self, BufRead},
    marker::PhantomData,
};

use itertools::Either;
use unicode_xid::UnicodeXID;

use crate::Span;

pub trait Tokenizer {
    type Token;
    type Error;

    fn next_token(&mut self) -> Option<Result<Token<Self::Token>, Self::Error>>;
}

impl<T: Tokenizer> Tokenizer for &mut T {
    type Token = T::Token;
    type Error = T::Error;

    fn next_token(&mut self) -> Option<Result<Token<Self::Token>, Self::Error>> {
        T::next_token(self)
    }
}

pub trait Source {
    type Char;
    type String;
    type Error;

    /// Returns the start index of the next character.
    fn next_index(&self) -> usize;

    fn is_empty(&self) -> bool;

    /// Advances in the input as long as the character matches the predicate, returning the
    /// matching string.
    fn advance_while(
        &mut self,
        predicate: impl FnMut(Self::Char) -> bool,
    ) -> Result<Self::String, Self::Error>;
}

pub struct StrSource<'s> {
    /// The full source string
    source: &'s str,
    /// The remainder of the input which has not been tokenized yet
    remainder: &'s str,
}

impl<'s> StrSource<'s> {
    pub fn new(source: &'s str) -> Self {
        Self {
            source,
            remainder: source,
        }
    }
}

impl<'s> Source for StrSource<'s> {
    type Char = char;
    type String = &'s str;
    type Error = Infallible;

    fn next_index(&self) -> usize {
        self.source.len() - self.remainder.len()
    }

    fn is_empty(&self) -> bool {
        self.remainder.is_empty()
    }

    fn advance_while(
        &mut self,
        mut predicate: impl FnMut(char) -> bool,
    ) -> Result<&'s str, Infallible> {
        let start = self.next_index();
        let offset = self
            .remainder
            .char_indices()
            .skip_while(|(_, c)| predicate(*c))
            .map(|(idx, _)| idx)
            .next()
            .unwrap_or(self.remainder.len());
        self.remainder = &self.remainder[offset..];
        Ok(&self.source[start..start + offset])
    }
}

pub struct BufReadSource<R> {
    reader: R,
    buffer: String,
    index: usize,
    is_empty: bool,
}

impl<R: BufRead> BufReadSource<R> {
    fn flil_buf(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            self.is_empty = self.reader.read_line(&mut self.buffer)? == 0;
        }
        Ok(())
    }
}

impl<R: BufRead> Source for BufReadSource<R> {
    type Char = char;
    type String = String;
    type Error = io::Error;

    fn next_index(&self) -> usize {
        self.index
    }

    fn is_empty(&self) -> bool {
        self.is_empty
    }

    fn advance_while(
        &mut self,
        mut predicate: impl FnMut(char) -> bool,
    ) -> Result<String, io::Error> {
        let mut token = String::new();
        while !self.is_empty {
            self.flil_buf()?;
            let buffer = &self.buffer;
            let offset = buffer
                .char_indices()
                .skip_while(|(_, c)| predicate(*c))
                .map(|(idx, _)| idx)
                .next()
                .unwrap_or(buffer.len());
            self.index += offset;
            let mut s = self.buffer.split_off(offset);
            std::mem::swap(&mut s, &mut self.buffer);
            if token.is_empty() {
                token = s;
            } else {
                token.push_str(&s);
            }
            if !self.buffer.is_empty() {
                break;
            }
        }
        Ok(token)
    }
}

/// A tokenizer which tokenizes characters by grouping them into sets.
pub struct CharSetTokenizer<S, C> {
    source: S,
    _marker: PhantomData<fn() -> C>,
}

pub enum CharSetResult<T, E> {
    /// Accept the character in the token
    Continue,
    /// Reject the character and optionally produce a token
    Done(Option<T>),
    /// Reject the character and discard the current token, returning an error
    Err(E),
}

pub trait CharSet<C>: Default {
    type TokenKind;
    type Error;

    /// Categorize a charcter while continuing a potential token.
    fn next_char(&mut self, c: C) -> CharSetResult<Self::TokenKind, Self::Error>;

    /// What token kind (if any) to return if end of input is reached.
    fn end_of_input(self) -> Result<Option<Self::TokenKind>, Self::Error>;
}

impl<S: Source, C: CharSet<S::Char>> CharSetTokenizer<S, C> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            _marker: PhantomData,
        }
    }

    /// Advances in the input as long as the character matches the character set.
    fn advance_while(
        &mut self,
    ) -> Result<Option<(S::String, C::TokenKind)>, Either<S::Error, C::Error>> {
        let mut result = Ok(None);
        let mut state = C::default();
        let predicate = |c| match state.next_char(c) {
            CharSetResult::Continue => true,
            CharSetResult::Done(new_kind) => {
                result = Ok(new_kind);
                false
            }
            CharSetResult::Err(err) => {
                result = Err(err);
                false
            }
        };
        let token = self.source.advance_while(predicate).map_err(Either::Left)?;
        let kind = result.map_err(Either::Right)?;
        let kind = if self.source.is_empty() {
            state.end_of_input().map_err(Either::Right)?
        } else {
            kind
        };
        Ok(kind.map(|kind| (token, kind)))
    }
}

impl<S: Source, C: CharSet<S::Char>> Tokenizer for CharSetTokenizer<S, C> {
    type Token = (S::String, C::TokenKind);
    type Error = Either<S::Error, C::Error>;

    fn next_token(&mut self) -> Option<Result<Token<Self::Token>, Self::Error>> {
        while !self.source.is_empty() {
            let start = self.source.next_index();
            match self.advance_while() {
                Ok(Some(token)) => {
                    let end = self.source.next_index();
                    return Some(Ok(Token {
                        span: Span { start, end },
                        kind: token,
                    }));
                }
                Ok(None) => continue,
                Err(err) => return Some(Err(err)),
            }
        }
        None
    }
}

impl<S: Source, C: CharSet<S::Char>> Iterator for CharSetTokenizer<S, C> {
    type Item = Result<Token<<Self as Tokenizer>::Token>, <Self as Tokenizer>::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

pub type SimpleTokenizer<'s> = CharSetTokenizer<StrSource<'s>, SimpleCharSet>;

#[derive(Clone, Copy, Default, Debug, Eq, PartialEq)]
pub enum SimpleCharSet {
    #[default]
    None,
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
    BreakNext(Option<SimpleCharSetTokenKind>),
}

impl SimpleCharSet {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SimpleCharSetError {
    #[error("unterminated string")]
    UnterminatedString,
}

impl CharSet<char> for SimpleCharSet {
    type TokenKind = SimpleCharSetTokenKind;
    type Error = SimpleCharSetError;

    fn next_char(&mut self, ch: char) -> CharSetResult<Self::TokenKind, Self::Error> {
        match (*self, ch) {
            (Self::None, ch) => {
                *self = Self::categorize(ch);
                CharSetResult::Continue
            }
            (Self::Number(mut state), ch) => {
                let res = state.next_char(ch);
                *self = Self::Number(state);
                res
            }
            (Self::Identifier, ch) if is_ident_char(ch) => CharSetResult::Continue,
            (Self::String(false), '"') => {
                *self = Self::BreakNext(Some(SimpleCharSetTokenKind::String));
                CharSetResult::Continue
            }
            (Self::String(escaped), '\\') => {
                *self = Self::String(!escaped);
                CharSetResult::Continue
            }
            (Self::String(_), _) => {
                *self = Self::String(false);
                CharSetResult::Continue
            }
            (Self::Comparison, ch) if is_comparison_char(ch) => CharSetResult::Continue,
            (Self::Dot, ch) if is_number_start_char(ch) => {
                *self = Self::Number(NumberState::Dot);
                CharSetResult::Continue
            }
            (Self::Dot, ch) if is_other_continuation_char(ch) => {
                *self = Self::Other;
                CharSetResult::Continue
            }
            (Self::Other, ch) if is_other_continuation_char(ch) => CharSetResult::Continue,
            (
                Self::Identifier | Self::Singleton | Self::Comparison | Self::Other | Self::Dot,
                _,
            ) => CharSetResult::Done(Some(SimpleCharSetTokenKind::Tag)),
            (Self::Whitespace, ch) if ch.is_whitespace() => CharSetResult::Continue,
            (Self::Whitespace, _) => CharSetResult::Done(None),
            (Self::BreakNext(kind), _) => CharSetResult::Done(kind),
        }
    }

    fn end_of_input(self) -> Result<Option<Self::TokenKind>, Self::Error> {
        match self {
            Self::None => Ok(None),
            Self::Number(state) => state.end_of_input(),
            Self::Identifier | Self::Singleton | Self::Comparison | Self::Dot | Self::Other => {
                Ok(Some(SimpleCharSetTokenKind::Tag))
            }
            Self::String(_) => Err(SimpleCharSetError::UnterminatedString),
            Self::BreakNext(kind) => Ok(kind),
            Self::Whitespace => Ok(None),
        }
    }
}

impl NumberState {
    fn next_char(&mut self, ch: char) -> CharSetResult<SimpleCharSetTokenKind, SimpleCharSetError> {
        match (*self, ch) {
            (Self::Integer, '.') => {
                *self = Self::Dot;
                CharSetResult::Continue
            }
            (Self::Dot, ch) if is_number_start_char(ch) => {
                *self = Self::Fractional;
                CharSetResult::Continue
            }
            (Self::Integer | Self::Fractional, ch) if is_number_char(ch) => CharSetResult::Continue,
            (Self::Integer, _) => CharSetResult::Done(Some(SimpleCharSetTokenKind::Integer)),
            (Self::Dot | Self::Fractional, _) => {
                CharSetResult::Done(Some(SimpleCharSetTokenKind::Float))
            }
        }
    }

    fn end_of_input(self) -> Result<Option<SimpleCharSetTokenKind>, SimpleCharSetError> {
        match self {
            Self::Integer => Ok(Some(SimpleCharSetTokenKind::Integer)),
            Self::Dot | Self::Fractional => Ok(Some(SimpleCharSetTokenKind::Float)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Token<T> {
    pub span: Span,
    pub kind: T,
}

impl<T: fmt::Display> fmt::Display for Token<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.kind, f)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimpleCharSetTokenKind {
    Tag,
    Integer,
    Float,
    String,
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

    use super::{SimpleCharSetTokenKind, SimpleTokenizer, StrSource};

    #[test_case("abc", SimpleCharSetTokenKind::Tag, "abc" ; "tag abc")]
    #[test_case("a\u{0300}bc", SimpleCharSetTokenKind::Tag, "a\u{0300}bc" ; "tag with combining char")]
    #[test_case("_0", SimpleCharSetTokenKind::Tag, "_0" ; "tag _0")]
    #[test_case("abc+", SimpleCharSetTokenKind::Tag, "abc" ; "tag followed by other char")]
    #[test_case("   \n\t\rabc", SimpleCharSetTokenKind::Tag, "abc" ; "leading whitespace")]
    #[test_case("<>", SimpleCharSetTokenKind::Tag, "<>" ; "comparison tag")]
    #[test_case("=-", SimpleCharSetTokenKind::Tag, "=" ; "comparison followed by other char")]
    #[test_case("-=", SimpleCharSetTokenKind::Tag, "-=" ; "other char followed by comparison")]
    #[test_case("..", SimpleCharSetTokenKind::Tag, ".." ; "tag starting with dot")]
    #[test_case("..123", SimpleCharSetTokenKind::Tag, ".." ; "tag starting with dot followed by number")]
    #[test_case("123", SimpleCharSetTokenKind::Integer, "123" ; "integer")]
    #[test_case("1_234", SimpleCharSetTokenKind::Integer, "1_234" ; "integer with underscores")]
    #[test_case("1.234", SimpleCharSetTokenKind::Float, "1.234" ; "simple float")]
    #[test_case(".234", SimpleCharSetTokenKind::Float, ".234" ; "float with no integer")]
    #[test_case("1.", SimpleCharSetTokenKind::Float, "1." ; "integer followed by dot")]
    #[test_case("1.234.5", SimpleCharSetTokenKind::Float, "1.234" ; "float with extra dot")]
    #[test_case(".234.5", SimpleCharSetTokenKind::Float, ".234" ; "float with no integer and extra dot")]
    #[test_case(r#""abc\"\\\"\\""#, SimpleCharSetTokenKind::String, r#""abc\"\\\"\\""# ; "string")]
    #[test_case("(((", SimpleCharSetTokenKind::Tag, "(" ; "singleton")]
    fn lex_one(source: &str, kind: SimpleCharSetTokenKind, as_str: &str) {
        let actual = SimpleTokenizer::new(StrSource::new(source)).next().unwrap().unwrap();
        assert_eq!(actual.kind, (as_str, kind));
    }

    #[test]
    fn unterminated_string() {
        let source = r#""abc\"\\\"abc"#;
        SimpleTokenizer::new(StrSource::new(source)).next().unwrap().unwrap_err();
    }
}
