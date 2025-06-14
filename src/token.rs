use std::{
    convert::Infallible,
    fmt,
    io::{self, BufRead},
    marker::PhantomData,
};

use bstr::ByteSlice;
use bytes::{Bytes, BytesMut};
use itertools::Either;
use unicode_xid::UnicodeXID;

use crate::Span;

pub trait Tokenizer {
    type Token;
    type Position: Clone + Default;
    type Error;

    fn next_token(&mut self) -> Option<Result<TokenFor<Self>, Self::Error>>;
}

impl<T: Tokenizer> Tokenizer for &mut T {
    type Token = T::Token;
    type Position = T::Position;
    type Error = T::Error;

    fn next_token(&mut self) -> Option<Result<TokenFor<Self>, Self::Error>> {
        T::next_token(self)
    }
}

pub struct IterTokenizer<I>(pub I);
impl<I, T, Idx, E> Tokenizer for IterTokenizer<I>
where
    I: Iterator<Item = Result<Token<T, Idx>, E>>,
    Idx: Clone + Default,
{
    type Token = T;
    type Position = Idx;
    type Error = E;

    fn next_token(&mut self) -> Option<Result<Token<Self::Token, Self::Position>, Self::Error>> {
        self.0.next()
    }
}

/// A tokenizer which tokenizes characters by grouping them into sets.
pub struct CharSetTokenizer<S, C> {
    pub source: S,
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

impl<T, E> CharSetResult<T, E> {
    pub fn map_into<U: From<T>, F: From<E>>(self) -> CharSetResult<U, F> {
        match self {
            CharSetResult::Continue => CharSetResult::Continue,
            CharSetResult::Done(Some(val)) => CharSetResult::Done(Some(val.into())),
            CharSetResult::Done(None) => CharSetResult::Done(None),
            CharSetResult::Err(err) => CharSetResult::Err(err.into()),
        }
    }
}

pub trait CharSet<C>: Default {
    type TokenKind;
    type Error;

    /// Categorize a charcter while continuing a potential token.
    fn next_char(&mut self, c: C) -> CharSetResult<Self::TokenKind, Self::Error>;

    /// What token kind (if any) to return if end of input is reached.
    fn end_of_input(self) -> Result<Option<Self::TokenKind>, Self::Error>;
}

type CharSetToken<S, C> = (
    <S as Source>::String,
    <C as CharSet<<S as Source>::Char>>::TokenKind,
);
type CharSetError<S, C> = Either<<S as Source>::Error, <C as CharSet<<S as Source>::Char>>::Error>;

impl<S: Source, C: CharSet<S::Char>> CharSetTokenizer<S, C> {
    pub fn new(source: S) -> Self {
        Self {
            source,
            _marker: PhantomData,
        }
    }

    /// Advances in the input as long as the character matches the character set.
    fn advance_while(&mut self) -> Result<Option<CharSetToken<S, C>>, CharSetError<S, C>> {
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
    type Token = CharSetToken<S, C>;
    type Position = usize;
    type Error = CharSetError<S, C>;

    fn next_token(&mut self) -> Option<Result<Token<Self::Token, Self::Position>, Self::Error>> {
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
    type Item = Result<Token<<Self as Tokenizer>::Token, usize>, <Self as Tokenizer>::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
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
    line_buffer: Vec<u8>,
    buffer: BytesMut,
    index: usize,
    is_empty: bool,
}

impl<R: BufRead> BufReadSource<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line_buffer: Vec::new(),
            buffer: BytesMut::new(),
            index: 0,
            is_empty: false,
        }
    }

    fn flil_buf(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            match self.reader.read_until(b'\n', &mut self.line_buffer) {
                Ok(0) => self.is_empty = true,
                Ok(_) => {}
                Err(error) => {
                    self.is_empty = true;
                    return Err(error);
                }
            }
            self.buffer.extend_from_slice(&self.line_buffer);
            self.line_buffer.clear();
        }
        Ok(())
    }
}

impl<R: BufRead> Source for BufReadSource<R> {
    type Char = char;
    type String = Bytes;
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
    ) -> Result<Self::String, io::Error> {
        let mut token = self.buffer.split_to(0);
        while !self.is_empty {
            self.flil_buf()?;
            let buffer = &self.buffer;
            let offset = buffer
                .char_indices()
                .skip_while(|(_, _, c)| predicate(*c))
                .map(|(idx, _, _)| idx)
                .next()
                .unwrap_or(buffer.len());
            self.index += offset;
            let s = self.buffer.split_to(offset);

            if let Err(e) = std::str::from_utf8(&s) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, e));
            }
            token.unsplit(s);
            if !self.buffer.is_empty() {
                break;
            }
        }
        Ok(token.freeze())
    }
}

pub type SimpleTokenizer<S> = CharSetTokenizer<S, SimpleCharSet>;

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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NumberState {
    /// The first digit of the integer part has been matched
    #[default]
    Integer,
    /// The dot separating the integer and fractional parts has been matched
    Dot,
    /// The first digit of the fractional part has been matched
    Fractional,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumberKind {
    Integer,
    Float,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SimpleCharSetError {
    #[error("unterminated string")]
    UnterminatedString,
}

impl From<Infallible> for SimpleCharSetError {
    fn from(err: Infallible) -> Self {
        match err {}
    }
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
                res.map_into()
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
            Self::Number(state) => state
                .end_of_input()
                .map(|opt| opt.map(Into::into))
                .map_err(Into::into),
            Self::Identifier | Self::Singleton | Self::Comparison | Self::Dot | Self::Other => {
                Ok(Some(SimpleCharSetTokenKind::Tag))
            }
            Self::String(_) => Err(SimpleCharSetError::UnterminatedString),
            Self::BreakNext(kind) => Ok(kind),
            Self::Whitespace => Ok(None),
        }
    }
}

impl CharSet<char> for NumberState {
    type TokenKind = NumberKind;
    type Error = Infallible;

    fn next_char(&mut self, ch: char) -> CharSetResult<Self::TokenKind, Self::Error> {
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
            (Self::Integer, _) => CharSetResult::Done(Some(NumberKind::Integer)),
            (Self::Dot | Self::Fractional, _) => CharSetResult::Done(Some(NumberKind::Float)),
        }
    }

    fn end_of_input(self) -> Result<Option<Self::TokenKind>, Self::Error> {
        match self {
            Self::Integer => Ok(Some(NumberKind::Integer)),
            Self::Dot | Self::Fractional => Ok(Some(NumberKind::Float)),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Token<T, Idx> {
    pub span: Span<Idx>,
    pub kind: T,
}

pub type TokenFor<T> = Token<<T as Tokenizer>::Token, <T as Tokenizer>::Position>;

impl<T: fmt::Display, Idx: fmt::Display> fmt::Display for Token<T, Idx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.kind, f)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimpleCharSetTokenKind {
    Tag,
    Number(NumberKind),
    String,
}

impl From<NumberKind> for SimpleCharSetTokenKind {
    fn from(number: NumberKind) -> Self {
        Self::Number(number)
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
        SimpleCharSet::categorize(ch),
        SimpleCharSet::Comparison | SimpleCharSet::Dot | SimpleCharSet::Other
    )
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use test_case::test_case;

    use super::{BufReadSource, NumberKind, SimpleCharSetTokenKind, SimpleTokenizer, StrSource};

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
    #[test_case("123", SimpleCharSetTokenKind::Number(NumberKind::Integer), "123" ; "integer")]
    #[test_case("1_234", SimpleCharSetTokenKind::Number(NumberKind::Integer), "1_234" ; "integer with underscoreNumberKind")]
    #[test_case("1.234", SimpleCharSetTokenKind::Number(NumberKind::Float), "1.234" ; "simple float")]
    #[test_case(".234", SimpleCharSetTokenKind::Number(NumberKind::Float), ".234" ; "float with no integer")]
    #[test_case("1.", SimpleCharSetTokenKind::Number(NumberKind::Float), "1." ; "integer followed by dot")]
    #[test_case("1.234.5", SimpleCharSetTokenKind::Number(NumberKind::Float), "1.234" ; "float with extra dot")]
    #[test_case(".234.5", SimpleCharSetTokenKind::Number(NumberKind::Float), ".234" ; "float with no integer and extra dot")]
    #[test_case(r#""abc\"\\\"\\""#, SimpleCharSetTokenKind::String, r#""abc\"\\\"\\""# ; "string")]
    #[test_case("(((", SimpleCharSetTokenKind::Tag, "(" ; "singleton")]
    fn lex_one(source: &str, kind: SimpleCharSetTokenKind, as_str: &str) {
        let actual = SimpleTokenizer::new(StrSource::new(source))
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(actual.kind, (as_str, kind));
    }

    #[test]
    fn unterminated_string() {
        let source = r#""abc\"\\\"abc"#;
        SimpleTokenizer::new(StrSource::new(source))
            .next()
            .unwrap()
            .unwrap_err();
    }

    #[test_case("abc<>+1", "abc <> + 1" ; "some tokens")]
    #[test_case("<->", "< ->" ; "arrows")]
    fn tokenize_many(source: &str, tokens: &str) {
        let actual_str = SimpleTokenizer::new(StrSource::new(source))
            .map(|res| res.map(|tok| tok.kind.0))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let expected = tokens.split_whitespace().collect::<Vec<_>>();
        assert_eq!(actual_str, expected);
        let actual_br = SimpleTokenizer::new(BufReadSource::new(source.as_bytes()))
            .map(|res| res.map(|tok| tok.kind.0))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(actual_br, expected);
    }

    #[test]
    fn invalid_utf8() {
        let source: &[u8] = b"abc\x80\x81def";
        let tokens = SimpleTokenizer::new(BufReadSource::new(source))
            .map(|res| res.map(|tok| tok.kind.0).map_err(|_| ()))
            .collect::<Vec<Result<_, _>>>();
        assert_eq!(
            tokens,
            &[
                Ok(Bytes::from_static("abc".as_bytes())),
                Err(()),
                Ok(Bytes::from_static("def".as_bytes()))
            ]
        );
    }
}
