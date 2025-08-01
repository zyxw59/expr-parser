use std::fmt;

use crate::Span;

#[cfg(feature = "charset-tokenizer")]
pub mod charset;

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
