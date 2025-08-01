use std::{error::Error, fmt};

use crate::{parser::Parser, token::Tokenizer, Span};

#[derive(Clone, Debug)]
pub struct ParseErrors<P, T, Idx> {
    pub errors: Vec<ParseError<P, T, Idx>>,
}

impl<P, T, Idx> Error for ParseErrors<P, T, Idx> where Self: fmt::Debug + fmt::Display {}

pub type ParseErrorsFor<P, T> = ParseErrors<
    <P as Parser<<T as Tokenizer>::Token>>::Error,
    <T as Tokenizer>::Error,
    <T as Tokenizer>::Position,
>;

impl<P: fmt::Display, T: fmt::Display, Idx: fmt::Display> fmt::Display for ParseErrors<P, T, Idx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.errors.len() == 1 {
            f.write_str("Encountered 1 error:\n")?;
        } else {
            writeln!(f, "Encountered {} errors:", self.errors.len())?;
        }
        for error in &self.errors {
            writeln!(f, "{error}")?;
        }
        Ok(())
    }
}

impl<P, T, Idx> From<Vec<ParseError<P, T, Idx>>> for ParseErrors<P, T, Idx> {
    fn from(errors: Vec<ParseError<P, T, Idx>>) -> Self {
        ParseErrors { errors }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParseError<P, T, Idx> {
    pub span: Span<Idx>,
    pub kind: ParseErrorKind<P, T, Idx>,
}

impl<P, T, Idx> Error for ParseError<P, T, Idx> where Self: fmt::Debug + fmt::Display {}

impl<P, T, Idx> fmt::Display for ParseError<P, T, Idx>
where
    Span<Idx>: fmt::Display,
    ParseErrorKind<P, T, Idx>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Parse error at {}: {}", self.span, self.kind)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseErrorKind<P, T, Idx> {
    EndOfInput { expected: &'static str },
    UnexpectedToken { expected: &'static str },
    MismatchedDelimiter { opening: Span<Idx> },
    UnmatchedRightDelimiter,
    UnmatchedLeftDelimiter,
    Parser(P),
    Tokenizer(T),
}

impl<P, T, Idx> fmt::Display for ParseErrorKind<P, T, Idx>
where
    P: fmt::Display,
    T: fmt::Display,
    Idx: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::EndOfInput { expected } => {
                write!(f, "Unexpected end of input (expected {expected})")
            }
            Self::UnexpectedToken { expected } => {
                write!(f, "Unexpected token (expected {expected})")
            }
            Self::MismatchedDelimiter { opening } => {
                write!(f, "Mismatched closing delimiter (opening {opening})")
            }
            Self::UnmatchedRightDelimiter => f.write_str("Unmatched closing delimiter"),
            Self::UnmatchedLeftDelimiter => f.write_str("Unmatched opening delimiter"),
            Self::Parser(err) => fmt::Display::fmt(err, f),
            Self::Tokenizer(err) => fmt::Display::fmt(err, f),
        }
    }
}

impl<P, T, Idx> ParseErrorKind<P, T, Idx> {
    #[cfg(test)]
    pub(crate) fn map_tokenizer_error<U>(
        self,
        f: impl FnOnce(T) -> U,
    ) -> ParseErrorKind<P, U, Idx> {
        match self {
            Self::EndOfInput { expected } => ParseErrorKind::EndOfInput { expected },
            Self::UnexpectedToken { expected } => ParseErrorKind::UnexpectedToken { expected },
            Self::MismatchedDelimiter { opening } => {
                ParseErrorKind::MismatchedDelimiter { opening }
            }
            Self::UnmatchedRightDelimiter => ParseErrorKind::UnmatchedRightDelimiter,
            Self::UnmatchedLeftDelimiter => ParseErrorKind::UnmatchedLeftDelimiter,
            Self::Parser(e) => ParseErrorKind::Parser(e),
            Self::Tokenizer(e) => ParseErrorKind::Tokenizer(f(e)),
        }
    }
}
