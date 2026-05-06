use std::{error::Error, fmt};

use crate::{token::Tokenizer, Span};

#[derive(Clone, Debug)]
pub struct ParseErrors<T, Idx> {
    pub errors: Vec<ParseError<T, Idx>>,
}

impl<T, Idx> Error for ParseErrors<T, Idx> where Self: fmt::Debug + fmt::Display {}

pub type ParseErrorsFor<T> = ParseErrors<<T as Tokenizer>::Error, <T as Tokenizer>::Position>;

impl<T: fmt::Display, Idx: fmt::Display> fmt::Display for ParseErrors<T, Idx> {
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

impl<T, Idx> From<Vec<ParseError<T, Idx>>> for ParseErrors<T, Idx> {
    fn from(errors: Vec<ParseError<T, Idx>>) -> Self {
        ParseErrors { errors }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParseError<T, Idx> {
    pub span: Span<Idx>,
    pub kind: ParseErrorKind<T, Idx>,
}

impl<T, Idx> Error for ParseError<T, Idx> where Self: fmt::Debug + fmt::Display {}

impl<T, Idx> fmt::Display for ParseError<T, Idx>
where
    Span<Idx>: fmt::Display,
    ParseErrorKind<T, Idx>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Parse error at {}: {}", self.span, self.kind)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseErrorKind<T, Idx> {
    EndOfInput { expected: &'static str },
    UnexpectedToken { expected: &'static str },
    MismatchedDelimiter { opening: Span<Idx> },
    UnmatchedClosingDelimiter,
    UnmatchedOpeningDelimiter,
    Tokenizer(T),
}

impl<T, Idx> fmt::Display for ParseErrorKind<T, Idx>
where
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
            Self::UnmatchedClosingDelimiter => f.write_str("Unmatched closing delimiter"),
            Self::UnmatchedOpeningDelimiter => f.write_str("Unmatched opening delimiter"),
            Self::Tokenizer(err) => fmt::Display::fmt(err, f),
        }
    }
}

impl<T, Idx> ParseErrorKind<T, Idx> {
    #[cfg(test)]
    pub(crate) fn map_tokenizer_error<U>(self, f: impl FnOnce(T) -> U) -> ParseErrorKind<U, Idx> {
        match self {
            Self::EndOfInput { expected } => ParseErrorKind::EndOfInput { expected },
            Self::UnexpectedToken { expected } => ParseErrorKind::UnexpectedToken { expected },
            Self::MismatchedDelimiter { opening } => {
                ParseErrorKind::MismatchedDelimiter { opening }
            }
            Self::UnmatchedClosingDelimiter => ParseErrorKind::UnmatchedClosingDelimiter,
            Self::UnmatchedOpeningDelimiter => ParseErrorKind::UnmatchedOpeningDelimiter,
            Self::Tokenizer(e) => ParseErrorKind::Tokenizer(f(e)),
        }
    }
}
