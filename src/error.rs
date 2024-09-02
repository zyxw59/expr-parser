use std::fmt;

use crate::{parser::Parser, token::Tokenizer, Span};

#[derive(Clone, Debug, thiserror::Error)]
pub struct ParseErrors<P, T, Idx> {
    pub errors: Vec<ParseError<P, T, Idx>>,
}

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

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("Parse error at {span}: {kind}")]
pub struct ParseError<P, T, Idx> {
    pub span: Span<Idx>,
    pub kind: ParseErrorKind<P, T, Idx>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ParseErrorKind<P, T, Idx> {
    #[error("Unexpected end of input (expected {expected})")]
    EndOfInput { expected: &'static str },
    #[error("Unexpected token (expected {expected})")]
    UnexpectedToken { expected: &'static str },
    #[error("Mismatched closing delimiter (opening {opening})")]
    MismatchedDelimiter { opening: Span<Idx> },
    #[error("Unmatched closing delimiter")]
    UnmatchedRightDelimiter,
    #[error("Unmatched opening delimiter")]
    UnmatchedLeftDelimiter,
    #[error(transparent)]
    Parser(P),
    #[error(transparent)]
    Tokenizer(T),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ParseIntError {
    #[error("Empty string")]
    Empty,
    #[error("Integer literal too large")]
    Overflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ParseFloatError {
    #[error("Empty string")]
    Empty,
    #[error("Invalid float literal")]
    Invalid,
}
