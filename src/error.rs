use std::fmt;

use crate::Span;

#[derive(Clone, Debug, thiserror::Error)]
pub struct ParseErrors<P, T> {
    pub errors: Vec<ParseError<P, T>>,
}

impl<P: fmt::Display, T: fmt::Display> fmt::Display for ParseErrors<P, T> {
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

impl<P, T> From<Vec<ParseError<P, T>>> for ParseErrors<P, T> {
    fn from(errors: Vec<ParseError<P, T>>) -> Self {
        ParseErrors { errors }
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("Parse error at {span}: {kind}")]
pub struct ParseError<P, T> {
    pub span: Span,
    pub kind: ParseErrorKind<P, T>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ParseErrorKind<P, T> {
    #[error("Unexpected end of input (expected {expected})")]
    EndOfInput { expected: &'static str },
    #[error("Unexpected token (expected {expected})")]
    UnexpectedToken { expected: &'static str },
    #[error("Mismatched closing delimiter (opening {opening})")]
    MismatchedDelimiter { opening: Span },
    #[error("Unmatched closing delimiter")]
    UnmatchedRightDelimiter,
    #[error("Unmatched opening delimiter")]
    UnmatchedLeftDelimiter,
    #[error(transparent)]
    Parser(P),
    #[error(transparent)]
    Tokenizer(T),
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
