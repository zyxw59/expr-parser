use std::fmt;

use crate::{token::Token, Span};

#[derive(Clone, Debug, thiserror::Error)]
pub struct ParseErrors<'s> {
    pub errors: Vec<ParseError<'s>>,
}

impl<'s> fmt::Display for ParseErrors<'s> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.errors.len() == 1 {
            f.write_str("Encountered 1 error:")?;
        } else {
            writeln!(f, "Encountered {} errors:", self.errors.len())?;
        }
        for error in &self.errors {
            writeln!(f, "{error}")?;
        }
        Ok(())
    }
}

impl<'s> From<Vec<ParseError<'s>>> for ParseErrors<'s> {
    fn from(errors: Vec<ParseError<'s>>) -> Self {
        ParseErrors { errors }
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("Parse error at {span}: {kind}")]
pub struct ParseError<'s> {
    span: Span,
    kind: ParseErrorKind<'s>,
}

impl<'s> ParseError<'s> {
    pub fn spanned<E>(error: E, span: Span) -> Self
    where
        ParseErrorKind<'s>: From<E>,
    {
        ParseError {
            span,
            kind: error.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum ParseErrorKind<'s> {
    #[error("Unexpected end of input (expected {expected})")]
    EndOfInput { expected: &'static str },
    #[error("Unexpected token {actual} (expected {expected})")]
    UnexpectedToken {
        actual: Token<'s>,
        expected: &'static str,
    },
    #[error("Invalid integer literal: {0}")]
    ParseInt(#[from] ParseIntError),
    #[error("Unterminated string literal")]
    UnterminatedString,
    #[error("Mismatched closing delimiter: {left} / {right}")]
    MismatchedDelimiter { left: Token<'s>, right: Token<'s> },
    #[error("Unmatched closing delimiter: {right}")]
    UnmatchedRightDelimiter { right: Token<'s> },
    #[error("Unmatched opening delimiter: {left}")]
    UnmatchedLeftDelimiter { left: Token<'s> },
    #[error("Operator with `Base` precedence")]
    OperatorWithBasePrecedence,
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum ParseIntError {
    #[error("Empty string")]
    Empty,
    #[error("Integer literal too large")]
    Overflow,
}
