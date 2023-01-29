use std::fmt;

use crate::Span;

#[derive(Clone, Debug, thiserror::Error)]
pub struct ParseErrors {
    pub errors: Vec<ParseError>,
}

impl fmt::Display for ParseErrors {
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

impl From<Vec<ParseError>> for ParseErrors {
    fn from(errors: Vec<ParseError>) -> Self {
        ParseErrors { errors }
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("Parse error at {span}: {kind}")]
pub struct ParseError {
    span: Span,
    kind: ParseErrorKind,
}

impl ParseError {
    pub fn spanned<E>(error: E, span: Span) -> Self
    where
        ParseErrorKind: From<E>,
    {
        ParseError {
            span,
            kind: error.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum ParseErrorKind {
    #[error("Unexpected end of input (expected {expected})")]
    EndOfInput { expected: &'static str },
    #[error("Unexpected token (expected {expected})")]
    UnexpectedToken { expected: &'static str },
    #[error("Invalid integer literal: {0}")]
    ParseInt(#[from] ParseIntError),
    #[error("Unterminated string literal")]
    UnterminatedString,
    #[error("Mismatched closing delimiter (opening {opening})")]
    MismatchedDelimiter { opening: Span },
    #[error("Unmatched closing delimiter")]
    UnmatchedRightDelimiter,
    #[error("Unmatched opening delimiter")]
    UnmatchedLeftDelimiter,
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
