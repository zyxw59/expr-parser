use std::{fmt, ops::Range};

pub mod error;
pub mod evaluate;
pub mod expression;
pub mod operator;
pub mod parser;
pub mod token;

pub use error::{ParseError, ParseErrorKind, ParseErrors};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Span<T> {
    pub start: T,
    pub end: T,
}

impl<T> Span<T> {
    pub fn new(range: Range<T>) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }

    /// Creates a new span encompassing both input spans.
    pub fn join(self, other: Self) -> Self
    where
        T: Ord,
    {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub fn into_range(self) -> Range<T> {
        self.into()
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Span<U> {
        Span {
            start: f(self.start),
            end: f(self.end),
        }
    }
}

impl<T> From<Span<T>> for Range<T> {
    fn from(span: Span<T>) -> Self {
        span.start..span.end
    }
}

impl<T> From<Range<T>> for Span<T> {
    fn from(range: Range<T>) -> Self {
        Self::new(range)
    }
}

impl<T: fmt::Display> fmt::Display for Span<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}
