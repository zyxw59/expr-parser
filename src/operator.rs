use crate::Span;

#[derive(Clone, Copy, Debug)]
pub struct BinaryOperator {
    fixity: Fixity,
    span: Span,
}

#[derive(Clone, Copy, Debug)]
pub struct UnaryOperator {
    precedence: Precedence,
    span: Span,
}

#[derive(Clone, Copy, Debug)]
pub enum Fixity {
    Left(Precedence),
    Right(Precedence),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Precedence {
    /// Baseline precedence, such as a top-level expression, or one enclosed in parentheses.
    Base,
    /// Comma
    Comma,
    /// Comparison operators, such as `==`, `>`, `!=` etc.
    Comparison,
    /// Additive operators, such as `+`, `-`, `++`, etc.
    Additive,
    /// Multiplicative operators, such as `*`, `/`, etc., as well as unary minus.
    Multiplicative,
    /// Exponential operators, such as `^`, as well as unary sine and cosine.
    Exponential,
}
