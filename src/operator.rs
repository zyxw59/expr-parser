#[derive(Clone, Copy, Debug)]
pub enum Fixity {
    Left(Precedence),
    Right(Precedence),
}

impl Fixity {
    pub fn precedence(&self) -> Precedence {
        match self {
            Fixity::Left(prec) | Fixity::Right(prec) => *prec,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Precedence {
    /// Baseline precedence, such as a top-level expression, or one enclosed in delimiters.
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
