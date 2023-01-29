use crate::{
    expression::{Expression, ExpressionKind},
    token::Token,
};

#[derive(Clone, Copy, Debug)]
pub struct BinaryOperator<'s> {
    fixity: Fixity,
    token: Token<'s>,
}

impl<'s> BinaryOperator<'s> {
    pub fn new(fixity: Fixity, token: Token<'s>) -> Self {
        Self { fixity, token }
    }

    pub fn token(&self) -> Token<'s> {
        self.token
    }

    pub fn fixity(&self) -> Fixity {
        self.fixity
    }

    pub fn precedence(&self) -> Precedence {
        self.fixity.precedence()
    }
}

impl<'s> From<BinaryOperator<'s>> for Expression<'s> {
    fn from(bin_op: BinaryOperator<'s>) -> Expression<'s> {
        Expression {
            kind: ExpressionKind::BinaryOperator(bin_op),
            token: bin_op.token(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UnaryOperator<'s> {
    precedence: Precedence,
    token: Token<'s>,
}

impl<'s> UnaryOperator<'s> {
    pub fn new(precedence: Precedence, token: Token<'s>) -> Self {
        Self { precedence, token }
    }

    pub fn token(&self) -> Token<'s> {
        self.token
    }

    pub fn precedence(&self) -> Precedence {
        self.precedence
    }
}

impl<'s> From<UnaryOperator<'s>> for Expression<'s> {
    fn from(un_op: UnaryOperator<'s>) -> Expression<'s> {
        Expression {
            kind: ExpressionKind::UnaryOperator(un_op),
            token: un_op.token(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LeftDelimiter<'s> {
    token: Token<'s>,
}

impl<'s> LeftDelimiter<'s> {
    pub fn new(token: Token<'s>) -> Self {
        LeftDelimiter { token }
    }

    pub fn token(&self) -> Token<'s> {
        self.token
    }

    pub fn into_application_operator(self) -> BinaryOperator<'s> {
        BinaryOperator {
            fixity: Fixity::Right(Precedence::Base),
            token: self.token,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RightDelimiter<'s> {
    token: Token<'s>,
}

impl<'s> RightDelimiter<'s> {
    pub fn new(token: Token<'s>) -> Self {
        RightDelimiter { token }
    }

    pub fn token(&self) -> Token<'s> {
        self.token
    }
}

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
