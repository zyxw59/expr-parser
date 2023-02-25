use crate::token::Token;

#[derive(Clone, Copy, Debug)]
pub struct Expression<'s, B, U> {
    pub kind: ExpressionKind<B, U>,
    pub token: Token<'s>,
}

#[derive(Clone, Copy, Debug)]
pub enum ExpressionKind<B, U> {
    BinaryOperator(B),
    UnaryOperator(U),
    Integer(i64),
    Float(f64),
    String,
    Variable,
}

pub trait FromExpressionValue<'s>: Sized {
    type Error;

    fn from_integer(value: i64) -> Result<Self, Self::Error>;
    fn from_float(value: f64) -> Result<Self, Self::Error>;
    fn from_string(value: &'s str) -> Result<Self, Self::Error>;
}
