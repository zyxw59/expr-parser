use crate::{operator, token::Token};

#[derive(Clone, Copy, Debug)]
pub struct Expression<'s> {
    pub kind: ExpressionKind<'s>,
    pub token: Token<'s>,
}

#[derive(Clone, Copy, Debug)]
pub enum ExpressionKind<'s> {
    BinaryOperator(operator::BinaryOperator<'s>),
    UnaryOperator(operator::UnaryOperator<'s>),
    Integer(i64),
    Float(f64),
    String,
    Variable,
}
