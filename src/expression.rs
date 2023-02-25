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
