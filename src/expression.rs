use crate::token::Token;

#[derive(Clone, Copy, Debug)]
pub struct Expression<'s, B, U, T> {
    pub kind: ExpressionKind<B, U, T>,
    pub token: Token<'s>,
}

#[derive(Clone, Copy, Debug)]
pub enum ExpressionKind<B, U, T> {
    BinaryOperator(B),
    UnaryOperator(U),
    Term(T),
    Null,
}
