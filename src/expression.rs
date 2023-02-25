use crate::token::Token;

#[derive(Clone, Copy, Debug)]
pub struct Expression<'s, O> {
    pub kind: ExpressionKind<O>,
    pub token: Token<'s>,
}

#[derive(Clone, Copy, Debug)]
pub enum ExpressionKind<O> {
    Operator(O),
    Integer(i64),
    Float(f64),
    String,
    Variable,
}
