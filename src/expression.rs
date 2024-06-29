use crate::Span;

#[derive(Clone, Copy, Debug)]
pub struct Expression<B, U, T> {
    pub kind: ExpressionKind<B, U, T>,
    pub span: Span,
}

#[derive(Clone, Copy, Debug)]
pub enum ExpressionKind<B, U, T> {
    BinaryOperator(B),
    UnaryOperator(U),
    Term(T),
}
