use crate::Span;

#[derive(Clone, Copy, Debug, Eq)]
pub struct Expression<Idx, B, U, T> {
    pub kind: ExpressionKind<B, U, T>,
    pub span: Span<Idx>,
}

/// Expression equality ignores the span
impl<Idx, B: PartialEq, U: PartialEq, T: PartialEq> PartialEq for Expression<Idx, B, U, T> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpressionKind<B, U, T> {
    BinaryOperator(B),
    UnaryOperator(U),
    Term(T),
}
