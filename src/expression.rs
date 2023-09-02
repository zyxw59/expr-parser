use crate::token::Token;

pub trait ExpressionTypes {
    type BinaryOperator;
    type UnaryOperator;
    type Term;
}

#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = "ExpressionKind<T>: Clone"),
    Copy(bound = "ExpressionKind<T>: Copy"),
    Debug(bound = "ExpressionKind<T>: std::fmt::Debug")
)]
pub struct Expression<'s, T: ExpressionTypes> {
    pub kind: ExpressionKind<T>,
    pub token: Token<'s>,
}

#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = "T::BinaryOperator: Clone, T::UnaryOperator: Clone, T::Term: Clone"),
    Copy(bound = "T::BinaryOperator: Copy, T::UnaryOperator: Copy, T::Term: Copy"),
    Debug(
        bound = "T::BinaryOperator: std::fmt::Debug, T::UnaryOperator: std::fmt::Debug, T::Term: std::fmt::Debug"
    )
)]
pub enum ExpressionKind<T: ExpressionTypes> {
    BinaryOperator(T::BinaryOperator),
    UnaryOperator(T::UnaryOperator),
    Term(T::Term),
    Null,
}
