use crate::{operator, Span};

#[derive(Clone, Copy, Debug)]
pub struct Expression<'a> {
    pub kind: ExprKind,
    pub span: Span,
    pub source: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub enum ExpressionKind {
    BinaryOperator(operator::BinaryOperator),
    UnaryOperator(operator::UnaryOperator),
    Integer(i64),
    Float(f64),
}
