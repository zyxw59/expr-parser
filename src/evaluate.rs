use crate::{
    expression::{Expression, ExpressionKind},
    token::Token,
};

pub trait EvaluationContext<'s, B, U, T> {
    type Value;
    type Error;

    fn evaluate_binary_operator(
        &self,
        token: Token<'s>,
        operator: B,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    fn evaluate_unary_operator(
        &self,
        token: Token<'s>,
        operator: U,
        argument: Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    fn evaluate_term(&self, token: Token<'s>, term: T) -> Result<Self::Value, Self::Error>;
}

/// Evaluate the input expression queue using the provided `EvaluationContext`.
///
/// # Panics
///
/// This function will panic if it encounters an operator and the stack does not contain enough
/// values for the operator's arguments. It will also panic if the input is empty.
pub fn evaluate<'s, C, I, B, U, T>(context: &C, input: I) -> Result<C::Value, C::Error>
where
    C: EvaluationContext<'s, B, U, T>,
    I: IntoIterator<Item = Expression<'s, B, U, T>>,
{
    const STACK_EMPTY: &str = "tried to pop from empty stack";

    let mut stack = Vec::new();
    for expr in input {
        match expr.kind {
            ExpressionKind::BinaryOperator(op) => {
                let rhs = stack.pop().expect(STACK_EMPTY);
                let lhs = stack.pop().expect(STACK_EMPTY);
                stack.push(context.evaluate_binary_operator(expr.token, op, lhs, rhs)?);
            }
            ExpressionKind::UnaryOperator(op) => {
                let argument = stack.pop().expect(STACK_EMPTY);
                stack.push(context.evaluate_unary_operator(expr.token, op, argument)?);
            }
            ExpressionKind::Term(term) => {
                stack.push(context.evaluate_term(expr.token, term)?);
            }
        }
    }

    Ok(stack.pop().expect(STACK_EMPTY))
}

/// An `EvaluationContext` whose `Value` type is the same as its `Term` type, and whose operators
/// are pure functions on that type that return `Result<Term, E>`
pub struct PureEvaluator;

impl<'s, B, U, T, E> EvaluationContext<'s, B, U, T> for PureEvaluator
where
    B: FnOnce(T, T) -> Result<T, E>,
    U: FnOnce(T) -> Result<T, E>,
{
    type Value = T;
    type Error = E;

    fn evaluate_binary_operator(
        &self,
        _token: Token<'s>,
        operator: B,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        operator(lhs, rhs)
    }

    fn evaluate_unary_operator(
        &self,
        _token: Token<'s>,
        operator: U,
        argument: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        operator(argument)
    }

    fn evaluate_term(&self, _token: Token<'s>, term: T) -> Result<Self::Value, Self::Error> {
        Ok(term)
    }
}
