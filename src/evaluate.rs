use crate::{
    expression::{Expression, ExpressionKind},
    token::Token,
};

pub trait EvaluationContext<'s, B, U> {
    type Value;
    type Error;

    fn evaluate_integer(&self, token: Token<'s>, value: i64) -> Result<Self::Value, Self::Error>;

    fn evaluate_float(&self, token: Token<'s>, value: f64) -> Result<Self::Value, Self::Error>;

    fn evaluate_string(&self, token: Token<'s>, value: &'s str)
        -> Result<Self::Value, Self::Error>;

    fn evaluate_variable(
        &self,
        token: Token<'s>,
        ident: &'s str,
    ) -> Result<Self::Value, Self::Error>;

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
}

/// Evaluate the input expression queue using the provided `EvaluationContext`.
///
/// # Panics
///
/// This function will panic if it encounters an operator and the stack does not contain enough
/// values for the operator's arguments. It will also panic if the input is empty.
pub fn evaluate<'s, C, I, B, U>(context: &C, input: I) -> Result<C::Value, C::Error>
where
    C: EvaluationContext<'s, B, U>,
    I: IntoIterator<Item = Expression<'s, B, U>>,
{
    const STACK_EMPTY: &str = "tried to pop from empty stack";

    let mut stack = Vec::new();
    for expr in input {
        match expr.kind {
            ExpressionKind::Integer(value) => {
                stack.push(context.evaluate_integer(expr.token, value)?)
            }
            ExpressionKind::Float(value) => stack.push(context.evaluate_float(expr.token, value)?),
            ExpressionKind::String => {
                stack.push(context.evaluate_string(expr.token, expr.token.as_str())?)
            }
            ExpressionKind::Variable => {
                stack.push(context.evaluate_variable(expr.token, expr.token.as_str())?)
            }
            ExpressionKind::BinaryOperator(op) => {
                let rhs = stack.pop().expect(STACK_EMPTY);
                let lhs = stack.pop().expect(STACK_EMPTY);
                stack.push(context.evaluate_binary_operator(expr.token, op, lhs, rhs)?);
            }
            ExpressionKind::UnaryOperator(op) => {
                let argument = stack.pop().expect(STACK_EMPTY);
                stack.push(context.evaluate_unary_operator(expr.token, op, argument)?);
            }
        }
    }

    Ok(stack.pop().expect(STACK_EMPTY))
}
