use crate::{
    expression::{Expression, ExpressionKind},
    Span,
};

pub trait Evaluator<Idx, B, U, T> {
    type Value;
    type Error;

    fn evaluate_binary_operator(
        &mut self,
        span: Span<Idx>,
        operator: B,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    fn evaluate_unary_operator(
        &mut self,
        span: Span<Idx>,
        operator: U,
        argument: Self::Value,
    ) -> Result<Self::Value, Self::Error>;

    fn evaluate_term(&mut self, span: Span<Idx>, term: T) -> Result<Self::Value, Self::Error>;

    fn evaluate<I>(&mut self, input: I) -> Result<Self::Value, Self::Error>
    where
        I: IntoIterator<Item = Expression<Idx, B, U, T>>,
    {
        evaluate(self, input)
    }
}

/// Evaluate the input expression queue using the provided `Evaluator`.
///
/// # Panics
///
/// This function will panic if it encounters an operator and the stack does not contain enough
/// values for the operator's arguments. It will also panic if the input is empty.
pub fn evaluate<E, I, Idx, B, U, T>(evaluator: &mut E, input: I) -> Result<E::Value, E::Error>
where
    E: Evaluator<Idx, B, U, T> + ?Sized,
    I: IntoIterator<Item = Expression<Idx, B, U, T>>,
{
    const STACK_EMPTY: &str = "tried to pop from empty stack";

    let mut stack = Vec::new();
    for expr in input {
        match expr.kind {
            ExpressionKind::BinaryOperator(op) => {
                let rhs = stack.pop().expect(STACK_EMPTY);
                let lhs = stack.pop().expect(STACK_EMPTY);
                stack.push(evaluator.evaluate_binary_operator(expr.span, op, lhs, rhs)?);
            }
            ExpressionKind::UnaryOperator(op) => {
                let argument = stack.pop().expect(STACK_EMPTY);
                stack.push(evaluator.evaluate_unary_operator(expr.span, op, argument)?);
            }
            ExpressionKind::Term(term) => {
                stack.push(evaluator.evaluate_term(expr.span, term)?);
            }
        }
    }

    Ok(stack.pop().expect(STACK_EMPTY))
}

/// An `Evaluator` whose `Value` type is the same as its `Term` type, and whose operators
/// are pure functions on that type that return `Result<Term, E>`
pub struct PureEvaluator;

impl<Idx, B, U, T, E> Evaluator<Idx, B, U, T> for PureEvaluator
where
    B: FnOnce(T, T) -> Result<T, E>,
    U: FnOnce(T) -> Result<T, E>,
{
    type Value = T;
    type Error = E;

    fn evaluate_binary_operator(
        &mut self,
        _span: Span<Idx>,
        operator: B,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        operator(lhs, rhs)
    }

    fn evaluate_unary_operator(
        &mut self,
        _span: Span<Idx>,
        operator: U,
        argument: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        operator(argument)
    }

    fn evaluate_term(&mut self, _span: Span<Idx>, term: T) -> Result<Self::Value, Self::Error> {
        Ok(term)
    }
}

#[cfg(test)]
mod tests {
    use test_case::test_case;

    use super::{Evaluator, PureEvaluator};
    use crate::{
        expression::{Expression, ExpressionKind},
        Span,
    };

    #[derive(Debug, Eq, PartialEq)]
    enum Error {
        DivideByZero,
    }

    fn add(lhs: Term, rhs: Term) -> Result<Term, Error> {
        Ok(lhs + rhs)
    }

    fn sub(lhs: Term, rhs: Term) -> Result<Term, Error> {
        Ok(lhs - rhs)
    }

    fn mul(lhs: Term, rhs: Term) -> Result<Term, Error> {
        Ok(lhs * rhs)
    }

    fn div(lhs: Term, rhs: Term) -> Result<Term, Error> {
        if rhs == 0 {
            Err(Error::DivideByZero)
        } else {
            Ok(lhs / rhs)
        }
    }

    fn neg(argument: Term) -> Result<Term, Error> {
        Ok(-argument)
    }

    type Term = i64;
    type BinaryOperator = fn(Term, Term) -> Result<Term, Error>;
    type UnaryOperator = fn(Term) -> Result<Term, Error>;

    #[test_case([
        ExpressionKind::Term(1), ExpressionKind::Term(1), ExpressionKind::BinaryOperator(add),
    ], Ok(2); "basic")]
    #[test_case([
        ExpressionKind::Term(1), ExpressionKind::Term(1), ExpressionKind::BinaryOperator(add),
        ExpressionKind::UnaryOperator(neg),
        ExpressionKind::Term(3), ExpressionKind::BinaryOperator(mul),
        ExpressionKind::Term(2), ExpressionKind::BinaryOperator(div),
    ], Ok(-3); "basic 2")]
    #[test_case([
        ExpressionKind::Term(1), ExpressionKind::Term(1), ExpressionKind::BinaryOperator(add),
        ExpressionKind::UnaryOperator(neg),
        ExpressionKind::Term(3), ExpressionKind::BinaryOperator(mul),
        ExpressionKind::Term(1), ExpressionKind::Term(1), ExpressionKind::BinaryOperator(sub),
        ExpressionKind::BinaryOperator(div),
    ], Err(Error::DivideByZero); "division by zero")]
    fn evaluate_expression<const N: usize>(
        expression: [ExpressionKind<BinaryOperator, UnaryOperator, Term>; N],
        result: Result<Term, Error>,
    ) {
        const EMPTY_SPAN: Span<usize> = Span { start: 0, end: 0 };
        let actual = PureEvaluator.evaluate(expression.into_iter().map(|kind| Expression {
            kind,
            span: EMPTY_SPAN,
        }));

        assert_eq!(actual, result);
    }
}
