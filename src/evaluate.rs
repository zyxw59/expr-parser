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

pub struct ImmediateEvaluator<'e, E: ?Sized, V, Error> {
    evaluator: &'e mut E,
    stack: Vec<V>,
    error: Option<Error>,
}

impl<'e, E: ?Sized, V, Error> ImmediateEvaluator<'e, E, V, Error> {
    const STACK_EMPTY: &'static str = "tried to pop from empty stack";

    pub fn new(evaluator: &'e mut E) -> Self {
        Self {
            evaluator,
            stack: Vec::new(),
            error: None,
        }
    }

    pub fn finish(mut self) -> Result<V, Error> {
        if let Some(err) = self.error {
            Err(err)
        } else {
            Ok(self.stack.pop().expect(Self::STACK_EMPTY))
        }
    }
}

impl<E, Idx, B, U, T> Extend<Expression<Idx, B, U, T>>
    for ImmediateEvaluator<'_, E, E::Value, E::Error>
where
    E: Evaluator<Idx, B, U, T> + ?Sized,
{
    /// Evaluate the expressions from the iterator
    ///
    /// # Panics
    ///
    /// This function will panic if it encounters an operator and the stack does not contain enough
    /// values for the operator's arguments. It will also panic if the input is empty.
    fn extend<It>(&mut self, iter: It)
    where
        It: IntoIterator<Item = Expression<Idx, B, U, T>>,
    {
        for expr in iter {
            match expr.kind {
                ExpressionKind::BinaryOperator(op) => {
                    let rhs = self.stack.pop().expect(Self::STACK_EMPTY);
                    let lhs = self.stack.pop().expect(Self::STACK_EMPTY);
                    match self
                        .evaluator
                        .evaluate_binary_operator(expr.span, op, lhs, rhs)
                    {
                        Ok(val) => self.stack.push(val),
                        Err(err) => {
                            self.error = Some(err);
                            break;
                        }
                    }
                }
                ExpressionKind::UnaryOperator(op) => {
                    let argument = self.stack.pop().expect(Self::STACK_EMPTY);
                    match self
                        .evaluator
                        .evaluate_unary_operator(expr.span, op, argument)
                    {
                        Ok(val) => self.stack.push(val),
                        Err(err) => {
                            self.error = Some(err);
                            break;
                        }
                    }
                }
                ExpressionKind::Term(term) => match self.evaluator.evaluate_term(expr.span, term) {
                    Ok(val) => self.stack.push(val),
                    Err(err) => {
                        self.error = Some(err);
                        break;
                    }
                },
            }
        }
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
    let mut evaluator = ImmediateEvaluator::new(evaluator);
    evaluator.extend(input);
    evaluator.finish()
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
