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
}

impl<E, Idx, B, U, T> Evaluator<Idx, B, U, T> for &'_ mut E
where
    E: Evaluator<Idx, B, U, T> + ?Sized,
{
    type Value = E::Value;
    type Error = E::Error;

    fn evaluate_binary_operator(
        &mut self,
        span: Span<Idx>,
        operator: B,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        E::evaluate_binary_operator(self, span, operator, lhs, rhs)
    }

    fn evaluate_unary_operator(
        &mut self,
        span: Span<Idx>,
        operator: U,
        argument: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        E::evaluate_unary_operator(self, span, operator, argument)
    }

    fn evaluate_term(&mut self, span: Span<Idx>, term: T) -> Result<Self::Value, Self::Error> {
        E::evaluate_term(self, span, term)
    }
}

/// A wrapper around an [`Evaluator`] that implements [`Extend`] by evaluating every item as it
/// comes in.
pub struct ImmediateEvaluator<E, V, Error> {
    evaluator: E,
    stack: Vec<V>,
    error: Option<Error>,
}

impl<E, V, Error> ImmediateEvaluator<E, V, Error> {
    const STACK_EMPTY: &'static str = "tried to pop from empty stack";

    pub fn new(evaluator: E) -> Self {
        Self {
            evaluator,
            stack: Vec::new(),
            error: None,
        }
    }

    /// Complete the evaluation and return the final result.
    ///
    /// # Panics
    ///
    /// This function will panic if there is not at least one value on the stack.
    pub fn finish(mut self) -> Result<V, Error> {
        if let Some(err) = self.error {
            Err(err)
        } else {
            Ok(self.stack.pop().expect(Self::STACK_EMPTY))
        }
    }
}

impl<E: Default, V, Error> Default for ImmediateEvaluator<E, V, Error> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<E, Idx, B, U, T> Extend<Expression<Idx, B, U, T>> for ImmediateEvaluator<E, E::Value, E::Error>
where
    E: Evaluator<Idx, B, U, T>,
{
    /// Evaluate the expressions from the iterator.
    ///
    /// If the evaluation returns an error, the error will be stored and the evaluation will stop
    /// without consuming any more expressions from the iterator.
    ///
    /// # Panics
    ///
    /// This function will panic if it encounters an operator and the stack does not contain enough
    /// values for the operator's arguments.
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
pub fn evaluate<E, I, Idx, B, U, T>(evaluator: E, input: I) -> Result<E::Value, E::Error>
where
    E: Evaluator<Idx, B, U, T>,
    I: IntoIterator<Item = Expression<Idx, B, U, T>>,
{
    let mut evaluator = ImmediateEvaluator::new(evaluator);
    evaluator.extend(input);
    evaluator.finish()
}

/// An [`Evaluator`] whose `Value` type is the same as its `Term` type, and whose operators
/// are pure functions on that type that return `Result<Term, E>`
#[derive(Default)]
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

/// An [`Evaluator`] which simply collects its expressions into an abstract syntax tree.
pub struct TreeEvaluator<Tree>(std::marker::PhantomData<fn() -> Tree>);

impl<Tree> Default for TreeEvaluator<Tree> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<Idx, B, U, T, Tree> Evaluator<Idx, B, U, T> for TreeEvaluator<Tree>
where
    Tree: ExpressionTree<Idx, B, U, T>,
{
    type Value = Tree;
    type Error = std::convert::Infallible;

    fn evaluate_binary_operator(
        &mut self,
        span: Span<Idx>,
        operator: B,
        left: Self::Value,
        right: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        Ok(Tree::from_node(
            span,
            ExpressionNode::Binary {
                operator,
                left,
                right,
            },
        ))
    }

    fn evaluate_unary_operator(
        &mut self,
        span: Span<Idx>,
        operator: U,
        argument: Self::Value,
    ) -> Result<Self::Value, Self::Error> {
        Ok(Tree::from_node(
            span,
            ExpressionNode::Unary { operator, argument },
        ))
    }

    fn evaluate_term(&mut self, span: Span<Idx>, value: T) -> Result<Self::Value, Self::Error> {
        Ok(Tree::from_node(span, ExpressionNode::Term { value }))
    }
}

pub trait ExpressionTree<Idx, B, U, T>: Sized {
    fn from_node(span: Span<Idx>, node: ExpressionNode<Self, B, U, T>) -> Self;
}

pub enum ExpressionNode<Tree, B, U, T> {
    Binary {
        operator: B,
        left: Tree,
        right: Tree,
    },
    Unary {
        operator: U,
        argument: Tree,
    },
    Term {
        value: T,
    },
}

#[cfg(test)]
mod tests {
    use test_case::test_case;

    use super::{evaluate, PureEvaluator};
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
        let actual = evaluate(
            PureEvaluator,
            expression.into_iter().map(|kind| Expression {
                kind,
                span: EMPTY_SPAN,
            }),
        );

        assert_eq!(actual, result);
    }
}
