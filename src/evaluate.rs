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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use test_case::test_case;

    use super::evaluate;
    use crate::{
        expression::FromExpressionValue,
        operator::{Fixity, Precedence},
        parser::{Delimiter, Parser, Postfix, Prefix},
        simple_contexts::{MapParseContext, VariableMapEvaluationContext, VariableNotFound},
    };

    #[derive(Clone, Debug, PartialEq)]
    enum Value<'s> {
        Integer(i64),
        Float(f64),
        String(Cow<'s, str>),
        Tuple(Vec<Value<'s>>),
        List(Vec<Value<'s>>),
    }

    impl Value<'_> {
        fn neg(this: Value<'_>) -> Result<Value<'_>, Error> {
            match this {
                Value::Integer(value) => Ok(Value::Integer(-value)),
                Value::Float(value) => Ok(Value::Float(-value)),
                _ => Err(Error::Type),
            }
        }

        fn list(this: Value<'_>) -> Result<Value<'_>, Error> {
            match this {
                Value::Tuple(values) => Ok(Value::List(values)),
                other => Ok(Value::List(vec![other])),
            }
        }

        fn comma<'s>(lhs: Value<'s>, rhs: Value<'s>) -> Result<Value<'s>, Error> {
            match lhs {
                Value::Tuple(mut values) => {
                    values.push(rhs);
                    Ok(Value::Tuple(values))
                }
                lhs => Ok(Value::Tuple(vec![lhs, rhs])),
            }
        }

        fn add<'s>(lhs: Value<'s>, rhs: Value<'s>) -> Result<Value<'s>, Error> {
            match (lhs, rhs) {
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                (Value::Float(a), Value::Integer(b)) | (Value::Integer(b), Value::Float(a)) => {
                    Ok(Value::Float(a + b as f64))
                }
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a + b)),
                _ => Err(Error::Type),
            }
        }

        fn sub<'s>(lhs: Value<'s>, rhs: Value<'s>) -> Result<Value<'s>, Error> {
            match (lhs, rhs) {
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                (Value::Float(a), Value::Integer(b)) | (Value::Integer(b), Value::Float(a)) => {
                    Ok(Value::Float(a - b as f64))
                }
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a - b)),
                _ => Err(Error::Type),
            }
        }

        fn mul<'s>(lhs: Value<'s>, rhs: Value<'s>) -> Result<Value<'s>, Error> {
            match (lhs, rhs) {
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                (Value::Float(a), Value::Integer(b)) | (Value::Integer(b), Value::Float(a)) => {
                    Ok(Value::Float(a * b as f64))
                }
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a * b)),
                _ => Err(Error::Type),
            }
        }

        fn div<'s>(lhs: Value<'s>, rhs: Value<'s>) -> Result<Value<'s>, Error> {
            match (lhs, rhs) {
                (_, Value::Float(b)) if b == 0.0 => Err(Error::Arithmetic),
                (_, Value::Integer(b)) if b == 0 => Err(Error::Arithmetic),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
                (Value::Float(a), Value::Integer(b)) | (Value::Integer(b), Value::Float(a)) => {
                    Ok(Value::Float(a / b as f64))
                }
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Float((a as f64) / (b as f64))),
                _ => Err(Error::Type),
            }
        }

        fn pow<'s>(lhs: Value<'s>, rhs: Value<'s>) -> Result<Value<'s>, Error> {
            match (lhs, rhs) {
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.powf(b))),
                (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a.powi(b as i32))),
                (Value::Integer(a), Value::Float(b)) => Ok(Value::Float((a as f64).powf(b))),
                (Value::Integer(a), Value::Integer(b)) => {
                    if let Ok(b) = b.try_into() {
                        Ok(Value::Integer(a.pow(b)))
                    } else {
                        Ok(Value::Float((a as f64).powi(b as i32)))
                    }
                }
                _ => Err(Error::Type),
            }
        }

        fn factorial(this: Value<'_>) -> Result<Value<'_>, Error> {
            if let Value::Integer(mut value) = this {
                if value < 0 {
                    return Err(Error::Arithmetic);
                }
                let mut acc = 1;
                while value > 1 {
                    acc *= value;
                    value -= 1;
                }
                Ok(Value::Integer(acc))
            } else {
                Err(Error::Type)
            }
        }
    }

    impl<'s> FromExpressionValue<'s> for Value<'s> {
        type Error = Error;

        fn from_integer(value: i64) -> Result<Self, Self::Error> {
            Ok(Value::Integer(value))
        }

        fn from_float(value: f64) -> Result<Self, Self::Error> {
            Ok(Value::Float(value))
        }

        fn from_string(value: &'s str) -> Result<Self, Self::Error> {
            Ok(Value::String(Cow::Borrowed(value)))
        }
    }

    #[derive(Debug, Eq, PartialEq, thiserror::Error)]
    enum Error {
        #[error("arithmetic error")]
        Arithmetic,
        #[error("type error")]
        Type,
        #[error("variable not found: {0}")]
        VariableNotFound(String),
    }

    impl From<VariableNotFound<'_>> for Error {
        fn from(err: VariableNotFound<'_>) -> Self {
            Error::VariableNotFound(err.0.to_owned())
        }
    }

    #[derive(Clone, Copy, Eq, PartialEq)]
    enum SimpleDelimiter {
        Paren,
        SquareBracket,
    }

    impl Delimiter for SimpleDelimiter {
        fn matches(&self, other: &Self) -> bool {
            self == other
        }
    }

    type BinFunc = for<'s> fn(Value<'s>, Value<'s>) -> Result<Value<'s>, Error>;
    type UnFunc = for<'s> fn(Value<'s>) -> Result<Value<'s>, Error>;

    lazy_static::lazy_static! {
        static ref PARSE_CONTEXT: MapParseContext<SimpleDelimiter, BinFunc, UnFunc> =
            MapParseContext {
                prefixes: [
                    ("-", Prefix::UnaryOperator {
                        precedence: Precedence::Multiplicative,
                        operator: Value::neg as _,
                    }),
                    ("(", Prefix::Delimiter {
                        delimiter: SimpleDelimiter::Paren,
                        operator: None,
                    }),
                    ("[", Prefix::Delimiter {
                        delimiter: SimpleDelimiter::SquareBracket,
                        operator: Some(Value::list as _),
                    }),
                ].into_iter().collect(),
                postfixes: [
                    (",", Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Comma),
                        operator: Value::comma as _,
                    }),
                    ("+", Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Additive),
                        operator: Value::add as _,
                    }),
                    ("-", Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Additive),
                        operator: Value::sub as _,
                    }),
                    ("*", Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Multiplicative),
                        operator: Value::mul as _,
                    }),
                    ("/", Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Multiplicative),
                        operator: Value::div as _,
                    }),
                    ("^", Postfix::BinaryOperator {
                        fixity: Fixity::Right(Precedence::Exponential),
                        operator: Value::pow as _,
                    }),
                    ("!", Postfix::PostfixOperator {
                        precedence: Precedence::Exponential,
                        operator: Value::factorial as _,
                    }),
                    (")", Postfix::RightDelimiter {
                        delimiter: SimpleDelimiter::Paren,
                    }),
                    ("]", Postfix::RightDelimiter {
                        delimiter: SimpleDelimiter::SquareBracket,
                    }),
                    ].into_iter().collect(),
            };

        static ref EVALUATION_CONTEXT: VariableMapEvaluationContext<Value<'static>> =
            VariableMapEvaluationContext {
                variables: [].into_iter().collect(),
            };
    }

    #[test_case("3 + 4 * 2 / (1 - 5)^2", Value::Float(3.5) ; "simple arithmetic" )]
    #[test_case("2^3!", Value::Integer(64) ; "postfix operators" )]
    #[test_case("-2^2 * (-2)^2", Value::Integer(-16); "prefix operators" )]
    #[test_case("[1, 2, 3]", Value::List(vec![
            Value::Integer(1), Value::Integer(2), Value::Integer(3)
    ]) ; "list" )]
    #[test_case("-4!", Value::Integer(-24); "pre and postfix" )]
    fn evaluate_expression(input: &str, expected: Value) -> anyhow::Result<()> {
        let values = Parser::new(input, &*PARSE_CONTEXT).parse()?;
        let actual = evaluate(&*EVALUATION_CONTEXT, values)?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
