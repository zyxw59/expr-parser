use std::collections::BTreeMap;

use crate::{
    evaluate::EvaluationContext,
    expression::FromExpressionValue,
    parser::{Delimiter, ParseContext, Postfix, Prefix},
    token::Token,
};

pub struct MapParseContext<D, B, U> {
    pub prefixes: BTreeMap<&'static str, Prefix<D, U>>,
    pub postfixes: BTreeMap<&'static str, Postfix<D, B, U>>,
}

impl<'s, D, B, U> ParseContext<'s> for MapParseContext<D, B, U>
where
    D: Delimiter + Clone,
    B: Clone,
    U: Clone,
{
    type Delimiter = D;
    type BinaryOperator = B;
    type UnaryOperator = U;

    fn get_prefix(&self, token: Token<'s>) -> Prefix<Self::Delimiter, Self::UnaryOperator> {
        self.prefixes
            .get(token.as_str())
            .cloned()
            .unwrap_or(Prefix::None)
    }

    fn get_postfix(
        &self,
        token: Token<'s>,
    ) -> Postfix<Self::Delimiter, Self::BinaryOperator, Self::UnaryOperator> {
        self.postfixes
            .get(token.as_str())
            .cloned()
            .unwrap_or(Postfix::None)
    }
}

pub struct VariableMapEvaluationContext<V> {
    pub variables: BTreeMap<&'static str, V>,
}

impl<'s, B, U, V, E> EvaluationContext<'s, B, U> for VariableMapEvaluationContext<V>
where
    B: FnOnce(V, V) -> Result<V, E>,
    U: FnOnce(V) -> Result<V, E>,
    V: FromExpressionValue<'s> + Clone,
    E: From<V::Error> + From<VariableNotFound<'s>>,
{
    type Value = V;
    type Error = E;

    fn evaluate_integer(&self, _token: Token<'s>, value: i64) -> Result<V, E> {
        V::from_integer(value).map_err(E::from)
    }

    fn evaluate_float(&self, _token: Token<'s>, value: f64) -> Result<V, E> {
        V::from_float(value).map_err(E::from)
    }

    fn evaluate_string(&self, _token: Token<'s>, value: &'s str) -> Result<V, E> {
        V::from_string(value).map_err(E::from)
    }

    fn evaluate_variable(&self, _token: Token<'s>, ident: &'s str) -> Result<V, E> {
        self.variables
            .get(ident)
            .cloned()
            .ok_or(VariableNotFound(ident))
            .map_err(E::from)
    }

    fn evaluate_binary_operator(
        &self,
        _token: Token<'s>,
        operator: B,
        lhs: V,
        rhs: V,
    ) -> Result<V, E> {
        operator(lhs, rhs)
    }

    fn evaluate_unary_operator(&self, _token: Token<'s>, operator: U, argument: V) -> Result<V, E> {
        operator(argument)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[error("variable not found: {0}")]
pub struct VariableNotFound<'s>(pub &'s str);
