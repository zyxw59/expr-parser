use crate::{
    expression::{Expression, ExpressionKind},
    operator,
    token::{TokenKind, Tokenizer},
    Span,
};

pub struct Parser<'s> {
    tokenizer: Tokenizer<'s>,
    state: State,
    stack: Stack,
}

impl<'s> Parser<'s> {
    pub fn new(source: &'s str) -> Self {
        Parser {
            tokenizer: Tokenizer::new(source),
            state: State::ExpectTerm,
            stack: Stack::new(),
        }
    }

    pub fn parse_term(&mut self) -> Option<Expression> {
        let token = self.tokenizer.next_token()?;
        match token.kind() {
            TokenKind::Tag => {
                // could be `(`, a unary operator, or a variable
                todo!();
            }
            TokenKind::Integer => {
                // parse integer
                todo!();
            }
            TokenKind::Float => {
                // parse float
                todo!();
            }
            TokenKind::String => {
                self.state = State::ExpectOperator;
                Some(Expression {
                    kind: ExpressionKind::String,
                    span: token.span(),
                    source: token.source(),
                })
            }
            TokenKind::UnterminatedString => {
                // unterminated string error
                todo!();
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum State {
    ExpectTerm,
    ExpectOperator,
}

#[derive(Clone, Debug, Default)]
struct Stack(Vec<StackElement>);

impl Stack {
    fn new() -> Self {
        Default::default()
    }
}

#[derive(Clone, Copy, Debug)]
enum StackElement {
    BinaryOperator(operator::BinaryOperator),
    UnaryOperator(operator::UnaryOperator),
    Paren(Paren),
}

#[derive(Clone, Copy, Debug)]
struct Paren {
    span: Span,
}
