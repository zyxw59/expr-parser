use crate::{operator, token::Tokenizer, Span};

pub struct Parser<'s> {
    tokenizer: Tokenizer<'s>,
    stack: Stack,
}

impl<'s> Parser<'s> {
    pub fn new(source: &'s str) -> Self {
        Parser {
            tokenizer: Tokenizer::new(source),
            stack: Stack::new(),
        }
    }
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
