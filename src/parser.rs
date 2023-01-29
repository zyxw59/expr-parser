use std::collections::VecDeque;

use crate::{
    error::{ParseError, ParseErrorKind, ParseIntError},
    expression::{Expression, ExpressionKind},
    operator,
    token::{Token, TokenKind, Tokenizer},
    Span,
};

const EXPECT_TERM: &str = "literal, variable, unary operator, or delimiter";
const EXPECT_OPERATOR: &str = "binary operator, delimiter, postfix operator, or end of input";

pub struct Parser<'s, C> {
    tokenizer: Tokenizer<'s>,
    context: C,
    state: State,
    stack: Stack<'s>,
    queue: VecDeque<Expression<'s>>,
}

impl<'s, C> Parser<'s, C>
where
    C: ParseContext,
{
    pub fn new(source: &'s str, context: C) -> Self {
        Parser {
            tokenizer: Tokenizer::new(source),
            context,
            state: State::ExpectTerm,
            stack: Stack::new(),
            queue: VecDeque::new(),
        }
    }

    pub fn parse(mut self) -> Result<VecDeque<Expression<'s>>, Vec<ParseError<'s>>> {
        let mut errors = Vec::new();
        while !self.source().is_empty() {
            if let Err(err) = self.parse_next() {
                errors.push(err);
            }
        }
        if errors.is_empty() {
            Ok(self.queue)
        } else {
            Err(errors)
        }
    }

    fn parse_next(&mut self) -> Result<(), ParseError<'s>> {
        match self.state {
            State::ExpectTerm => self.parse_term(),
            State::ExpectOperator => self.parse_operator(),
        }
    }

    pub fn parse_term(&mut self) -> Result<(), ParseError<'s>> {
        let token = self
            .tokenizer
            .next_token()
            .ok_or_else(|| self.end_of_input(EXPECT_TERM))?;
        match token.kind() {
            TokenKind::Tag => match self.context.get_prefix(token) {
                Prefix::Delimiter(delim) => {
                    self.stack.push(StackElement::Delimiter(delim));
                    self.state = State::ExpectTerm;
                }
                Prefix::UnaryOperator(un_op) => {
                    self.stack.push(StackElement::UnaryOperator(un_op));
                    self.state = State::ExpectTerm;
                }
                Prefix::None => {
                    self.state = State::ExpectOperator;
                    self.queue
                        .push_back(token.to_expression(ExpressionKind::Variable));
                }
            },
            TokenKind::Integer => {
                let int = parse_integer(token.as_str())
                    .map_err(|e| ParseError::spanned(e, token.span()))?;
                self.queue
                    .push_back(token.to_expression(ExpressionKind::Integer(int)));
            }
            TokenKind::Float => {
                // parse float
                todo!();
            }
            TokenKind::String => {
                self.state = State::ExpectOperator;
                self.queue
                    .push_back(token.to_expression(ExpressionKind::String));
            }
            TokenKind::UnterminatedString => {
                // unterminated string error
                return Err(ParseError::spanned(
                    ParseErrorKind::UnterminatedString,
                    token.span(),
                ));
            }
        }
        Ok(())
    }

    pub fn parse_operator(&mut self) -> Result<(), ParseError<'s>> {
        let Some(token) = self.tokenizer.next_token() else {
            return Ok(());
        };
        match token.kind() {
            TokenKind::Tag => match self.context.get_postfix(token) {
                Postfix::RightDelimiter(right) => self.process_right_delimiter(right),
                Postfix::BinaryOperator(bin_op) => self.process_binary_operator(bin_op),
                Postfix::PostfixOperator(post_op) => self.process_postfix_operator(post_op),
                Postfix::LeftDelimiter(delim) => {
                    self.state = State::ExpectTerm;
                    // left delimiter in operator position indicates a function call or similar.
                    // this is indicated by adding a binary operator (with the same token as the
                    // delimiter) to the stack immediately after the delimiter itself. this
                    // operator will then function as the "function application" operator (or a
                    // related operator, such as "struct construction") when it is popped from the
                    // stack after the closing delimiter is matched
                    self.stack.push(StackElement::Delimiter(delim));
                    self.stack.push(StackElement::BinaryOperator(
                        delim.into_application_operator(),
                    ));
                    Ok(())
                }
                Postfix::None => {
                    self.state = State::ExpectTerm;
                    Err(ParseError::spanned(
                        ParseErrorKind::UnexpectedToken {
                            expected: EXPECT_OPERATOR,
                            actual: token,
                        },
                        token.span(),
                    ))
                }
            },
            _ => {
                self.state = State::ExpectTerm;
                Err(ParseError::spanned(
                    ParseErrorKind::UnexpectedToken {
                        expected: EXPECT_OPERATOR,
                        actual: token,
                    },
                    token.span(),
                ))
            }
        }
    }

    fn process_right_delimiter(
        &mut self,
        right: operator::RightDelimiter<'s>,
    ) -> Result<(), ParseError<'s>> {
        self.state = State::ExpectOperator;
        while let Some(op) = self.stack.pop() {
            match op {
                StackElement::Delimiter(left) => {
                    if self.context.match_delimiters(left, right) {
                        return Ok(());
                    } else {
                        return Err(ParseError::spanned(
                            ParseErrorKind::MismatchedDelimiter {
                                left: left.token(),
                                right: right.token(),
                            },
                            right.token().span(),
                        ));
                    }
                }
                StackElement::UnaryOperator(un_op) => {
                    self.queue.push_back(un_op.into());
                }
                StackElement::BinaryOperator(bin_op) => {
                    self.queue.push_back(bin_op.into());
                }
            }
        }
        Err(ParseError::spanned(
            ParseErrorKind::UnmatchedRightDelimiter {
                right: right.token(),
            },
            right.token().span(),
        ))
    }

    fn process_binary_operator(
        &mut self,
        bin_op: operator::BinaryOperator<'s>,
    ) -> Result<(), ParseError<'s>> {
        self.state = State::ExpectTerm;
        self.pop_while_lower_precedence(bin_op.fixity(), bin_op.token().span())?;
        self.stack.push(StackElement::BinaryOperator(bin_op));
        Ok(())
    }

    fn process_postfix_operator(
        &mut self,
        post_op: operator::UnaryOperator<'s>,
    ) -> Result<(), ParseError<'s>> {
        self.state = State::ExpectOperator;
        let fixity = operator::Fixity::Right(post_op.precedence());
        self.pop_while_lower_precedence(fixity, post_op.token().span())?;
        self.queue.push_back(post_op.into());
        Ok(())
    }

    fn pop_while_lower_precedence(
        &mut self,
        fixity: operator::Fixity,
        span: Span,
    ) -> Result<(), ParseError<'s>> {
        while let Some(prev_op) = self.stack.pop_if_lower_precedence(fixity) {
            match prev_op {
                StackElement::BinaryOperator(prev) => self.queue.push_back(prev.into()),
                StackElement::UnaryOperator(prev) => self.queue.push_back(prev.into()),
                StackElement::Delimiter(_) => {
                    return Err(ParseError::spanned(
                        ParseErrorKind::OperatorWithBasePrecedence,
                        span,
                    ))
                }
            }
        }
        Ok(())
    }

    fn source(&self) -> &'s str {
        self.tokenizer.source()
    }

    fn end_of_input(&self, expected: &'static str) -> ParseError<'s> {
        let len = self.source().len();
        ParseError::spanned(
            ParseErrorKind::EndOfInput { expected },
            Span {
                start: len,
                end: len,
            },
        )
    }
}

pub trait ParseContext {
    fn get_prefix<'s>(&self, token: Token<'s>) -> Prefix<'s>;

    fn get_postfix<'s>(&self, token: Token<'s>) -> Postfix<'s>;

    fn match_delimiters<'s>(
        &self,
        left: operator::LeftDelimiter<'s>,
        right: operator::RightDelimiter<'s>,
    ) -> bool;
}

pub enum Prefix<'s> {
    UnaryOperator(operator::UnaryOperator<'s>),
    Delimiter(operator::LeftDelimiter<'s>),
    None,
}

pub enum Postfix<'s> {
    BinaryOperator(operator::BinaryOperator<'s>),
    PostfixOperator(operator::UnaryOperator<'s>),
    LeftDelimiter(operator::LeftDelimiter<'s>),
    RightDelimiter(operator::RightDelimiter<'s>),
    None,
}

#[derive(Clone, Copy, Debug)]
enum State {
    ExpectTerm,
    ExpectOperator,
}

#[derive(Clone, Debug, Default)]
struct Stack<'s>(Vec<StackElement<'s>>);

impl<'s> Stack<'s> {
    fn new() -> Self {
        Default::default()
    }

    fn push(&mut self, element: StackElement<'s>) {
        self.0.push(element);
    }

    fn pop(&mut self) -> Option<StackElement<'s>> {
        self.0.pop()
    }

    /// Pops the stack if the new operator has lower precedence than the top of the stack
    fn pop_if_lower_precedence(&mut self, fixity: operator::Fixity) -> Option<StackElement<'s>> {
        if match fixity {
            operator::Fixity::Left(prec) => prec <= self.precedence(),
            operator::Fixity::Right(prec) => prec < self.precedence(),
        } {
            self.pop()
        } else {
            None
        }
    }

    fn precedence(&self) -> operator::Precedence {
        self.0
            .last()
            .map(StackElement::precedence)
            .unwrap_or(operator::Precedence::Base)
    }
}

#[derive(Clone, Copy, Debug)]
enum StackElement<'s> {
    BinaryOperator(operator::BinaryOperator<'s>),
    UnaryOperator(operator::UnaryOperator<'s>),
    Delimiter(operator::LeftDelimiter<'s>),
}

impl<'s> StackElement<'s> {
    fn precedence(&self) -> operator::Precedence {
        match self {
            StackElement::BinaryOperator(bin_op) => bin_op.precedence(),
            StackElement::UnaryOperator(un_op) => un_op.precedence(),
            StackElement::Delimiter(_) => operator::Precedence::Base,
        }
    }
}

fn parse_integer(s: &str) -> Result<i64, ParseIntError> {
    if s.is_empty() {
        return Err(ParseIntError::Empty);
    }
    let mut x: i64 = 0;
    for c in s.chars() {
        if let Some(digit) = c.to_digit(10) {
            x = x
                .checked_mul(10)
                .and_then(|x| x.checked_add(digit as i64))
                .ok_or(ParseIntError::Overflow)?;
        }
    }
    Ok(x)
}
