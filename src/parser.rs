use std::collections::VecDeque;

use crate::{
    error::{ParseError, ParseErrorKind, ParseErrors, ParseIntError},
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

    pub fn parse(mut self) -> Result<VecDeque<Expression<'s>>, ParseErrors> {
        let mut errors = Vec::new();
        while !self.tokenizer.is_empty() {
            if let Err(err) = self.parse_next() {
                errors.push(err);
            }
        }
        if self.state == State::ExpectTerm {
            errors.push(self.end_of_input(EXPECT_TERM));
        }
        while let Some(op) = self.stack.pop() {
            match op {
                StackElement::BinaryOperator(bin_op) => self.queue.push_back(bin_op.into()),
                StackElement::UnaryOperator(un_op) => self.queue.push_back(un_op.into()),
                StackElement::Delimiter(delim) => errors.push(ParseError {
                    kind: ParseErrorKind::UnmatchedLeftDelimiter,
                    span: delim.token().span(),
                }),
            }
        }
        if errors.is_empty() {
            Ok(self.queue)
        } else {
            Err(errors.into())
        }
    }

    fn parse_next(&mut self) -> Result<(), ParseError> {
        match self.state {
            State::ExpectTerm => self.parse_term(),
            State::ExpectOperator => self.parse_operator(),
        }
    }

    fn parse_term(&mut self) -> Result<(), ParseError> {
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
                Prefix::DelimiterOperator(delim) => {
                    self.stack.push(StackElement::Delimiter(delim));
                    let un_op = delim.into_delimiter_operator();
                    self.stack.push(StackElement::UnaryOperator(un_op));
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
                self.state = State::ExpectOperator;
                let int = parse_integer(token.as_str()).map_err(|e| ParseError {
                    kind: e.into(),
                    span: token.span(),
                })?;
                self.queue
                    .push_back(token.to_expression(ExpressionKind::Integer(int)));
            }
            TokenKind::Float => {
                self.state = State::ExpectOperator;
                // parse float
                todo!();
            }
            TokenKind::String => {
                self.state = State::ExpectOperator;
                self.queue
                    .push_back(token.to_expression(ExpressionKind::String));
            }
            TokenKind::UnterminatedString => {
                self.state = State::ExpectOperator;
                return Err(ParseError {
                    kind: ParseErrorKind::UnterminatedString,
                    span: token.span(),
                });
            }
        }
        Ok(())
    }

    fn parse_operator(&mut self) -> Result<(), ParseError> {
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
                    Err(ParseError {
                        kind: ParseErrorKind::UnexpectedToken {
                            expected: EXPECT_OPERATOR,
                        },
                        span: token.span(),
                    })
                }
            },
            _ => {
                self.state = State::ExpectTerm;
                Err(ParseError {
                    kind: ParseErrorKind::UnexpectedToken {
                        expected: EXPECT_OPERATOR,
                    },
                    span: token.span(),
                })
            }
        }
    }

    fn process_right_delimiter(
        &mut self,
        right: operator::RightDelimiter<'s>,
    ) -> Result<(), ParseError> {
        self.state = State::ExpectOperator;
        while let Some(op) = self.stack.pop() {
            match op {
                StackElement::Delimiter(left) => {
                    if self.context.match_delimiters(left, right) {
                        return Ok(());
                    } else {
                        return Err(ParseError {
                            kind: ParseErrorKind::MismatchedDelimiter {
                                opening: left.token().span(),
                            },
                            span: right.token().span(),
                        });
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
        Err(ParseError {
            kind: ParseErrorKind::UnmatchedRightDelimiter,
            span: right.token().span(),
        })
    }

    fn process_binary_operator(
        &mut self,
        bin_op: operator::BinaryOperator<'s>,
    ) -> Result<(), ParseError> {
        self.state = State::ExpectTerm;
        self.pop_while_lower_precedence(bin_op.fixity(), bin_op.token().span())?;
        self.stack.push(StackElement::BinaryOperator(bin_op));
        Ok(())
    }

    fn process_postfix_operator(
        &mut self,
        post_op: operator::UnaryOperator<'s>,
    ) -> Result<(), ParseError> {
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
    ) -> Result<(), ParseError> {
        while let Some(prev_op) = self.stack.pop_if_lower_precedence(fixity) {
            match prev_op {
                StackElement::BinaryOperator(prev) => self.queue.push_back(prev.into()),
                StackElement::UnaryOperator(prev) => self.queue.push_back(prev.into()),
                StackElement::Delimiter(_) => {
                    return Err(ParseError {
                        kind: ParseErrorKind::OperatorWithBasePrecedence,
                        span,
                    })
                }
            }
        }
        Ok(())
    }

    fn source(&self) -> &'s str {
        self.tokenizer.source()
    }

    fn end_of_input(&self, expected: &'static str) -> ParseError {
        let len = self.source().len();
        ParseError {
            kind: ParseErrorKind::EndOfInput { expected },
            span: Span {
                start: len,
                end: len,
            },
        }
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
    /// A combination of a delimiter and a unary operator which is applied to the contents of the
    /// delimited group. This allows for different delimiters to have different behavior.
    DelimiterOperator(operator::LeftDelimiter<'s>),
    None,
}

pub enum Postfix<'s> {
    BinaryOperator(operator::BinaryOperator<'s>),
    PostfixOperator(operator::UnaryOperator<'s>),
    LeftDelimiter(operator::LeftDelimiter<'s>),
    RightDelimiter(operator::RightDelimiter<'s>),
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use test_case::test_case;

    use super::{ParseContext, Parser, Postfix, Prefix, EXPECT_OPERATOR, EXPECT_TERM};
    use crate::{
        error::ParseErrorKind,
        operator::{
            BinaryOperator, Fixity, LeftDelimiter, Precedence, RightDelimiter, UnaryOperator,
        },
        token::Token,
    };

    struct SimpleExprContext;

    impl ParseContext for SimpleExprContext {
        fn get_prefix<'s>(&self, token: Token<'s>) -> Prefix<'s> {
            match token.as_str() {
                "-" => Prefix::UnaryOperator(UnaryOperator::new(Precedence::Multiplicative, token)),
                "(" => Prefix::Delimiter(LeftDelimiter::new(token)),
                "[" => Prefix::DelimiterOperator(LeftDelimiter::new(token)),
                _ => Prefix::None,
            }
        }

        fn get_postfix<'s>(&self, token: Token<'s>) -> Postfix<'s> {
            match token.as_str() {
                "," => Postfix::BinaryOperator(BinaryOperator::new(
                    Fixity::Left(Precedence::Comma),
                    token,
                )),
                "+" | "-" => Postfix::BinaryOperator(BinaryOperator::new(
                    Fixity::Left(Precedence::Additive),
                    token,
                )),
                "*" | "/" => Postfix::BinaryOperator(BinaryOperator::new(
                    Fixity::Left(Precedence::Multiplicative),
                    token,
                )),
                "^" => Postfix::BinaryOperator(BinaryOperator::new(
                    Fixity::Right(Precedence::Exponential),
                    token,
                )),
                "!" => Postfix::PostfixOperator(UnaryOperator::new(Precedence::Exponential, token)),
                "(" => Postfix::LeftDelimiter(LeftDelimiter::new(token)),
                ")" | "]" => Postfix::RightDelimiter(RightDelimiter::new(token)),
                _ => Postfix::None,
            }
        }

        fn match_delimiters<'s>(&self, left: LeftDelimiter<'s>, right: RightDelimiter<'s>) -> bool {
            matches!(
                (left.token().as_str(), right.token().as_str()),
                ("(", ")") | ("[", "]")
            )
        }
    }

    #[test_case("3 + 4 * 2 / ( 1 - 5 ) ^ 2 ^ 3", "3 4 2 * 1 5 - 2 3 ^ ^ / +" ; "simple arithmetic" )]
    #[test_case("sin(max(5/2, 3)) / 3 * pi", "sin max 5 2 / 3 , ( ( 3 / pi *" ; "with functions" )]
    #[test_case("2^3!", "2 3 ! ^" ; "postfix operators" )]
    #[test_case("-2^3 + (-2)^3", "2 3 ^ - 2 - 3 ^ +" ; "prefix operators" )]
    #[test_case("[1, (2, 3), 4]", "1 2 3 , , 4 , [" ; "delimiter operators" )]
    fn parse_expression(input: &str, output: &str) -> anyhow::Result<()> {
        let actual = Parser::new(input, SimpleExprContext)
            .parse()?
            .into_iter()
            .map(|expr| expr.token.as_str())
            .collect::<Vec<_>>();
        let expected = output.split_whitespace().collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test_case("1 )", &[(ParseErrorKind::UnmatchedRightDelimiter, 2..3)] ; "unmatched right paren" )]
    #[test_case("1 +", &[(ParseErrorKind::EndOfInput { expected: EXPECT_TERM }, 3..3)] ; "end of input" )]
    #[test_case("(5 5", &[
        (ParseErrorKind::UnexpectedToken { expected: EXPECT_OPERATOR }, 3..4),
        (ParseErrorKind::EndOfInput { expected: EXPECT_TERM }, 4..4),
        (ParseErrorKind::UnmatchedLeftDelimiter, 0..1),
    ] ; "multiple errors")]
    #[test_case("[ 1 )", &[
        (ParseErrorKind::MismatchedDelimiter { opening: (0..1).into() }, 4..5),
    ] ; "mismatched delimiters" )]
    fn parse_expression_fail(input: &str, expected: &[(ParseErrorKind, Range<usize>)]) {
        let actual = Parser::new(input, SimpleExprContext)
            .parse()
            .unwrap_err()
            .errors
            .into_iter()
            .map(|err| (err.kind, err.span.into_range()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
