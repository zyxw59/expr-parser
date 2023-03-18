use std::{borrow::Cow, collections::VecDeque};

use crate::{
    error::{ParseError, ParseErrorKind, ParseErrors, ParseFloatError, ParseIntError},
    expression::{Expression, ExpressionKind},
    operator::{Fixity, Precedence},
    token::{Token, TokenKind, Tokenizer},
    Span,
};

const EXPECT_TERM: &str = "literal, variable, unary operator, or delimiter";
const EXPECT_OPERATOR: &str = "binary operator, delimiter, postfix operator, or end of input";

pub type ExpressionQueue<'s, C> = VecDeque<
    Expression<'s, <C as ParseContext<'s>>::BinaryOperator, <C as ParseContext<'s>>::UnaryOperator>,
>;

pub struct Parser<'s, C: ParseContext<'s>> {
    tokenizer: Tokenizer<'s>,
    context: C,
    state: State,
    stack: Stack<'s, C::Delimiter, C::BinaryOperator, C::UnaryOperator>,
    queue: ExpressionQueue<'s, C>,
}

impl<'s, C> Parser<'s, C>
where
    C: ParseContext<'s>,
{
    pub fn new(source: &'s str, context: C) -> Self {
        Parser {
            tokenizer: Tokenizer::new(source),
            context,
            state: State::Initial,
            stack: Stack::new(),
            queue: VecDeque::new(),
        }
    }

    pub fn parse(mut self) -> Result<ExpressionQueue<'s, C>, ParseErrors> {
        let mut errors = Vec::new();
        while !self.tokenizer.is_empty() {
            if let Err(err) = self.parse_next() {
                errors.push(err);
            }
        }
        match self.state {
            State::Initial => self.queue.push_back(Expression {
                token: Token::new(self.end_of_input_span(), self.source()),
                kind: ExpressionKind::Null,
            }),
            State::PostOperator => errors.push(self.end_of_input(EXPECT_TERM)),
            State::PostTerm => {}
        }
        while let Some(el) = self.stack.pop() {
            if let Some(Err(err)) = self.push_stack_element_to_queue(el, |token, _delim| {
                Err(ParseError {
                    kind: ParseErrorKind::UnmatchedLeftDelimiter,
                    span: token.span(),
                })
            }) {
                errors.push(err);
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
            State::Initial | State::PostOperator => self.parse_term(),
            State::PostTerm => self.parse_operator(),
        }
    }

    fn parse_term(&mut self) -> Result<(), ParseError> {
        let Some((token, kind)) = self.tokenizer.next_token() else {
            return Ok(());
        };
        match kind {
            TokenKind::Tag => match self.context.get_prefix(token) {
                Prefix::LeftDelimiter {
                    delimiter,
                    operator,
                } => {
                    self.stack.push(StackElement {
                        token,
                        precedence: Precedence::Base,
                        delimiter: Some(delimiter),
                        operator: StackOperator::from_unary_option(operator),
                    });
                    self.state = State::Initial;
                }
                Prefix::RightDelimiter { delimiter } => {
                    if self.state == State::Initial {
                        self.queue.push_back(Expression {
                            token,
                            kind: ExpressionKind::Null,
                        });
                        self.process_right_delimiter(token, delimiter)?;
                    } else {
                        return Err(ParseError {
                            kind: ParseErrorKind::UnexpectedToken {
                                expected: EXPECT_TERM,
                            },
                            span: token.span(),
                        });
                    }
                }
                Prefix::UnaryOperator {
                    precedence,
                    operator,
                } => {
                    self.stack.push(StackElement {
                        token,
                        precedence,
                        delimiter: None,
                        operator: StackOperator::Unary(operator),
                    });
                    self.state = State::PostOperator;
                }
                Prefix::None => {
                    self.state = State::PostTerm;
                    self.queue.push_back(Expression {
                        token,
                        kind: ExpressionKind::Variable,
                    });
                }
            },
            TokenKind::Integer => {
                self.state = State::PostTerm;
                let int = parse_integer(token.as_str()).map_err(|e| ParseError {
                    kind: e.into(),
                    span: token.span(),
                })?;
                self.queue.push_back(Expression {
                    token,
                    kind: ExpressionKind::Integer(int),
                });
            }
            TokenKind::Float => {
                self.state = State::PostTerm;
                let float = parse_float(token.as_str()).map_err(|e| ParseError {
                    kind: e.into(),
                    span: token.span(),
                })?;
                self.queue.push_back(Expression {
                    token,
                    kind: ExpressionKind::Float(float),
                });
            }
            TokenKind::String => {
                self.state = State::PostTerm;
                self.queue.push_back(Expression {
                    token,
                    kind: ExpressionKind::String,
                });
            }
            TokenKind::UnterminatedString => {
                self.state = State::PostTerm;
                return Err(ParseError {
                    kind: ParseErrorKind::UnterminatedString,
                    span: token.span(),
                });
            }
        }
        Ok(())
    }

    fn parse_operator(&mut self) -> Result<(), ParseError> {
        let Some((token, kind)) = self.tokenizer.next_token() else {
            return Ok(());
        };
        match kind {
            TokenKind::Tag => match self.context.get_postfix(token) {
                Postfix::RightDelimiter { delimiter } => {
                    self.process_right_delimiter(token, delimiter)
                }
                Postfix::BinaryOperator { fixity, operator } => {
                    self.state = State::PostOperator;
                    self.process_binary_operator(token, fixity, operator)
                }
                Postfix::MixedOperator { fixity, operator } => {
                    self.state = State::Initial;
                    self.process_binary_operator(token, fixity, operator)
                }
                Postfix::PostfixOperator {
                    precedence,
                    operator,
                } => self.process_postfix_operator(token, precedence, operator),
                Postfix::LeftDelimiter {
                    delimiter,
                    operator,
                } => {
                    self.state = State::Initial;
                    // left delimiter in operator position indicates a function call or similar.
                    // this is indicated by adding a binary operator (with the same token as the
                    // delimiter) to the stack immediately after the delimiter itself. this
                    // operator will then function as the "function application" operator (or a
                    // related operator, such as "struct construction") when it is popped from the
                    // stack after the closing delimiter is matched
                    self.stack.push(StackElement {
                        token,
                        precedence: Precedence::Base,
                        delimiter: Some(delimiter),
                        operator: StackOperator::Unary(operator),
                    });
                    Ok(())
                }
                Postfix::None => {
                    self.state = State::PostOperator;
                    Err(ParseError {
                        kind: ParseErrorKind::UnexpectedToken {
                            expected: EXPECT_OPERATOR,
                        },
                        span: token.span(),
                    })
                }
            },
            _ => {
                self.state = State::PostOperator;
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
        token: Token<'s>,
        right: C::Delimiter,
    ) -> Result<(), ParseError> {
        self.state = State::PostTerm;
        while let Some(el) = self.stack.pop() {
            if let Some(result) = self.push_stack_element_to_queue(el, |left_token, left| {
                if left.matches(&right) {
                    Ok(())
                } else {
                    Err(ParseError {
                        kind: ParseErrorKind::MismatchedDelimiter {
                            opening: left_token.span(),
                        },
                        span: token.span(),
                    })
                }
            }) {
                return result;
            }
        }
        Err(ParseError {
            kind: ParseErrorKind::UnmatchedRightDelimiter,
            span: token.span(),
        })
    }

    fn process_binary_operator(
        &mut self,
        token: Token<'s>,
        fixity: Fixity,
        operator: C::BinaryOperator,
    ) -> Result<(), ParseError> {
        self.pop_while_lower_precedence(fixity, token.span())?;
        self.stack.push(StackElement {
            token,
            precedence: fixity.precedence(),
            delimiter: None,
            operator: StackOperator::Binary(operator),
        });
        Ok(())
    }

    fn process_postfix_operator(
        &mut self,
        token: Token<'s>,
        precedence: Precedence,
        operator: C::UnaryOperator,
    ) -> Result<(), ParseError> {
        self.state = State::PostTerm;
        let fixity = Fixity::Right(precedence);
        self.pop_while_lower_precedence(fixity, token.span())?;
        self.queue.push_back(Expression {
            token,
            kind: ExpressionKind::UnaryOperator(operator),
        });
        Ok(())
    }

    fn pop_while_lower_precedence(&mut self, fixity: Fixity, span: Span) -> Result<(), ParseError> {
        while let Some(el) = self.stack.pop_if_lower_precedence(fixity) {
            self.push_stack_element_to_queue(el, |_, _| {
                Err(ParseError {
                    kind: ParseErrorKind::OperatorWithBasePrecedence,
                    span,
                })
            })
            .transpose()?;
        }
        Ok(())
    }

    /// Push the operator from the stack element onto the queue. If the stack element has a
    /// delimiter, the provided function will be called.
    fn push_stack_element_to_queue(
        &mut self,
        el: StackElement<'s, C::Delimiter, C::BinaryOperator, C::UnaryOperator>,
        delimiter_predicate: impl FnOnce(Token<'s>, C::Delimiter) -> Result<(), ParseError>,
    ) -> Option<Result<(), ParseError>> {
        match el.operator {
            StackOperator::Unary(op) => self.queue.push_back(Expression {
                token: el.token,
                kind: ExpressionKind::UnaryOperator(op),
            }),
            StackOperator::Binary(op) => self.queue.push_back(Expression {
                token: el.token,
                kind: ExpressionKind::BinaryOperator(op),
            }),
            StackOperator::None => {}
        };
        el.delimiter
            .map(|delim| delimiter_predicate(el.token, delim))
    }

    fn source(&self) -> &'s str {
        self.tokenizer.source()
    }

    fn end_of_input_span(&self) -> Span {
        let len = self.source().len();
        Span {
            start: len,
            end: len,
        }
    }

    fn end_of_input(&self, expected: &'static str) -> ParseError {
        ParseError {
            kind: ParseErrorKind::EndOfInput { expected },
            span: self.end_of_input_span(),
        }
    }
}

pub trait ParseContext<'s> {
    type Delimiter: Delimiter;
    type BinaryOperator;
    type UnaryOperator;

    fn get_prefix(&self, token: Token<'s>) -> Prefix<Self::Delimiter, Self::UnaryOperator>;

    fn get_postfix(
        &self,
        token: Token<'s>,
    ) -> Postfix<Self::Delimiter, Self::BinaryOperator, Self::UnaryOperator>;
}

pub trait Delimiter {
    fn matches(&self, other: &Self) -> bool;
}

pub enum Prefix<D, U> {
    UnaryOperator { precedence: Precedence, operator: U },
    LeftDelimiter { delimiter: D, operator: Option<U> },
    RightDelimiter { delimiter: D },
    None,
}

pub enum Postfix<D, B, U> {
    BinaryOperator {
        fixity: Fixity,
        operator: B,
    },
    /// A mixed operator is a binary operator whose right-hand side is optional.
    MixedOperator {
        fixity: Fixity,
        operator: B,
    },
    PostfixOperator {
        precedence: Precedence,
        operator: U,
    },
    LeftDelimiter {
        delimiter: D,
        operator: U,
    },
    RightDelimiter {
        delimiter: D,
    },
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Initial,
    PostOperator,
    PostTerm,
}

#[derive(Clone, Debug)]
struct Stack<'s, D, B, U>(Vec<StackElement<'s, D, B, U>>);

impl<'s, D, B, U> Default for Stack<'s, D, B, U> {
    fn default() -> Self {
        Stack(Default::default())
    }
}

impl<'s, D, B, U> Stack<'s, D, B, U> {
    fn new() -> Self {
        Default::default()
    }

    fn push(&mut self, element: StackElement<'s, D, B, U>) {
        self.0.push(element);
    }

    fn pop(&mut self) -> Option<StackElement<'s, D, B, U>> {
        self.0.pop()
    }

    /// Pops the stack if the new operator has lower precedence than the top of the stack
    fn pop_if_lower_precedence(&mut self, fixity: Fixity) -> Option<StackElement<'s, D, B, U>> {
        if match fixity {
            Fixity::Left(prec) => prec <= self.precedence(),
            Fixity::Right(prec) => prec < self.precedence(),
        } {
            self.pop()
        } else {
            None
        }
    }

    fn precedence(&self) -> Precedence {
        self.0
            .last()
            .map(StackElement::precedence)
            .unwrap_or(Precedence::Base)
    }
}

#[derive(Clone, Copy, Debug)]
struct StackElement<'s, D, B, U> {
    token: Token<'s>,
    precedence: Precedence,
    delimiter: Option<D>,
    operator: StackOperator<B, U>,
}

impl<'s, D, B, U> StackElement<'s, D, B, U> {
    fn precedence(&self) -> Precedence {
        self.precedence
    }
}

#[derive(Clone, Copy, Debug)]
enum StackOperator<B, U> {
    None,
    Binary(B),
    Unary(U),
}

impl<B, U> StackOperator<B, U> {
    fn from_unary_option(option: Option<U>) -> Self {
        match option {
            None => Self::None,
            Some(op) => Self::Unary(op),
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

fn parse_float(s: &str) -> Result<f64, ParseFloatError> {
    let mut s = Cow::Borrowed(s);
    if s.contains('_') {
        // float parsing is really hard, and writing our own float parser to do this in a zero-copy
        // way is not worth it.
        s.to_mut().retain(|c| c != '_');
    }
    if s.is_empty() {
        return Err(ParseFloatError::Empty);
    }
    s.parse().map_err(|_| ParseFloatError::Invalid)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use test_case::test_case;

    use super::{Delimiter, ParseContext, Parser, Postfix, Prefix, EXPECT_OPERATOR, EXPECT_TERM};
    use crate::{
        error::ParseErrorKind,
        operator::{Fixity, Precedence},
        token::Token,
    };

    struct SimpleExprContext;

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

    impl<'s> ParseContext<'s> for SimpleExprContext {
        type Delimiter = SimpleDelimiter;
        type BinaryOperator = &'s str;
        type UnaryOperator = &'s str;

        fn get_prefix(&self, token: Token<'s>) -> Prefix<Self::Delimiter, Self::UnaryOperator> {
            let s = token.as_str();
            match s {
                "-" => Prefix::UnaryOperator {
                    precedence: Precedence::Multiplicative,
                    operator: s,
                },
                "(" => Prefix::LeftDelimiter {
                    delimiter: SimpleDelimiter::Paren,
                    operator: None,
                },
                "[" => Prefix::LeftDelimiter {
                    delimiter: SimpleDelimiter::SquareBracket,
                    operator: Some(s),
                },
                ")" => Prefix::RightDelimiter {
                    delimiter: SimpleDelimiter::Paren,
                },
                "]" => Prefix::RightDelimiter {
                    delimiter: SimpleDelimiter::SquareBracket,
                },
                _ => Prefix::None,
            }
        }

        fn get_postfix(
            &self,
            token: Token<'s>,
        ) -> Postfix<Self::Delimiter, Self::BinaryOperator, Self::UnaryOperator> {
            let s = token.as_str();
            match s {
                "," => Postfix::MixedOperator {
                    fixity: Fixity::Left(Precedence::Comma),
                    operator: s,
                },
                "+" | "-" => Postfix::BinaryOperator {
                    fixity: Fixity::Left(Precedence::Additive),
                    operator: s,
                },
                "*" | "/" => Postfix::BinaryOperator {
                    fixity: Fixity::Left(Precedence::Multiplicative),
                    operator: s,
                },
                "^" => Postfix::BinaryOperator {
                    fixity: Fixity::Right(Precedence::Exponential),
                    operator: s,
                },
                "!" => Postfix::PostfixOperator {
                    precedence: Precedence::Exponential,
                    operator: s,
                },
                "(" => Postfix::LeftDelimiter {
                    delimiter: SimpleDelimiter::Paren,
                    operator: s,
                },
                ")" => Postfix::RightDelimiter {
                    delimiter: SimpleDelimiter::Paren,
                },
                "]" => Postfix::RightDelimiter {
                    delimiter: SimpleDelimiter::SquareBracket,
                },
                _ => Postfix::None,
            }
        }
    }

    #[test_case("3 + 4 * 2 / ( 1 - 5 ) ^ 2 ^ 3", "3 4 2 * 1 5 - 2 3 ^ ^ / +" ; "simple arithmetic" )]
    #[test_case("sin(max(5/2, 3)) / 3 * pi", "sin max 5 2 / 3 , ( ( 3 / pi *" ; "with functions" )]
    #[test_case("2^3!", "2 3 ! ^" ; "postfix operators" )]
    #[test_case("-2^3 + (-2)^3", "2 3 ^ - 2 - 3 ^ +" ; "prefix operators" )]
    #[test_case("[1, 2, 3, 4]", "1 2 , 3 , 4 , [" ; "delimiter operators" )]
    #[test_case("[1, (2, 3), 4]", "1 2 3 , , 4 , [" ; "nested delimiter operators" )]
    #[test_case("[ ]", "] [" ; "empty list" )]
    #[test_case("f()", "f ) (" ; "empty function call" )]
    #[test_case("[1, 2, 3, 4, ]", "1 2 , 3 , 4 , ] , [" ; "trailing comma" )]
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
