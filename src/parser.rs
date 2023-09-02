use std::{borrow::Cow, collections::VecDeque};

use crate::{
    error::{ParseError, ParseErrorKind, ParseErrors, ParseFloatError, ParseIntError},
    expression::{Expression, ExpressionKind},
    operator::{Fixity, Precedence},
    token::{Token, Tokenizer},
    Span,
};

const EXPECT_TERM: &str = "literal, variable, unary operator, or delimiter";
const EXPECT_OPERATOR: &str = "binary operator, delimiter, postfix operator, or end of input";

pub type ExpressionQueue<'s, C> = VecDeque<
    Expression<
        's,
        <C as ParseContext<'s>>::BinaryOperator,
        <C as ParseContext<'s>>::UnaryOperator,
        <C as ParseContext<'s>>::Term,
    >,
>;

pub struct Parser<'s, T, C: ParseContext<'s>> {
    tokenizer: T,
    context: C,
    state: State,
    stack: Stack<'s, C::Delimiter, C::BinaryOperator, C::UnaryOperator>,
    queue: ExpressionQueue<'s, C>,
}

impl<'s, T, C> Parser<'s, T, C>
where
    C: ParseContext<'s, TokenKind = T::TokenKind>,
    T: Tokenizer<'s>,
{
    pub fn new(tokenizer: T, context: C) -> Self {
        Parser {
            tokenizer,
            context,
            state: State::Initial,
            stack: Stack::new(),
            queue: VecDeque::new(),
        }
    }

    pub fn parse(mut self) -> Result<ExpressionQueue<'s, C>, ParseErrors<C::Error>> {
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

    fn parse_next(&mut self) -> Result<(), ParseError<C::Error>> {
        match self.state {
            State::Initial | State::PostOperator => self.parse_term(),
            State::PostTerm => self.parse_operator(),
        }
    }

    fn parse_term(&mut self) -> Result<(), ParseError<C::Error>> {
        let Some((token, kind)) = self.tokenizer.next_token() else {
            return Ok(());
        };
        match self.context.parse_token(token, kind).prefix {
            Prefix::LeftDelimiter {
                delimiter,
                operator,
                rhs_required,
            } => {
                self.stack.push(StackElement {
                    token,
                    precedence: Precedence::Base,
                    delimiter: Some(delimiter),
                    operator: StackOperator::from_unary_option(operator),
                });
                self.state = State::post_operator(rhs_required);
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
                rhs_required,
            } => {
                self.stack.push(StackElement {
                    token,
                    precedence,
                    delimiter: None,
                    operator: StackOperator::Unary(operator),
                });
                self.state = State::post_operator(rhs_required);
            }
            Prefix::Term { term } => {
                self.state = State::PostTerm;
                self.queue.push_back(Expression {
                    token,
                    kind: ExpressionKind::Term(term),
                });
            }
            Prefix::None => {
                self.state = State::PostTerm;
                return Err(ParseError {
                    kind: ParseErrorKind::UnexpectedToken {
                        expected: EXPECT_TERM,
                    },
                    span: token.span(),
                });
            }
        };
        /*
        match kind {
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
        */
        Ok(())
    }

    fn parse_operator(&mut self) -> Result<(), ParseError<C::Error>> {
        let Some((token, kind)) = self.tokenizer.next_token() else {
            return Ok(());
        };
        match self.context.parse_token(token, kind).postfix {
            Postfix::RightDelimiter { delimiter } => self.process_right_delimiter(token, delimiter),
            Postfix::BinaryOperator {
                fixity,
                operator,
                rhs_required,
            } => {
                self.state = State::post_operator(rhs_required);
                self.process_binary_operator(token, fixity, operator)
            }
            Postfix::PostfixOperator {
                precedence,
                operator,
            } => self.process_postfix_operator(token, precedence, operator),
            Postfix::LeftDelimiter {
                delimiter,
                operator,
                rhs_required,
            } => {
                self.state = State::post_operator(rhs_required);
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
                    operator: StackOperator::Binary(operator),
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
        }
        /*
         match kind {
            TokenKind::Tag => ,
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
        */
    }

    fn process_right_delimiter(
        &mut self,
        token: Token<'s>,
        right: C::Delimiter,
    ) -> Result<(), ParseError<C::Error>> {
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
    ) -> Result<(), ParseError<C::Error>> {
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
    ) -> Result<(), ParseError<C::Error>> {
        self.state = State::PostTerm;
        let fixity = Fixity::Right(precedence);
        self.pop_while_lower_precedence(fixity, token.span())?;
        self.queue.push_back(Expression {
            token,
            kind: ExpressionKind::UnaryOperator(operator),
        });
        Ok(())
    }

    fn pop_while_lower_precedence(
        &mut self,
        fixity: Fixity,
        span: Span,
    ) -> Result<(), ParseError<C::Error>> {
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
        delimiter_predicate: impl FnOnce(Token<'s>, C::Delimiter) -> Result<(), ParseError<C::Error>>,
    ) -> Option<Result<(), ParseError<C::Error>>> {
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

    fn end_of_input(&self, expected: &'static str) -> ParseError<C::Error> {
        ParseError {
            kind: ParseErrorKind::EndOfInput { expected },
            span: self.end_of_input_span(),
        }
    }
}

pub trait ParseContext<'s> {
    type TokenKind;
    type Delimiter: Delimiter;
    type BinaryOperator;
    type UnaryOperator;
    type Term;
    type Error;

    fn parse_token(
        &self,
        token: Token<'s>,
        kind: Self::TokenKind,
    ) -> Element<Self::Delimiter, Self::BinaryOperator, Self::UnaryOperator, Self::Term>;
}

pub trait Delimiter {
    fn matches(&self, other: &Self) -> bool;
}

pub struct Element<D, B, U, T> {
    pub prefix: Prefix<D, U, T>,
    pub postfix: Postfix<D, B, U>,
}

pub enum Prefix<D, U, T> {
    UnaryOperator {
        precedence: Precedence,
        operator: U,
        rhs_required: bool,
    },
    LeftDelimiter {
        delimiter: D,
        operator: Option<U>,
        rhs_required: bool,
    },
    RightDelimiter {
        delimiter: D,
    },
    Term {
        term: T,
    },
    None,
}

pub enum Postfix<D, B, U> {
    BinaryOperator {
        fixity: Fixity,
        operator: B,
        rhs_required: bool,
    },
    PostfixOperator {
        precedence: Precedence,
        operator: U,
    },
    LeftDelimiter {
        delimiter: D,
        operator: B,
        rhs_required: bool,
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

impl State {
    /// Returns the next state after an operator or similar. If the right-hand-side is required,
    /// the next state will be `PostOperator`, and if it is not required, the next state will be
    /// `Initial`
    fn post_operator(rhs_required: bool) -> Self {
        if rhs_required {
            Self::PostOperator
        } else {
            Self::Initial
        }
    }
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

    use super::{
        Delimiter, Element, ParseContext, Parser, Postfix, Prefix, EXPECT_OPERATOR, EXPECT_TERM,
    };
    use crate::{
        error::ParseErrorKind,
        operator::{Fixity, Precedence},
        token::{SimpleCharSetTokenKind, SimpleTokenizer, Token},
    };

    struct SimpleExprContext;

    #[derive(Debug, Eq, PartialEq, thiserror::Error)]
    enum SimpleParserError {}

    #[derive(Clone, Copy, Eq, PartialEq)]
    enum SimpleDelimiter {
        Paren,
        SquareBracket,
        Pipe,
    }

    impl Delimiter for SimpleDelimiter {
        fn matches(&self, other: &Self) -> bool {
            self == other
        }
    }

    impl<'s> ParseContext<'s> for SimpleExprContext {
        type Error = SimpleParserError;
        type TokenKind = SimpleCharSetTokenKind;
        type Delimiter = SimpleDelimiter;
        type BinaryOperator = &'s str;
        type UnaryOperator = &'s str;
        type Term = &'s str;

        fn parse_token(
            &self,
            token: Token<'s>,
            kind: Self::TokenKind,
        ) -> Element<Self::Delimiter, Self::BinaryOperator, Self::UnaryOperator, Self::Term>
        {
            let s = token.as_str();
            match s {
                "(" => Element {
                    prefix: Prefix::LeftDelimiter {
                        delimiter: SimpleDelimiter::Paren,
                        operator: None,
                        rhs_required: true,
                    },
                    postfix: Postfix::LeftDelimiter {
                        delimiter: SimpleDelimiter::Paren,
                        operator: s,
                        rhs_required: false,
                    },
                },
                ")" => Element {
                    prefix: Prefix::RightDelimiter {
                        delimiter: SimpleDelimiter::Paren,
                    },
                    postfix: Postfix::RightDelimiter {
                        delimiter: SimpleDelimiter::Paren,
                    },
                },
                "[" => Element {
                    prefix: Prefix::LeftDelimiter {
                        delimiter: SimpleDelimiter::SquareBracket,
                        operator: Some(s),
                        rhs_required: false,
                    },
                    postfix: Postfix::None,
                },
                "]" => Element {
                    prefix: Prefix::RightDelimiter {
                        delimiter: SimpleDelimiter::SquareBracket,
                    },
                    postfix: Postfix::RightDelimiter {
                        delimiter: SimpleDelimiter::SquareBracket,
                    },
                },
                "|" => Element {
                    prefix: Prefix::LeftDelimiter {
                        delimiter: SimpleDelimiter::Pipe,
                        operator: Some(s),
                        rhs_required: true,
                    },
                    postfix: Postfix::RightDelimiter {
                        delimiter: SimpleDelimiter::Pipe,
                    },
                },
                "," => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Comma),
                        operator: s,
                        rhs_required: false,
                    },
                },
                "-" => Element {
                    prefix: Prefix::UnaryOperator {
                        precedence: Precedence::Multiplicative,
                        operator: s,
                        rhs_required: true,
                    },
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Additive),
                        operator: s,
                        rhs_required: true,
                    },
                },
                "+" => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Additive),
                        operator: s,
                        rhs_required: true,
                    },
                },
                "*" | "/" => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Multiplicative),
                        operator: s,
                        rhs_required: true,
                    },
                },
                "^" => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Right(Precedence::Exponential),
                        operator: s,
                        rhs_required: true,
                    },
                },
                "!" => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::PostfixOperator {
                        precedence: Precedence::Exponential,
                        operator: s,
                    },
                },
                _ => Element {
                    prefix: Prefix::Term { term: s },
                    postfix: Postfix::None,
                },
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
    #[test_case("a * |b|", "a b | *" ; "absolute value" )]
    fn parse_expression(input: &str, output: &str) -> anyhow::Result<()> {
        let actual = Parser::new(SimpleTokenizer::new(input), SimpleExprContext)
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
    fn parse_expression_fail(
        input: &str,
        expected: &[(ParseErrorKind<SimpleParserError>, Range<usize>)],
    ) {
        let actual = Parser::new(SimpleTokenizer::new(input), SimpleExprContext)
            .parse()
            .unwrap_err()
            .errors
            .into_iter()
            .map(|err| (err.kind, err.span.into_range()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
