use std::{borrow::Cow, collections::VecDeque};

use crate::{
    error::{ParseError, ParseErrorKind, ParseErrors, ParseFloatError, ParseIntError},
    expression::{Expression, ExpressionKind},
    operator::{Fixity, Precedence},
    token::Token,
    Span,
};

const EXPECT_TERM: &str = "literal, variable, unary operator, or delimiter";
const EXPECT_OPERATOR: &str = "binary operator, delimiter, postfix operator, or end of input";

pub type ExpressionQueue<'s, C, T> = VecDeque<
    Expression<
        's,
        <C as ParseContext<'s, T>>::BinaryOperator,
        <C as ParseContext<'s, T>>::UnaryOperator,
        <C as ParseContext<'s, T>>::Term,
    >,
>;

pub struct Parser<'s, I, T, C: ParseContext<'s, T>> {
    tokenizer: I,
    context: C,
    state: State,
    stack: Stack<'s, C::Delimiter, C::BinaryOperator, C::UnaryOperator, C::Term>,
    queue: ExpressionQueue<'s, C, T>,
}

impl<'s, I, T, C> Parser<'s, I, T, C>
where
    C: ParseContext<'s, T>,
    I: Iterator<Item = (Token<'s>, T)>,
{
    pub fn new(tokenizer: I, context: C) -> Self {
        Parser {
            tokenizer,
            context,
            state: State::PostOperator,
            stack: Stack::new(),
            queue: VecDeque::new(),
        }
    }

    pub fn parse(mut self) -> Result<ExpressionQueue<'s, C, T>, ParseErrors<C::Error>> {
        let mut errors = Vec::new();
        let mut end_of_input = 0;
        while let Some((token, kind)) = self.tokenizer.next() {
            end_of_input = token.span().end;
            if let Err(err) = self.parse_next(token, kind) {
                errors.push(err);
            }
        }
        if self.state != State::PostTerm {
            if let Some(el) = self.stack.pop() {
                if let Some(kind) = el.operator.expression_kind_no_rhs() {
                    self.queue.push_back(Expression {
                        kind,
                        token: el.token,
                    });
                } else {
                    errors.push(ParseError {
                        kind: ParseErrorKind::EndOfInput {
                            expected: EXPECT_TERM,
                        },
                        span: Span {
                            start: end_of_input,
                            end: end_of_input,
                        },
                    })
                }
                if el.delimiter.is_some() {
                    errors.push(ParseError {
                        kind: ParseErrorKind::UnmatchedLeftDelimiter,
                        span: el.token.span(),
                    })
                }
            }
        }
        while let Some(el) = self.stack.pop() {
            if let Some(kind) = el.operator.expression_kind_rhs() {
                self.queue.push_back(Expression {
                    kind,
                    token: el.token,
                });
            }
            if el.delimiter.is_some() {
                errors.push(ParseError {
                    kind: ParseErrorKind::UnmatchedLeftDelimiter,
                    span: el.token.span(),
                })
            }
        }
        if errors.is_empty() {
            Ok(self.queue)
        } else {
            Err(errors.into())
        }
    }

    fn parse_next(&mut self, token: Token<'s>, kind: T) -> Result<(), ParseError<C::Error>> {
        match self.state {
            State::PostOperator => self.parse_term(token, kind),
            State::PostTerm => self.parse_operator(token, kind),
        }
    }

    fn parse_term(&mut self, token: Token<'s>, kind: T) -> Result<(), ParseError<C::Error>> {
        match self.context.parse_token(token, kind).prefix {
            Prefix::LeftDelimiter {
                delimiter,
                operator,
                empty,
            } => {
                self.stack.push(StackElement {
                    token,
                    precedence: Precedence::Base,
                    delimiter: Some(delimiter),
                    operator: StackOperator::unary_delimiter(operator, empty),
                });
                self.state = State::PostOperator;
            }
            Prefix::RightDelimiter { delimiter } => {
                self.process_right_delimiter(token, delimiter)?;
            }
            Prefix::UnaryOperator {
                precedence,
                operator,
                no_rhs,
            } => {
                self.stack.push(StackElement {
                    token,
                    precedence,
                    delimiter: None,
                    operator: StackOperator::Unary {
                        unary: operator,
                        term: no_rhs,
                    },
                });
                self.state = State::PostOperator;
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

    fn parse_operator(&mut self, token: Token<'s>, kind: T) -> Result<(), ParseError<C::Error>> {
        match self.context.parse_token(token, kind).postfix {
            Postfix::RightDelimiter { delimiter } => self.process_right_delimiter(token, delimiter),
            Postfix::BinaryOperator {
                fixity,
                operator,
                no_rhs,
            } => {
                self.state = State::PostOperator;
                self.process_binary_operator(token, fixity, operator, no_rhs)
            }
            Postfix::PostfixOperator {
                precedence,
                operator,
            } => self.process_postfix_operator(token, precedence, operator),
            Postfix::LeftDelimiter {
                delimiter,
                operator,
                empty,
            } => {
                self.state = State::PostOperator;
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
                    operator: StackOperator::Binary {
                        binary: operator,
                        unary: empty,
                    },
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
        // If we don't have a right-hand operand, demote the operator on the top of the stack
        // (binary -> unary, unary -> term) if possible. If it is not possible (i.e. that operator
        // requires a right-hand operand), then construct an error. We don't early return with this
        // error, though, since we still want to find the matching delimiter so that we can
        // continue parsing.
        let no_rhs_result = if self.state != State::PostTerm {
            if let Some(el) = self.stack.pop() {
                let no_rhs_result = if let Some(kind) = el.operator.expression_kind_no_rhs() {
                    self.queue.push_back(Expression {
                        kind,
                        token: el.token,
                    });
                    Ok(())
                } else {
                    Err(ParseError {
                        kind: ParseErrorKind::UnexpectedToken {
                            expected: EXPECT_TERM,
                        },
                        span: token.span(),
                    })
                };
                if let Some(left) = el.delimiter {
                    self.state = State::PostTerm;
                    return Self::check_delimiter_match(
                        no_rhs_result,
                        left,
                        el.token,
                        right,
                        token,
                    );
                }
                no_rhs_result
            } else {
                Ok(())
            }
        } else {
            Ok(())
        };
        self.state = State::PostTerm;
        while let Some(el) = self.stack.pop() {
            if let Some(kind) = el.operator.expression_kind_rhs() {
                self.queue.push_back(Expression {
                    kind,
                    token: el.token,
                });
            }
            if let Some(left) = el.delimiter {
                return Self::check_delimiter_match(no_rhs_result, left, el.token, right, token);
            }
        }
        Err(ParseError {
            kind: ParseErrorKind::UnmatchedRightDelimiter,
            span: token.span(),
        })
    }

    fn check_delimiter_match(
        no_rhs_result: Result<(), ParseError<C::Error>>,
        left: C::Delimiter,
        left_token: Token<'s>,
        right: C::Delimiter,
        right_token: Token<'s>,
    ) -> Result<(), ParseError<C::Error>> {
        if left.matches(&right) {
            no_rhs_result
        } else {
            // Ideally we would return both errors here, but that would be hard to work
            // into the normal `Result` paradigm. So instead, we just return the
            // "unexpected token" error if it exists, or the "mismatched delimiter" error
            // if it doesn't
            no_rhs_result.and_then(|()| {
                Err(ParseError {
                    kind: ParseErrorKind::MismatchedDelimiter {
                        opening: left_token.span(),
                    },
                    span: right_token.span(),
                })
            })
        }
    }

    fn process_binary_operator(
        &mut self,
        token: Token<'s>,
        fixity: Fixity,
        binary: C::BinaryOperator,
        unary: Option<C::UnaryOperator>,
    ) -> Result<(), ParseError<C::Error>> {
        self.pop_while_lower_precedence(fixity, token.span())?;
        self.stack.push(StackElement {
            token,
            precedence: fixity.precedence(),
            delimiter: None,
            operator: StackOperator::Binary { binary, unary },
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
            if let Some(kind) = el.operator.expression_kind_rhs() {
                self.queue.push_back(Expression {
                    kind,
                    token: el.token,
                });
            }
            if el.delimiter.is_some() {
                return Err(ParseError {
                    kind: ParseErrorKind::OperatorWithBasePrecedence,
                    span,
                });
            }
        }
        Ok(())
    }
}

pub trait ParseContext<'s, T> {
    type Delimiter: Delimiter;
    type BinaryOperator;
    type UnaryOperator;
    type Term;
    type Error;

    fn parse_token(
        &self,
        token: Token<'s>,
        kind: T,
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
        no_rhs: Option<T>,
    },
    LeftDelimiter {
        delimiter: D,
        operator: Option<U>,
        empty: Option<T>,
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
        no_rhs: Option<U>,
    },
    PostfixOperator {
        precedence: Precedence,
        operator: U,
    },
    LeftDelimiter {
        delimiter: D,
        operator: B,
        empty: Option<U>,
    },
    RightDelimiter {
        delimiter: D,
    },
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    PostOperator,
    PostTerm,
}

#[derive(Clone, Debug)]
struct Stack<'s, D, B, U, T>(Vec<StackElement<'s, D, B, U, T>>);

impl<'s, D, B, U, T> Default for Stack<'s, D, B, U, T> {
    fn default() -> Self {
        Stack(Default::default())
    }
}

impl<'s, D, B, U, T> Stack<'s, D, B, U, T> {
    fn new() -> Self {
        Default::default()
    }

    fn push(&mut self, element: StackElement<'s, D, B, U, T>) {
        self.0.push(element);
    }

    fn pop(&mut self) -> Option<StackElement<'s, D, B, U, T>> {
        self.0.pop()
    }

    /// Pops the stack if the new operator has lower precedence than the top of the stack
    fn pop_if_lower_precedence(&mut self, fixity: Fixity) -> Option<StackElement<'s, D, B, U, T>> {
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
struct StackElement<'s, D, B, U, T> {
    token: Token<'s>,
    precedence: Precedence,
    delimiter: Option<D>,
    operator: StackOperator<B, U, T>,
}

impl<'s, D, B, U, T> StackElement<'s, D, B, U, T> {
    fn precedence(&self) -> Precedence {
        self.precedence
    }
}

#[derive(Clone, Copy, Debug)]
enum StackOperator<B, U, T> {
    None { term: Option<T> },
    Binary { binary: B, unary: Option<U> },
    Unary { unary: U, term: Option<T> },
}

impl<B, U, T> StackOperator<B, U, T> {
    fn unary_delimiter(unary: Option<U>, term: Option<T>) -> Self {
        match unary {
            None => Self::None { term },
            Some(unary) => Self::Unary { unary, term },
        }
    }

    fn expression_kind_rhs(self) -> Option<ExpressionKind<B, U, T>> {
        match self {
            Self::None { .. } => None,
            Self::Binary { binary, .. } => Some(ExpressionKind::BinaryOperator(binary)),
            Self::Unary { unary, .. } => Some(ExpressionKind::UnaryOperator(unary)),
        }
    }

    fn expression_kind_no_rhs(self) -> Option<ExpressionKind<B, U, T>> {
        match self {
            Self::None { term } | Self::Unary { term, .. } => term.map(ExpressionKind::Term),
            Self::Binary { unary, .. } => unary.map(ExpressionKind::UnaryOperator),
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
        expression::{Expression, ExpressionKind},
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

    impl<'s> ParseContext<'s, SimpleCharSetTokenKind> for SimpleExprContext {
        type Error = SimpleParserError;
        type Delimiter = SimpleDelimiter;
        type BinaryOperator = &'s str;
        type UnaryOperator = &'s str;
        type Term = &'s str;

        fn parse_token(
            &self,
            token: Token<'s>,
            _kind: SimpleCharSetTokenKind,
        ) -> Element<Self::Delimiter, Self::BinaryOperator, Self::UnaryOperator, Self::Term>
        {
            let s = token.as_str();
            match s {
                "(" => Element {
                    prefix: Prefix::LeftDelimiter {
                        delimiter: SimpleDelimiter::Paren,
                        operator: None,
                        empty: None,
                    },
                    postfix: Postfix::LeftDelimiter {
                        delimiter: SimpleDelimiter::Paren,
                        operator: s,
                        empty: Some("()"),
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
                        empty: Some("[]"),
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
                        empty: None,
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
                        no_rhs: Some("(,)"),
                    },
                },
                "-" => Element {
                    prefix: Prefix::UnaryOperator {
                        precedence: Precedence::Multiplicative,
                        operator: s,
                        no_rhs: None,
                    },
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Additive),
                        operator: s,
                        no_rhs: None,
                    },
                },
                "+" => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Additive),
                        operator: s,
                        no_rhs: None,
                    },
                },
                "*" | "/" => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Left(Precedence::Multiplicative),
                        operator: s,
                        no_rhs: None,
                    },
                },
                "^" => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Right(Precedence::Exponential),
                        operator: s,
                        no_rhs: None,
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

    fn expr_to_str<'s>(expr: Expression<'s, &'s str, &'s str, &'s str>) -> &'s str {
        match expr.kind {
            ExpressionKind::BinaryOperator(s) => s,
            ExpressionKind::UnaryOperator(s) => s,
            ExpressionKind::Term(s) => s,
        }
    }

    #[test_case("3 + 4 * 2 / ( 1 - 5 ) ^ 2 ^ 3", "3 4 2 * 1 5 - 2 3 ^ ^ / +" ; "simple arithmetic" )]
    #[test_case("sin(max(5/2, 3)) / 3 * pi", "sin max 5 2 / 3 , ( ( 3 / pi *" ; "with functions" )]
    #[test_case("2^3!", "2 3 ! ^" ; "postfix operators" )]
    #[test_case("-2^3 + (-2)^3", "2 3 ^ - 2 - 3 ^ +" ; "prefix operators" )]
    #[test_case("[1, 2, 3, 4]", "1 2 , 3 , 4 , [" ; "delimiter operators" )]
    #[test_case("[1, (2, 3), 4]", "1 2 3 , , 4 , [" ; "nested delimiter operators" )]
    #[test_case("[ ]", "[]" ; "empty list" )]
    #[test_case("[ ] + [ ]", "[] [] +" ; "adding lists" )]
    #[test_case("f()", "f ()" ; "empty function call" )]
    #[test_case("[1, 2, 3, 4, ]", "1 2 , 3 , 4 , (,) [" ; "trailing comma" )]
    #[test_case("a * |b|", "a b | *" ; "absolute value" )]
    fn parse_expression(input: &str, output: &str) -> anyhow::Result<()> {
        let actual = Parser::new(SimpleTokenizer::new(input), SimpleExprContext)
            .parse()?
            .into_iter()
            .map(expr_to_str)
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
