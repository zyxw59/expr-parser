use crate::{
    error::{ParseError, ParseErrorKind, ParseErrors, ParseErrorsFor},
    expression::{Expression, ExpressionKind},
    operator::Fixity,
    token::{Token, Tokenizer},
    Span,
};

const EXPECT_TERM: &str = "literal, variable, unary operator, or delimiter";
const EXPECT_OPERATOR: &str = "binary operator, delimiter, postfix operator, or end of input";

pub fn parse<T, P>(
    mut tokenizer: T,
    parser: P,
) -> Result<ExpressionQueueFor<P, T>, ParseErrorsFor<P, T>>
where
    P: Parser<T::Token>,
    T: Tokenizer,
{
    let mut state = ParseState::new(parser);
    while let Some(token) = tokenizer.next_token() {
        state.parse_result(token);
    }
    state.finish()
}

/// Parses until a single term has been completed.
///
/// This means zero or more prefix operators followed by either a term token or a delimited
/// group.
pub fn parse_one_term<T, P>(
    mut tokenizer: T,
    parser: P,
) -> Result<ExpressionQueueFor<P, T>, ParseErrorsFor<P, T>>
where
    P: Parser<T::Token>,
    T: Tokenizer,
{
    let mut state = ParseState::new(parser);
    while let Some(token) = tokenizer.next_token() {
        state.parse_result(token);
        if state.has_parsed_term() {
            break;
        }
    }
    state.finish()
}

pub type ExpressionQueueFor<P, T> = Vec<
    Expression<
        <T as Tokenizer>::Position,
        <P as Parser<<T as Tokenizer>::Token>>::BinaryOperator,
        <P as Parser<<T as Tokenizer>::Token>>::UnaryOperator,
        <P as Parser<<T as Tokenizer>::Token>>::Term,
    >,
>;

pub type ExpressionQueue<T, Idx, P> = Vec<
    Expression<
        Idx,
        <P as Parser<T>>::BinaryOperator,
        <P as Parser<T>>::UnaryOperator,
        <P as Parser<T>>::Term,
    >,
>;

pub struct ParseState<T, TokErr, Idx, P: Parser<T>> {
    parser: P,
    end_of_input: Idx,
    state: State,
    stack: Stack<T, Idx, P>,
    first_delimiter_stack_idx: Option<usize>,
    queue: ExpressionQueue<T, Idx, P>,
    errors: Vec<ParseError<P::Error, TokErr, Idx>>,
}

impl<T, TokErr, Idx: Default + Clone, P: Parser<T>> ParseState<T, TokErr, Idx, P> {
    pub fn new(parser: P) -> Self {
        Self {
            parser,
            end_of_input: Default::default(),
            state: State::PostOperator,
            stack: Stack::new(),
            first_delimiter_stack_idx: None,
            queue: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub fn parse_result(&mut self, result: Result<Token<T, Idx>, TokErr>) {
        match result {
            Err(e) => self.errors.push(ParseError {
                span: Span {
                    start: self.end_of_input.clone(),
                    end: self.end_of_input.clone(),
                },
                kind: ParseErrorKind::Tokenizer(e),
            }),
            Ok(token) => {
                self.parse_token(token);
            }
        }
    }

    pub fn parse_token(&mut self, token: Token<T, Idx>) {
        self.end_of_input = token.span.end.clone();
        match self.parser.parse_token(token.kind) {
            Ok(element) => match self.state {
                State::PostOperator => self.parse_term(token.span, element),
                State::PostTerm => self.parse_operator(token.span, element),
            },
            Err(error) => {
                self.errors.push(ParseError {
                    span: token.span,
                    kind: ParseErrorKind::Parser(error),
                });
                // assume that the invalid token was the more expected kind, ideally producing the
                // most useful errors.
                match self.state {
                    State::PostOperator => self.state = State::PostTerm,
                    State::PostTerm => self.state = State::PostOperator,
                }
            }
        }
    }

    /// Returns whether the parser has parsed a single top-level term. This could be any number of
    /// unary operators, followed by a delimited expression or a basic term (literal or variable).
    ///
    /// This method must be called after every call to `parse_result` or `parse_token` in order to
    /// update its state. If it is not called after each of those calls, it may return incorrect
    /// results.
    ///
    /// If parsing continues after this method returns `true`, it is not guaranteed to return
    /// useful information.
    // TODO: can this be reworked so that it always indicates basically, "can we stop here?"
    // TODO: can it be reworked so that it doesn't need to be called constantly to maintain state?
    //       the only thing it needs to update is `self.first_delimiter_stack_idx`; it could
    //       instead just check if there are any delimiters on the stack. Or maybe it's a small
    //       enough overhead to just store that info in the stack itself, so it will never get
    //       out-of-date
    pub fn has_parsed_term(&mut self) -> bool {
        if let Some(idx) = self.first_delimiter_stack_idx {
            // if we've popped that delimiter, we've parsed a term
            self.stack.len() < idx
        } else {
            // we haven't encountered any open delimiters yet
            if let Some(top) = self.stack.peek_top() {
                if top.order.is_delimiter() {
                    // just opened a delimiter; we won't have parsed a term until we close it
                    self.first_delimiter_stack_idx = Some(self.stack.len());
                    false
                } else if self.state == State::PostTerm {
                    // we just parsed a Term, this term is complete
                    true
                } else {
                    false
                }
            } else {
                // nothing on the stack: unless the output is empty, we must have parsed a term
                !self.queue.is_empty()
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn finish(
        mut self,
    ) -> Result<ExpressionQueue<T, Idx, P>, ParseErrors<P::Error, TokErr, Idx>> {
        if self.state != State::PostTerm {
            if let Some(el) = self.stack.pop() {
                if let Some(kind) = el.operator.expression_kind_no_rhs() {
                    self.queue.push(Expression {
                        kind,
                        span: el.span.clone(),
                    });
                } else {
                    self.errors.push(ParseError {
                        kind: ParseErrorKind::EndOfInput {
                            expected: EXPECT_TERM,
                        },
                        span: Span {
                            start: self.end_of_input.clone(),
                            end: self.end_of_input,
                        },
                    })
                }
                if el.order.is_delimiter() {
                    self.errors.push(ParseError {
                        kind: ParseErrorKind::UnmatchedLeftDelimiter,
                        span: el.span,
                    })
                }
            }
        }
        while let Some(el) = self.stack.pop() {
            if let Some(kind) = el.operator.expression_kind_rhs() {
                self.queue.push(Expression {
                    kind,
                    span: el.span.clone(),
                });
            }
            if el.order.is_delimiter() {
                self.errors.push(ParseError {
                    kind: ParseErrorKind::UnmatchedLeftDelimiter,
                    span: el.span,
                })
            }
        }
        if self.errors.is_empty() {
            Ok(self.queue)
        } else {
            Err(self.errors.into())
        }
    }

    fn parse_term(&mut self, span: Span<Idx>, element: ParserElement<P, T>) {
        match element.prefix {
            Prefix::LeftDelimiter {
                delimiter,
                operator,
                empty,
            } => {
                self.stack.push(StackElement {
                    span,
                    order: StackOrder::Delimiter(delimiter),
                    operator: StackOperator::unary_delimiter(operator, empty),
                });
                self.state = State::PostOperator;
            }
            Prefix::RightDelimiter { delimiter } => {
                self.process_right_delimiter(span, delimiter);
            }
            Prefix::UnaryOperator {
                precedence,
                operator,
                no_rhs,
            } => {
                self.stack.push(StackElement {
                    span,
                    order: StackOrder::Precedence(precedence),
                    operator: StackOperator::Unary {
                        unary: operator,
                        term: no_rhs,
                    },
                });
                self.state = State::PostOperator;
            }
            Prefix::Term { term } => {
                self.state = State::PostTerm;
                self.queue.push(Expression {
                    span,
                    kind: ExpressionKind::Term(term),
                });
            }
            Prefix::None => {
                self.state = State::PostTerm;
                if let Some(el) = self.stack.pop() {
                    if let Some(kind) = el.operator.expression_kind_no_rhs() {
                        self.queue.push(Expression {
                            kind,
                            span: el.span,
                        });
                    } else {
                        self.errors.push(ParseError {
                            kind: ParseErrorKind::UnexpectedToken {
                                expected: EXPECT_TERM,
                            },
                            span: span.clone(),
                        });
                    };
                }
                self.parse_operator(span, element);
            }
        }
    }

    fn parse_operator(&mut self, span: Span<Idx>, element: ParserElement<P, T>) {
        match element.postfix {
            Postfix::RightDelimiter { delimiter } => self.process_right_delimiter(span, delimiter),
            Postfix::BinaryOperator {
                fixity,
                operator,
                no_rhs,
            } => {
                self.state = State::PostOperator;
                self.process_binary_operator(span, fixity, operator, no_rhs);
            }
            Postfix::PostfixOperator {
                precedence,
                operator,
            } => self.process_postfix_operator(span, precedence, operator),
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
                    span,
                    order: StackOrder::Delimiter(delimiter),
                    operator: StackOperator::Binary {
                        binary: operator,
                        unary: empty,
                    },
                });
            }
            Postfix::None => {
                self.state = State::PostOperator;
                // TODO(#11) here is where we would support implicit operators
                self.errors.push(ParseError {
                    kind: ParseErrorKind::UnexpectedToken {
                        expected: EXPECT_OPERATOR,
                    },
                    span,
                });
            }
        }
    }

    fn process_right_delimiter(&mut self, span: Span<Idx>, right: P::Delimiter) {
        // If we don't have a right-hand operand, demote the operator on the top of the stack
        // (binary -> unary, unary -> term) if possible. If it is not possible (i.e. that operator
        // requires a right-hand operand), then push an error. We don't early return though, since
        // we still want to find the matching delimiter so that we can continue parsing.
        if self.state != State::PostTerm {
            if let Some(el) = self.stack.pop() {
                if let Some(kind) = el.operator.expression_kind_no_rhs() {
                    self.queue.push(Expression {
                        kind,
                        span: el.span.clone(),
                    });
                } else {
                    self.errors.push(ParseError {
                        kind: ParseErrorKind::UnexpectedToken {
                            expected: EXPECT_TERM,
                        },
                        span: span.clone(),
                    });
                };
                if let StackOrder::Delimiter(left) = el.order {
                    self.state = State::PostTerm;
                    self.check_delimiter_match(left, el.span, right, span);
                    return;
                }
            }
        };
        self.state = State::PostTerm;
        while let Some(el) = self.stack.pop() {
            if let Some(kind) = el.operator.expression_kind_rhs() {
                self.queue.push(Expression {
                    kind,
                    span: el.span.clone(),
                });
            }
            if let StackOrder::Delimiter(left) = el.order {
                self.check_delimiter_match(left, el.span, right, span);
                return;
            }
        }
        self.errors.push(ParseError {
            kind: ParseErrorKind::UnmatchedRightDelimiter,
            span,
        })
    }

    fn check_delimiter_match(
        &mut self,
        left: P::Delimiter,
        left_span: Span<Idx>,
        right: P::Delimiter,
        right_span: Span<Idx>,
    ) {
        if !left.matches(&right) {
            self.errors.push(ParseError {
                kind: ParseErrorKind::MismatchedDelimiter { opening: left_span },
                span: right_span,
            });
        }
    }

    fn process_binary_operator(
        &mut self,
        span: Span<Idx>,
        fixity: Fixity<P::Precedence>,
        binary: P::BinaryOperator,
        unary: Option<P::UnaryOperator>,
    ) {
        self.pop_while_lower_precedence(&fixity);
        self.stack.push(StackElement {
            span,
            order: StackOrder::Precedence(fixity.into_precedence()),
            operator: StackOperator::Binary { binary, unary },
        });
    }

    fn process_postfix_operator(
        &mut self,
        span: Span<Idx>,
        precedence: P::Precedence,
        operator: P::UnaryOperator,
    ) {
        self.state = State::PostTerm;
        let fixity = Fixity::Right(precedence);
        self.pop_while_lower_precedence(&fixity);
        self.queue.push(Expression {
            span,
            kind: ExpressionKind::UnaryOperator(operator),
        });
    }

    fn pop_while_lower_precedence(&mut self, fixity: &Fixity<P::Precedence>) {
        while let Some(el) = self.stack.pop_if_lower_precedence(fixity) {
            if let Some(kind) = el.operator.expression_kind_rhs() {
                self.queue.push(Expression {
                    kind,
                    span: el.span,
                });
            }
        }
    }
}

impl<T, TokErr, Idx: Default + Clone, P: Parser<T>> Extend<Token<T, Idx>>
    for ParseState<T, TokErr, Idx, P>
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Token<T, Idx>>,
    {
        iter.into_iter().for_each(|tok| self.parse_token(tok))
    }
}

impl<T, TokErr, Idx: Default + Clone, P: Parser<T>> Extend<Result<Token<T, Idx>, TokErr>>
    for ParseState<T, TokErr, Idx, P>
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Result<Token<T, Idx>, TokErr>>,
    {
        iter.into_iter().for_each(|res| self.parse_result(res))
    }
}

pub trait Parser<T> {
    type Precedence: Ord;
    type Delimiter: Delimiter;
    type BinaryOperator;
    type UnaryOperator;
    type Term;
    type Error;

    fn parse_token(&self, kind: T) -> Result<ParserElement<Self, T>, Self::Error>;
}

impl<P, T> Parser<T> for &'_ P
where
    P: Parser<T> + ?Sized,
{
    type Precedence = P::Precedence;
    type Delimiter = P::Delimiter;
    type BinaryOperator = P::BinaryOperator;
    type UnaryOperator = P::UnaryOperator;
    type Term = P::Term;
    type Error = P::Error;

    fn parse_token(&self, kind: T) -> Result<ParserElement<Self, T>, Self::Error> {
        P::parse_token(self, kind)
    }
}

pub trait Delimiter {
    fn matches(&self, other: &Self) -> bool;
}

pub struct Element<P, D, B, U, T> {
    pub prefix: Prefix<P, D, U, T>,
    pub postfix: Postfix<P, D, B, U>,
}

pub type ParserElement<P, T> = Element<
    <P as Parser<T>>::Precedence,
    <P as Parser<T>>::Delimiter,
    <P as Parser<T>>::BinaryOperator,
    <P as Parser<T>>::UnaryOperator,
    <P as Parser<T>>::Term,
>;

pub enum Prefix<P, D, U, T> {
    UnaryOperator {
        precedence: P,
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

pub enum Postfix<P, D, B, U> {
    BinaryOperator {
        fixity: Fixity<P>,
        operator: B,
        no_rhs: Option<U>,
    },
    PostfixOperator {
        precedence: P,
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

pub enum StackOrder<P, D> {
    Precedence(P),
    Delimiter(D),
}

impl<P, D> StackOrder<P, D> {
    fn precedence(&self) -> Option<&P> {
        match self {
            Self::Precedence(p) => Some(p),
            Self::Delimiter(_) => None,
        }
    }

    fn is_delimiter(&self) -> bool {
        matches!(self, Self::Delimiter(_))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    PostOperator,
    PostTerm,
}

struct Stack<T, Idx, P: Parser<T>> {
    stack: Vec<StackElement<T, Idx, P>>,
}

impl<T, Idx, P: Parser<T>> Default for Stack<T, Idx, P> {
    fn default() -> Self {
        Stack {
            stack: Default::default(),
        }
    }
}

impl<T, Idx, P: Parser<T>> Stack<T, Idx, P> {
    fn new() -> Self {
        Default::default()
    }

    fn push(&mut self, element: StackElement<T, Idx, P>) {
        self.stack.push(element);
    }

    fn pop(&mut self) -> Option<StackElement<T, Idx, P>> {
        self.stack.pop()
    }

    fn peek_top(&self) -> Option<&StackElement<T, Idx, P>> {
        self.stack.last()
    }

    fn len(&self) -> usize {
        self.stack.len()
    }

    /// Pops the stack if the new operator has lower precedence than the top of the stack
    fn pop_if_lower_precedence(
        &mut self,
        fixity: &Fixity<P::Precedence>,
    ) -> Option<StackElement<T, Idx, P>> {
        if match fixity {
            Fixity::Left(prec) => Some(prec) <= self.precedence(),
            Fixity::Right(prec) => Some(prec) < self.precedence(),
        } {
            self.pop()
        } else {
            None
        }
    }

    fn precedence(&self) -> Option<&P::Precedence> {
        self.peek_top().and_then(StackElement::precedence)
    }
}

struct StackElement<T, Idx, P: Parser<T>> {
    span: Span<Idx>,
    order: StackOrder<P::Precedence, P::Delimiter>,
    operator: StackOperator<P::BinaryOperator, P::UnaryOperator, P::Term>,
}

impl<T, Idx, P: Parser<T>> StackElement<T, Idx, P> {
    fn precedence(&self) -> Option<&P::Precedence> {
        self.order.precedence()
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

#[cfg(test)]
mod tests {
    #![allow(clippy::type_complexity)]

    use std::{convert::Infallible, ops::Range};

    use test_case::test_case;

    use super::{
        parse, parse_one_term, Delimiter, Element, Parser, Postfix, Prefix, EXPECT_OPERATOR,
        EXPECT_TERM,
    };
    use crate::{
        error::ParseErrorKind,
        expression::{Expression, ExpressionKind},
        operator::Fixity,
        token::{SimpleCharSetTokenKind, SimpleTokenizer, StrSource},
    };

    struct SimpleExprContext;

    #[derive(Clone, Copy, Eq, PartialEq)]
    enum SimpleDelimiter {
        Paren,
        SquareBracket,
        Pipe,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
    enum SimplePrecedence {
        /// Comma
        Comma,
        /// Additive operators, such as `+` and `-`
        Additive,
        /// Multiplicative operators, such as `*` and `/`, as well as unary minus.
        Multiplicative,
        /// Exponential operators, such as `^` and `!`
        Exponential,
    }

    impl Delimiter for SimpleDelimiter {
        fn matches(&self, other: &Self) -> bool {
            self == other
        }
    }

    impl<'s> Parser<(&'s str, SimpleCharSetTokenKind)> for SimpleExprContext {
        type Error = Infallible;
        type Precedence = SimplePrecedence;
        type Delimiter = SimpleDelimiter;
        type BinaryOperator = &'s str;
        type UnaryOperator = &'s str;
        type Term = &'s str;

        fn parse_token(
            &self,
            (s, _kind): (&'s str, SimpleCharSetTokenKind),
        ) -> Result<
            Element<
                Self::Precedence,
                Self::Delimiter,
                Self::BinaryOperator,
                Self::UnaryOperator,
                Self::Term,
            >,
            Self::Error,
        > {
            Ok(match s {
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
                        fixity: Fixity::Left(SimplePrecedence::Comma),
                        operator: s,
                        no_rhs: Some("(,)"),
                    },
                },
                "-" => Element {
                    prefix: Prefix::UnaryOperator {
                        precedence: SimplePrecedence::Multiplicative,
                        operator: s,
                        no_rhs: None,
                    },
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Left(SimplePrecedence::Additive),
                        operator: s,
                        no_rhs: None,
                    },
                },
                "+" => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Left(SimplePrecedence::Additive),
                        operator: s,
                        no_rhs: None,
                    },
                },
                "*" | "/" => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Left(SimplePrecedence::Multiplicative),
                        operator: s,
                        no_rhs: None,
                    },
                },
                "^" => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::BinaryOperator {
                        fixity: Fixity::Right(SimplePrecedence::Exponential),
                        operator: s,
                        no_rhs: None,
                    },
                },
                "!" => Element {
                    prefix: Prefix::None,
                    postfix: Postfix::PostfixOperator {
                        precedence: SimplePrecedence::Exponential,
                        operator: s,
                    },
                },
                _ => Element {
                    prefix: Prefix::Term { term: s },
                    postfix: Postfix::None,
                },
            })
        }
    }

    fn expr_to_str<'s, Idx>(expr: Expression<Idx, &'s str, &'s str, &'s str>) -> &'s str {
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
    #[test_case("a, * b", "a (,) b *" ; "trailing comma with binary operator" )]
    fn parse_expression(input: &str, output: &str) -> anyhow::Result<()> {
        let actual = parse(
            SimpleTokenizer::new(StrSource::new(input)),
            SimpleExprContext,
        )?
        .into_iter()
        .map(expr_to_str)
        .collect::<Vec<_>>();
        let expected = output.split_whitespace().collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test_case("3", "3", "" ; "single term" )]
    #[test_case("-3!", "3 -", "!" ; "unary operators" )]
    #[test_case("3!", "3", "!" ; "postfix operator" )]
    #[test_case("-3 a", "3 -", "a" ; "unary operators with additional" )]
    #[test_case("(5 + 4) * (3 - 2)", "5 4 +", "* (3 - 2)" ; "delimited group" )]
    #[test_case("(3)!", "3", "!" ; "delimited with unary operators" )]
    #[test_case("abc def)", "abc", "def)" ; "ignores invalid after first term" )]
    fn parse_one(input: &str, output: &str, rest: &str) -> anyhow::Result<()> {
        let mut tokens = SimpleTokenizer::new(StrSource::new(input));
        let actual = parse_one_term(&mut tokens, SimpleExprContext)?
            .into_iter()
            .map(expr_to_str)
            .collect::<Vec<_>>();
        let expected = output.split_whitespace().collect::<Vec<_>>();
        assert_eq!(actual, expected);
        let actual_rest = tokens
            .map(|res| res.map(|tok| tok.kind))
            .collect::<Result<Vec<_>, _>>()?;
        let expected_rest = SimpleTokenizer::new(StrSource::new(rest))
            .map(|res| res.map(|tok| tok.kind))
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(actual_rest, expected_rest);
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
    #[test_case("( [ 1 )", &[
        (ParseErrorKind::MismatchedDelimiter { opening: (2..3).into() }, 6..7),
        (ParseErrorKind::UnmatchedLeftDelimiter, 0..1),
    ] ; "mismatched delimiters 2" )]
    #[test_case("[ 1 + )", &[
        (ParseErrorKind::UnexpectedToken { expected: EXPECT_TERM }, 6..7),
        (ParseErrorKind::MismatchedDelimiter { opening: (0..1).into() }, 6..7),
    ] ; "mismatched delimiters with missing rhs" )]
    #[test_case("1 + * 2", &[
        (ParseErrorKind::UnexpectedToken { expected: EXPECT_TERM }, 4..5),
    ] ; "extra operator" )]
    fn parse_expression_fail(
        input: &str,
        expected: &[(ParseErrorKind<Infallible, Infallible, usize>, Range<usize>)],
    ) {
        let actual = parse(
            SimpleTokenizer::new(StrSource::new(input)),
            SimpleExprContext,
        )
        .unwrap_err()
        .errors
        .into_iter()
        .map(|err| {
            (
                err.kind.map_tokenizer_error(|_| unreachable!()),
                err.span.into_range(),
            )
        })
        .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
