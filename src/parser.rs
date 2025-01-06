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

pub fn parse_with_backstop<T, P>(
    mut tokenizer: T,
    parser: P,
    backstop: Backstop<T::Position, P::Precedence, P::Delimiter>,
) -> Result<ExpressionQueueFor<P, T>, ParseErrorsFor<P, T>>
where
    P: Parser<T::Token>,
    T: Tokenizer,
{
    let mut state = ParseState::new(parser);
    state.stack.backstop = Some(StackElement {
        span: backstop.span,
        kind: backstop.kind,
        operator: StackOperator::None { term: None },
    });
    while let Some(token) = tokenizer.next_token() {
        state.parse_result(token);
        if state.stack.backstop.is_none() {
            break;
        }
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
    let mut delimiter_stack_index = None;
    while let Some(token) = tokenizer.next_token() {
        state.parse_result(token);
        if let Some(top) = state.stack.peek_top() {
            if top.kind.is_delimiter() {
                delimiter_stack_index = Some(state.stack.len());
                break;
            }
            if state.state == State::PostTerm {
                break;
            }
        } else {
            break;
        }
    }
    if let Some(delimiter_stack_index) = delimiter_stack_index {
        while let Some(token) = tokenizer.next_token() {
            state.parse_result(token);
            if state.stack.len() < delimiter_stack_index {
                break;
            }
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

pub type ExpressionQueue<Tok, Pos, Par> = Vec<
    Expression<
        Pos,
        <Par as Parser<Tok>>::BinaryOperator,
        <Par as Parser<Tok>>::UnaryOperator,
        <Par as Parser<Tok>>::Term,
    >,
>;

struct ParseState<Tok, TokErr, Pos, Par: Parser<Tok>> {
    parser: Par,
    end_of_input: Pos,
    state: State,
    stack: Stack<Tok, Pos, Par>,
    queue: ExpressionQueue<Tok, Pos, Par>,
    errors: Vec<ParseError<Par::Error, TokErr, Pos>>,
}

impl<Tok, TokErr, Pos: Default + Clone, Par: Parser<Tok>> ParseState<Tok, TokErr, Pos, Par> {
    fn new(parser: Par) -> Self {
        Self {
            parser,
            end_of_input: Default::default(),
            state: State::PostOperator,
            stack: Stack::new(),
            queue: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub fn parse_result(&mut self, result: Result<Token<Tok, Pos>, TokErr>) {
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

    pub fn parse_token(&mut self, token: Token<Tok, Pos>) {
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

    #[allow(clippy::type_complexity)]
    pub fn finish(
        mut self,
    ) -> Result<ExpressionQueue<Tok, Pos, Par>, ParseErrors<Par::Error, TokErr, Pos>> {
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
                if el.kind.is_delimiter() {
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
            if el.kind.is_delimiter() {
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

    fn parse_term(&mut self, span: Span<Pos>, element: ParserElement<Par, Tok>) {
        match element.prefix {
            Prefix::LeftDelimiter {
                delimiter,
                operator,
                empty,
            } => {
                self.stack.push(StackElement {
                    span,
                    kind: BackstopKind::Delimiter(delimiter),
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
                    kind: BackstopKind::Precedence(precedence),
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

    fn parse_operator(&mut self, span: Span<Pos>, element: ParserElement<Par, Tok>) {
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
                    kind: BackstopKind::Delimiter(delimiter),
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

    fn process_right_delimiter(&mut self, span: Span<Pos>, right: Par::Delimiter) {
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
                if let BackstopKind::Delimiter(left) = el.kind {
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
            if let BackstopKind::Delimiter(left) = el.kind {
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
        left: Par::Delimiter,
        left_span: Span<Pos>,
        right: Par::Delimiter,
        right_span: Span<Pos>,
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
        span: Span<Pos>,
        fixity: Fixity<Par::Precedence>,
        binary: Par::BinaryOperator,
        unary: Option<Par::UnaryOperator>,
    ) {
        if self.pop_while_lower_precedence(&fixity) {
            // backstop was hit
            return;
        }
        self.stack.push(StackElement {
            span,
            kind: BackstopKind::Precedence(fixity.into_precedence()),
            operator: StackOperator::Binary { binary, unary },
        });
    }

    fn process_postfix_operator(
        &mut self,
        span: Span<Pos>,
        precedence: Par::Precedence,
        operator: Par::UnaryOperator,
    ) {
        self.state = State::PostTerm;
        let fixity = Fixity::Right(precedence);
        if self.pop_while_lower_precedence(&fixity) {
            // backstop was hit
            return;
        }
        self.queue.push(Expression {
            span,
            kind: ExpressionKind::UnaryOperator(operator),
        });
    }

    // returns whether the backstop was popped
    fn pop_while_lower_precedence(&mut self, fixity: &Fixity<Par::Precedence>) -> bool {
        let has_backstop = self.stack.backstop.is_some();
        while let Some(el) = self.stack.pop_if_lower_precedence(fixity) {
            if let Some(kind) = el.operator.expression_kind_rhs() {
                self.queue.push(Expression {
                    kind,
                    span: el.span,
                });
            }
        }
        has_backstop && self.stack.backstop.is_none()
    }
}

impl<Tok, TokErr, Pos: Default + Clone, Par: Parser<Tok>> Extend<Token<Tok, Pos>>
    for ParseState<Tok, TokErr, Pos, Par>
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Token<Tok, Pos>>,
    {
        iter.into_iter().for_each(|tok| self.parse_token(tok))
    }
}

impl<Tok, TokErr, Pos: Default + Clone, Par: Parser<Tok>> Extend<Result<Token<Tok, Pos>, TokErr>>
    for ParseState<Tok, TokErr, Pos, Par>
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Result<Token<Tok, Pos>, TokErr>>,
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

pub struct Backstop<Idx, P, D> {
    pub span: Span<Idx>,
    pub kind: BackstopKind<P, D>,
}

pub enum BackstopKind<P, D> {
    Precedence(P),
    Delimiter(D),
}

impl<P, D> BackstopKind<P, D> {
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

struct Stack<Tok, Pos, Par: Parser<Tok>> {
    stack: Vec<StackElement<Tok, Pos, Par>>,
    backstop: Option<StackElement<Tok, Pos, Par>>,
}

impl<Tok, Pos, Par: Parser<Tok>> Default for Stack<Tok, Pos, Par> {
    fn default() -> Self {
        Stack {
            stack: Default::default(),
            backstop: None,
        }
    }
}

impl<Tok, Pos, Par: Parser<Tok>> Stack<Tok, Pos, Par> {
    fn new() -> Self {
        Default::default()
    }

    fn push(&mut self, element: StackElement<Tok, Pos, Par>) {
        self.stack.push(element);
    }

    fn pop(&mut self) -> Option<StackElement<Tok, Pos, Par>> {
        self.stack.pop().or_else(|| self.backstop.take())
    }

    fn peek_top(&self) -> Option<&StackElement<Tok, Pos, Par>> {
        self.stack.last().or(self.backstop.as_ref())
    }

    fn len(&self) -> usize {
        self.stack.len()
    }

    /// Pops the stack if the new operator has lower precedence than the top of the stack
    fn pop_if_lower_precedence(
        &mut self,
        fixity: &Fixity<Par::Precedence>,
    ) -> Option<StackElement<Tok, Pos, Par>> {
        if match fixity {
            Fixity::Left(prec) => Some(prec) <= self.precedence(),
            Fixity::Right(prec) => Some(prec) < self.precedence(),
        } {
            self.pop()
        } else {
            None
        }
    }

    fn precedence(&self) -> Option<&Par::Precedence> {
        self.peek_top().and_then(StackElement::precedence)
    }
}

struct StackElement<Tok, Pos, Par: Parser<Tok>> {
    span: Span<Pos>,
    kind: BackstopKind<Par::Precedence, Par::Delimiter>,
    operator: StackOperator<Par::BinaryOperator, Par::UnaryOperator, Par::Term>,
}

impl<Tok, Pos, Par: Parser<Tok>> StackElement<Tok, Pos, Par> {
    fn precedence(&self) -> Option<&Par::Precedence> {
        self.kind.precedence()
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
        parse, parse_one_term, parse_with_backstop, Backstop, BackstopKind, Delimiter, Element,
        Parser, Postfix, Prefix, EXPECT_OPERATOR, EXPECT_TERM,
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

    #[test_case("1 * (3 + 4)) rest", BackstopKind::Delimiter(SimpleDelimiter::Paren), "1 3 4 + *", "rest" ; "delimited")]
    #[test_case("1 * (3, 4), rest", BackstopKind::Precedence(SimplePrecedence::Comma), "1 3 4 , *", "rest" ; "precedence")]
    #[test_case("1 * 3 + 4", BackstopKind::Precedence(SimplePrecedence::Additive), "1 3 *", "4" ; "precedence without rhs")]
    fn parse_backstop(
        input: &str,
        backstop: BackstopKind<SimplePrecedence, SimpleDelimiter>,
        output: &str,
        rest: &str,
    ) -> anyhow::Result<()> {
        let mut tokens = SimpleTokenizer::new(StrSource::new(input));
        let backstop = Backstop {
            kind: backstop,
            span: Default::default(),
        };
        let actual = parse_with_backstop(&mut tokens, SimpleExprContext, backstop)?
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
