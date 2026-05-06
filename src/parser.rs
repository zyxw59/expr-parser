use crate::{
    error::{ParseError, ParseErrorKind, ParseErrors, ParseErrorsFor},
    expression::{Expression, ExpressionKind},
    token::{Token, Tokenizer},
    Span,
};

const EXPECT_TERM: &str = "literal, variable, unary operator, or delimiter";
const EXPECT_OPERATOR: &str = "binary operator, delimiter, postfix operator, or end of input";

pub fn parse<T, P, Q>(tokenizer: T, parser: P) -> Result<Q, ParseErrorsFor<T>>
where
    P: Parser<T::Token>,
    T: Tokenizer,
    Q: Default + Extend<Expression<T::Position, P::BinaryOperator, P::UnaryOperator, P::Term>>,
{
    parse_into(tokenizer, parser, Q::default())
}

pub fn parse_into<T, P, Q>(mut tokenizer: T, parser: P, output: Q) -> Result<Q, ParseErrorsFor<T>>
where
    P: Parser<T::Token>,
    T: Tokenizer,
    Q: Extend<Expression<T::Position, P::BinaryOperator, P::UnaryOperator, P::Term>>,
{
    let mut state = ParseState::with_output(parser, output);
    while let Some(token) = tokenizer.next_token() {
        state.parse_result(token);
    }
    state.finish()
}

/// Parses until a single term has been completed.
///
/// This means zero or more prefix operators followed by either a term token or a delimited
/// group.
pub fn parse_one_term<T, P, Q>(tokenizer: T, parser: P) -> Result<Q, ParseErrorsFor<T>>
where
    P: Parser<T::Token>,
    T: Tokenizer,
    Q: Default + Extend<Expression<T::Position, P::BinaryOperator, P::UnaryOperator, P::Term>>,
{
    parse_one_term_into(tokenizer, parser, Q::default())
}

/// Parses until a single term has been completed.
///
/// This means zero or more prefix operators followed by either a term token or a delimited
/// group.
pub fn parse_one_term_into<T, P, Q>(
    mut tokenizer: T,
    parser: P,
    output: Q,
) -> Result<Q, ParseErrorsFor<T>>
where
    P: Parser<T::Token>,
    T: Tokenizer,
    Q: Extend<Expression<T::Position, P::BinaryOperator, P::UnaryOperator, P::Term>>,
{
    let mut state = ParseState::with_output(parser, output);
    while let Some(token) = tokenizer.next_token() {
        state.parse_result(token);
        if state.has_parsed_expression() {
            break;
        }
    }
    state.finish()
}

pub struct ParseState<T, TokErr, Idx, P: Parser<T>, Q> {
    parser: P,
    end_of_input: Idx,
    state: State,
    stack: Stack<T, Idx, P>,
    queue: Q,
    errors: Vec<ParseError<TokErr, Idx>>,
}

impl<T, TokErr, Idx, P, Q> ParseState<T, TokErr, Idx, P, Q>
where
    Idx: Default + Clone,
    P: Parser<T>,
    Q: Default + Extend<Expression<Idx, P::BinaryOperator, P::UnaryOperator, P::Term>>,
{
    pub fn new(parser: P) -> Self {
        Self::with_output(parser, Q::default())
    }
}

impl<T, TokErr, Idx, P, Q> ParseState<T, TokErr, Idx, P, Q>
where
    Idx: Default + Clone,
    P: Parser<T>,
    Q: Extend<Expression<Idx, P::BinaryOperator, P::UnaryOperator, P::Term>>,
{
    pub fn with_output(parser: P, output: Q) -> Self {
        Self {
            parser,
            end_of_input: Default::default(),
            state: State::Leader,
            stack: Stack::new(),
            queue: output,
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
        let element = self.parser.parse_token(token.kind);
        match (self.state, element) {
            (
                State::Leader,
                Element {
                    leader: Some(Leader { post_binding, .. }),
                    ..
                },
            ) => {
                self.parse_leader(token.span.clone(), post_binding);
            }
            (
                State::Trailer | State::Leader,
                Element {
                    trailer: Some(trailer),
                    ..
                },
            ) => {
                self.parse_trailer(token.span.clone(), trailer);
            }
            (
                State::Trailer,
                Element {
                    trailer: None,
                    leader:
                        Some(Leader {
                            implicit_operator: Some(implicit_operator),
                            post_binding,
                        }),
                },
            ) => {
                self.handle_implicit_operator(token.span.clone(), implicit_operator);
                self.parse_leader(token.span, post_binding);
            }
            (
                State::Leader,
                Element {
                    leader: None,
                    trailer: None,
                },
            ) => {
                self.errors.push(ParseError {
                    span: token.span,
                    kind: ParseErrorKind::UnexpectedToken {
                        expected: EXPECT_TERM,
                    },
                });
                self.state = State::Trailer;
            }
            (State::Trailer, Element { trailer: None, .. }) => {
                self.errors.push(ParseError {
                    span: token.span,
                    kind: ParseErrorKind::UnexpectedToken {
                        expected: EXPECT_OPERATOR,
                    },
                });
                self.state = State::Leader;
            }
        }
    }

    /// Returns whether the parser has parsed a complete expression.
    pub fn has_parsed_expression(&mut self) -> bool {
        // no delimiters on the stack
        !self.stack.has_delimiter()
            && (
                // in trailer state or the top of the stack can go without a following argument
                self.state == State::Trailer
                    || self
                        .stack
                        .peek_top()
                        .is_some_and(|top| top.operator.can_have_no_follower())
            )
    }

    /// Returns whether the parser has encountered at least one error
    pub fn has_error(&mut self) -> bool {
        !self.errors.is_empty()
    }

    pub fn finish(mut self) -> Result<Q, ParseErrors<TokErr, Idx>> {
        if self.state != State::Trailer {
            if let Some(el) = self.stack.pop() {
                match el.operator.expression_kind_no_follower() {
                    Some(kind) => self.push_expression(Expression {
                        kind,
                        span: el.span.clone(),
                    }),
                    None => self.errors.push(ParseError {
                        kind: ParseErrorKind::EndOfInput {
                            expected: EXPECT_TERM,
                        },
                        span: Span {
                            start: self.end_of_input.clone(),
                            end: self.end_of_input.clone(),
                        },
                    }),
                }
                if el.binding.is_delimiter() {
                    self.errors.push(ParseError {
                        kind: ParseErrorKind::UnmatchedOpeningDelimiter,
                        span: el.span,
                    })
                }
            }
        }
        while let Some(el) = self.stack.pop() {
            let kind = el.operator.expression_kind_follower();
            self.push_expression(Expression {
                kind,
                span: el.span.clone(),
            });

            if el.binding.is_delimiter() {
                self.errors.push(ParseError {
                    kind: ParseErrorKind::UnmatchedOpeningDelimiter,
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

    fn push_expression(
        &mut self,
        expr: Expression<Idx, P::BinaryOperator, P::UnaryOperator, P::Term>,
    ) {
        if self.errors.is_empty() {
            self.queue.extend(Some(expr));
        }
    }

    fn parse_leader(
        &mut self,
        span: Span<Idx>,
        post_binding: PostBinding<P::Precedence, P::Delimiter, P::UnaryOperator, P::Term>,
    ) {
        match post_binding {
            PostBinding::Terminal(term) => {
                self.push_expression(Expression {
                    span,
                    kind: ExpressionKind::Term(term),
                });
                self.state = State::Trailer;
            }
            PostBinding::Nonterminal {
                terminal,
                binding,
                nonterminal,
            } => {
                self.stack.push(StackElement {
                    span,
                    binding,
                    operator: StackOperator::Unary {
                        unary: nonterminal,
                        term: terminal,
                    },
                });
                self.state = State::Leader;
            }
        }
    }

    fn parse_trailer(&mut self, span: Span<Idx>, trailer: ParserTrailer<P, T>) {
        match trailer.pre_binding {
            Binding::Delimiter(delimiter) => {
                let mut delimiter = Some(delimiter);
                self.handle_missing_follower(span.clone(), &mut delimiter);
                if let Some(delimiter) = delimiter {
                    self.process_close_delimiter(span.clone(), delimiter);
                }
            }
            Binding::Precedence(precedence) => {
                self.handle_missing_follower(span.clone(), &mut None);
                self.pop_while_lower_precedence(&precedence);
            }
        }
        match trailer.post_binding {
            PostBinding::Terminal(operator) => {
                self.push_expression(Expression {
                    span,
                    kind: ExpressionKind::UnaryOperator(operator),
                });
                self.state = State::Trailer;
            }
            PostBinding::Nonterminal {
                terminal,
                binding,
                nonterminal: binary,
            } => {
                self.stack.push(StackElement {
                    span,
                    binding,
                    operator: StackOperator::Binary {
                        binary,
                        unary: terminal,
                    },
                });
                self.state = State::Leader;
            }
        }
    }

    fn handle_implicit_operator(
        &mut self,
        span: Span<Idx>,
        ImplicitOperator {
            pre_binding,
            operator,
            post_binding,
        }: ImplicitOperator<P::Precedence, P::BinaryOperator>,
    ) {
        self.pop_while_lower_precedence(&pre_binding);
        self.stack.push(StackElement {
            span,
            binding: Binding::Precedence(post_binding),
            operator: StackOperator::Binary {
                binary: operator,
                unary: None,
            },
        })
    }

    fn handle_missing_follower(&mut self, span: Span<Idx>, delimiter: &mut Option<P::Delimiter>) {
        if self.state == State::Trailer {
            return;
        }
        if self.get_missing_follower(span.clone(), delimiter).is_none() {
            self.errors.push(ParseError {
                kind: ParseErrorKind::UnexpectedToken {
                    expected: EXPECT_TERM,
                },
                span,
            });
        }
    }

    fn get_missing_follower(
        &mut self,
        span: Span<Idx>,
        delimiter: &mut Option<P::Delimiter>,
    ) -> Option<()> {
        let el = self.stack.pop()?;

        if let Binding::Delimiter(open) = el.binding {
            if let Some(close) = delimiter.take() {
                self.check_delimiter_match(open, el.span.clone(), close, span);
            } else {
                // put the opening delimiter back on the stack so that it can match (or fail to match)
                // later.
                let el = StackElement {
                    binding: Binding::Delimiter(open),
                    ..el
                };
                self.stack.push(el);
                return None;
            }
        }

        let kind = el.operator.expression_kind_no_follower()?;
        self.push_expression(Expression {
            kind,
            span: el.span,
        });

        Some(())
    }

    fn process_close_delimiter(&mut self, span: Span<Idx>, close: P::Delimiter) {
        self.state = State::Trailer;
        while let Some(el) = self.stack.pop() {
            let kind = el.operator.expression_kind_follower();
            self.push_expression(Expression {
                kind,
                span: el.span.clone(),
            });

            if let Binding::Delimiter(open) = el.binding {
                self.check_delimiter_match(open, el.span, close, span);
                return;
            }
        }
        self.errors.push(ParseError {
            kind: ParseErrorKind::UnmatchedClosingDelimiter,
            span,
        })
    }

    fn check_delimiter_match(
        &mut self,
        open: P::Delimiter,
        open_span: Span<Idx>,
        close: P::Delimiter,
        close_span: Span<Idx>,
    ) {
        if !open.matches(&close) {
            self.errors.push(ParseError {
                kind: ParseErrorKind::MismatchedDelimiter { opening: open_span },
                span: close_span,
            });
        }
    }

    fn pop_while_lower_precedence(&mut self, precedence: &P::Precedence) {
        while let Some(el) = self.stack.pop_if_lower_precedence(precedence) {
            let kind = el.operator.expression_kind_follower();
            self.push_expression(Expression {
                kind,
                span: el.span,
            });
        }
    }
}

impl<T, TokErr, Idx, P, Q> Extend<Token<T, Idx>> for ParseState<T, TokErr, Idx, P, Q>
where
    Idx: Default + Clone,
    P: Parser<T>,
    Q: Extend<Expression<Idx, P::BinaryOperator, P::UnaryOperator, P::Term>>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Token<T, Idx>>,
    {
        iter.into_iter().for_each(|tok| self.parse_token(tok))
    }
}

impl<T, TokErr, Idx, P, Q> Extend<Result<Token<T, Idx>, TokErr>>
    for ParseState<T, TokErr, Idx, P, Q>
where
    Idx: Default + Clone,
    P: Parser<T>,
    Q: Extend<Expression<Idx, P::BinaryOperator, P::UnaryOperator, P::Term>>,
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

    fn parse_token(&self, kind: T) -> ParserElement<Self, T>;
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

    fn parse_token(&self, kind: T) -> ParserElement<Self, T> {
        P::parse_token(self, kind)
    }
}

pub trait Delimiter {
    fn matches(&self, other: &Self) -> bool;
}

pub struct Element<P, D, B, U, T> {
    pub leader: Option<Leader<P, D, B, U, T>>,
    pub trailer: Option<Trailer<P, D, B, U>>,
}

pub type ParserElement<P, T> = Element<
    <P as Parser<T>>::Precedence,
    <P as Parser<T>>::Delimiter,
    <P as Parser<T>>::BinaryOperator,
    <P as Parser<T>>::UnaryOperator,
    <P as Parser<T>>::Term,
>;

pub struct Leader<P, D, B, U, T> {
    pub implicit_operator: Option<ImplicitOperator<P, B>>,
    pub post_binding: PostBinding<P, D, U, T>,
}

pub struct ImplicitOperator<P, B> {
    pub pre_binding: P,
    pub operator: B,
    pub post_binding: P,
}

pub enum PostBinding<P, D, U, T> {
    Terminal(T),
    Nonterminal {
        terminal: Option<T>,
        binding: Binding<P, D>,
        nonterminal: U,
    },
}

pub struct Trailer<P, D, B, U> {
    pub pre_binding: Binding<P, D>,
    pub post_binding: PostBinding<P, D, B, U>,
}

type ParserTrailer<P, T> = Trailer<
    <P as Parser<T>>::Precedence,
    <P as Parser<T>>::Delimiter,
    <P as Parser<T>>::BinaryOperator,
    <P as Parser<T>>::UnaryOperator,
>;

pub enum Binding<P, D> {
    Precedence(P),
    Delimiter(D),
}

impl<P, D> Binding<P, D> {
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
    Leader,
    Trailer,
}

struct Stack<T, Idx, P: Parser<T>> {
    stack: Vec<StackElement<T, Idx, P>>,
    first_delimiter_idx: Option<usize>,
}

impl<T, Idx, P: Parser<T>> Default for Stack<T, Idx, P> {
    fn default() -> Self {
        Stack {
            stack: Default::default(),
            first_delimiter_idx: None,
        }
    }
}

impl<T, Idx, P: Parser<T>> Stack<T, Idx, P> {
    fn new() -> Self {
        Default::default()
    }

    fn push(&mut self, element: StackElement<T, Idx, P>) {
        if element.binding.is_delimiter() && self.first_delimiter_idx.is_none() {
            self.first_delimiter_idx = Some(self.stack.len());
        }
        self.stack.push(element);
    }

    fn pop(&mut self) -> Option<StackElement<T, Idx, P>> {
        let el = self.stack.pop();
        if Some(self.stack.len()) == self.first_delimiter_idx {
            self.first_delimiter_idx = None
        }
        el
    }

    fn peek_top(&self) -> Option<&StackElement<T, Idx, P>> {
        self.stack.last()
    }

    fn has_delimiter(&self) -> bool {
        self.first_delimiter_idx.is_some()
    }

    /// Pops the stack if the new operator's precedence is less than or equal the top of the stack
    fn pop_if_lower_precedence(
        &mut self,
        precedence: &P::Precedence,
    ) -> Option<StackElement<T, Idx, P>> {
        if Some(precedence) <= self.precedence() {
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
    binding: Binding<P::Precedence, P::Delimiter>,
    operator: StackOperator<P::BinaryOperator, P::UnaryOperator, P::Term>,
}

impl<T, Idx, P: Parser<T>> StackElement<T, Idx, P> {
    fn precedence(&self) -> Option<&P::Precedence> {
        self.binding.precedence()
    }
}

#[derive(Clone, Copy, Debug)]
enum StackOperator<B, U, T> {
    Binary { binary: B, unary: Option<U> },
    Unary { unary: U, term: Option<T> },
}

impl<B, U, T> StackOperator<B, U, T> {
    fn expression_kind_follower(self) -> ExpressionKind<B, U, T> {
        match self {
            Self::Binary { binary, .. } => ExpressionKind::BinaryOperator(binary),
            Self::Unary { unary, .. } => ExpressionKind::UnaryOperator(unary),
        }
    }

    fn expression_kind_no_follower(self) -> Option<ExpressionKind<B, U, T>> {
        match self {
            Self::Unary { term, .. } => term.map(ExpressionKind::Term),
            Self::Binary { unary, .. } => unary.map(ExpressionKind::UnaryOperator),
        }
    }
    fn can_have_no_follower(&self) -> bool {
        match self {
            Self::Unary { term, .. } => term.is_some(),
            Self::Binary { unary, .. } => unary.is_some(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, ops::Range};

    use test_case::test_case;

    use super::{
        parse, parse_one_term, Binding, Delimiter, Element, ImplicitOperator, Leader, ParseState,
        Parser, PostBinding, Trailer, EXPECT_OPERATOR, EXPECT_TERM,
    };
    use crate::{
        error::ParseErrorKind,
        expression::{Expression, ExpressionKind},
        token::{
            charset::{SimpleCharSetTokenKind, SimpleTokenizer, StrSource},
            Tokenizer,
        },
    };

    struct SimpleExprContext;

    #[derive(Clone, Copy, Eq, PartialEq)]
    enum SimpleDelimiter {
        Paren,
        SquareBracket,
        Pipe,
        Conditional,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
    enum SimplePrecedence {
        /// Comma
        Comma,
        /// Conditional operators, `?` and `:`
        Conditional,
        /// Additive operators, such as `+` and `-`
        Additive,
        /// Multiplicative operators, such as `*` and `/`, as well as unary minus.
        Multiplicative,
        /// Exponential operators, such as `^` and `!`
        Exponential,
        /// Function call
        FunctionCall,
    }

    impl Delimiter for SimpleDelimiter {
        fn matches(&self, other: &Self) -> bool {
            self == other
        }
    }

    impl<'s> Parser<(&'s str, SimpleCharSetTokenKind)> for SimpleExprContext {
        type Precedence = SimplePrecedence;
        type Delimiter = SimpleDelimiter;
        type BinaryOperator = &'s str;
        type UnaryOperator = Option<&'s str>;
        type Term = &'s str;

        fn parse_token(
            &self,
            (s, kind): (&'s str, SimpleCharSetTokenKind),
        ) -> Element<
            Self::Precedence,
            Self::Delimiter,
            Self::BinaryOperator,
            Self::UnaryOperator,
            Self::Term,
        > {
            match s {
                "(" => Element {
                    leader: Some(Leader {
                        implicit_operator: None,
                        post_binding: PostBinding::Nonterminal {
                            terminal: None,
                            binding: Binding::Delimiter(SimpleDelimiter::Paren),
                            nonterminal: None,
                        },
                    }),
                    trailer: Some(Trailer {
                        pre_binding: Binding::Precedence(SimplePrecedence::FunctionCall),
                        post_binding: PostBinding::Nonterminal {
                            terminal: Some(Some("()")),
                            binding: Binding::Delimiter(SimpleDelimiter::Paren),
                            nonterminal: s,
                        },
                    }),
                },
                ")" => Element {
                    leader: None,
                    trailer: Some(Trailer {
                        pre_binding: Binding::Delimiter(SimpleDelimiter::Paren),
                        post_binding: PostBinding::Terminal(None),
                    }),
                },
                "[" => Element {
                    leader: Some(Leader {
                        implicit_operator: None,
                        post_binding: PostBinding::Nonterminal {
                            terminal: Some("[]"),
                            binding: Binding::Delimiter(SimpleDelimiter::SquareBracket),
                            nonterminal: Some(s),
                        },
                    }),
                    trailer: None,
                },
                "]" => Element {
                    leader: None,
                    trailer: Some(Trailer {
                        pre_binding: Binding::Delimiter(SimpleDelimiter::SquareBracket),
                        post_binding: PostBinding::Terminal(None),
                    }),
                },
                "|" => Element {
                    leader: Some(Leader {
                        implicit_operator: None,
                        post_binding: PostBinding::Nonterminal {
                            terminal: None,
                            binding: Binding::Delimiter(SimpleDelimiter::Pipe),
                            nonterminal: Some(s),
                        },
                    }),
                    trailer: Some(Trailer {
                        pre_binding: Binding::Delimiter(SimpleDelimiter::Pipe),
                        post_binding: PostBinding::Terminal(None),
                    }),
                },
                "," => Element {
                    leader: None,
                    trailer: Some(Trailer {
                        pre_binding: Binding::Precedence(SimplePrecedence::Comma),
                        post_binding: PostBinding::Nonterminal {
                            terminal: Some(Some("(,)")),
                            binding: Binding::Precedence(SimplePrecedence::Comma),
                            nonterminal: s,
                        },
                    }),
                },
                "-" => Element {
                    leader: Some(Leader {
                        implicit_operator: None,
                        post_binding: PostBinding::Nonterminal {
                            terminal: None,
                            binding: Binding::Precedence(SimplePrecedence::Multiplicative),
                            nonterminal: Some("(-)"),
                        },
                    }),
                    trailer: Some(Trailer {
                        pre_binding: Binding::Precedence(SimplePrecedence::Additive),
                        post_binding: PostBinding::Nonterminal {
                            terminal: None,
                            binding: Binding::Precedence(SimplePrecedence::Additive),
                            nonterminal: s,
                        },
                    }),
                },
                "+" => Element {
                    leader: None,
                    trailer: Some(Trailer {
                        pre_binding: Binding::Precedence(SimplePrecedence::Additive),
                        post_binding: PostBinding::Nonterminal {
                            terminal: None,
                            binding: Binding::Precedence(SimplePrecedence::Additive),
                            nonterminal: s,
                        },
                    }),
                },
                "*" | "/" => Element {
                    leader: None,
                    trailer: Some(Trailer {
                        pre_binding: Binding::Precedence(SimplePrecedence::Multiplicative),
                        post_binding: PostBinding::Nonterminal {
                            terminal: None,
                            binding: Binding::Precedence(SimplePrecedence::Multiplicative),
                            nonterminal: s,
                        },
                    }),
                },
                "^" => Element {
                    leader: None,
                    trailer: Some(Trailer {
                        pre_binding: Binding::Precedence(SimplePrecedence::Exponential),
                        post_binding: PostBinding::Nonterminal {
                            terminal: None,
                            binding: Binding::Precedence(SimplePrecedence::Multiplicative),
                            nonterminal: s,
                        },
                    }),
                },
                "!" => Element {
                    leader: None,
                    trailer: Some(Trailer {
                        pre_binding: Binding::Precedence(SimplePrecedence::Exponential),
                        post_binding: PostBinding::Terminal(Some(s)),
                    }),
                },
                "?" => Element {
                    leader: None,
                    trailer: Some(Trailer {
                        pre_binding: Binding::Precedence(SimplePrecedence::Conditional),
                        post_binding: PostBinding::Nonterminal {
                            terminal: None,
                            binding: Binding::Delimiter(SimpleDelimiter::Conditional),
                            nonterminal: s,
                        },
                    }),
                },
                ":" => Element {
                    leader: None,
                    trailer: Some(Trailer {
                        pre_binding: Binding::Delimiter(SimpleDelimiter::Conditional),
                        post_binding: PostBinding::Nonterminal {
                            terminal: None,
                            binding: Binding::Precedence(SimplePrecedence::Comma),
                            nonterminal: s,
                        },
                    }),
                },
                _ => {
                    // variables get implicit multiplication, other tokens don't (so that we can
                    // test unexpected token errors)
                    let implicit_operator = if let SimpleCharSetTokenKind::Tag = kind {
                        Some(ImplicitOperator {
                            pre_binding: SimplePrecedence::Multiplicative,
                            operator: "{*}",
                            post_binding: SimplePrecedence::Multiplicative,
                        })
                    } else {
                        None
                    };
                    Element {
                        leader: Some(Leader {
                            implicit_operator,
                            post_binding: PostBinding::Terminal(s),
                        }),
                        trailer: None,
                    }
                }
            }
        }
    }

    fn expr_to_str<'s, Idx>(
        expr: Expression<Idx, &'s str, Option<&'s str>, &'s str>,
    ) -> Option<&'s str> {
        match expr.kind {
            ExpressionKind::BinaryOperator(s) => Some(s),
            ExpressionKind::UnaryOperator(s) => s,
            ExpressionKind::Term(s) => Some(s),
        }
    }

    #[test_case("3 + 4 * 2 / ( 1 - 5 ) ^ 2 ^ 3", "3 4 2 * 1 5 - 2 3 ^ ^ / +" ; "simple arithmetic" )]
    #[test_case("sin(max(5/2, 3)) / 3 * pi", "sin max 5 2 / 3 , ( ( 3 / pi *" ; "with functions" )]
    #[test_case("2^3!", "2 3 ! ^" ; "postfix operators" )]
    #[test_case("-2^3 + (-2)^3", "2 3 ^ (-) 2 (-) 3 ^ +" ; "prefix operators" )]
    #[test_case("[1, 2, 3, 4]", "1 2 , 3 , 4 , [" ; "delimiter operators" )]
    #[test_case("[1, (2, 3), 4]", "1 2 3 , , 4 , [" ; "nested delimiter operators" )]
    #[test_case("[ ]", "[]" ; "empty list" )]
    #[test_case("[ ] + [ ]", "[] [] +" ; "adding lists" )]
    #[test_case("f()", "f ()" ; "empty function call" )]
    #[test_case("-f()", "f () (-)" ; "function call with prefix operator" )]
    #[test_case("[1, 2, 3, 4, ]", "1 2 , 3 , 4 , (,) [" ; "trailing comma" )]
    #[test_case("a * |b|", "a b | *" ; "absolute value" )]
    #[test_case("a, * b", "a (,) b *" ; "trailing comma with binary operator" )]
    #[test_case("1 , - 2", "1 2 (-) ,"; "comma with prefix operator")]
    #[test_case("5x^2", "5 x 2 ^ {*}" ; "implicit operator" )]
    #[test_case("2 ^ 3 * 4", "2 3 ^ 4 *" ; "right associativity" )]
    #[test_case("P ? a : Q ? b : c", "P a ? Q b ? c : :" ; "chained conditionals" )]
    #[test_case("P ? Q ? a : b : c", "P Q a ? b : ? c :" ; "nested conditionals" )]
    #[test_case("P ? a, x : b, Q ? c : d", "P a x , ? b : Q c ? d : ," ; "conditionals and commas" )]
    fn parse_expression(input: &str, output: &str) -> anyhow::Result<()> {
        let actual = parse::<_, _, Vec<_>>(
            SimpleTokenizer::new(StrSource::new(input)),
            SimpleExprContext,
        )?
        .into_iter()
        .filter_map(expr_to_str)
        .collect::<Vec<_>>();
        let expected = output.split_whitespace().collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test_case("3", "3", "" ; "single term" )]
    #[test_case("-3!", "3 (-)", "!" ; "unary operators" )]
    #[test_case("3!", "3", "!" ; "postfix operator" )]
    #[test_case("-3 a", "3 (-)", "a" ; "unary operators with additional" )]
    #[test_case("(5 + 4) * (3 - 2)", "5 4 +", "* (3 - 2)" ; "delimited group" )]
    #[test_case("(3)!", "3", "!" ; "delimited with unary operators" )]
    #[test_case("abc def)", "abc", "def)" ; "ignores invalid after first term" )]
    fn parse_one(input: &str, output: &str, rest: &str) -> anyhow::Result<()> {
        let mut tokens = SimpleTokenizer::new(StrSource::new(input));
        let actual = parse_one_term::<_, _, Vec<_>>(&mut tokens, SimpleExprContext)?
            .into_iter()
            .filter_map(expr_to_str)
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

    #[test_case("-", false ; "unary")]
    #[test_case("-3", true ; "unary with rhs")]
    #[test_case("-3 *", false ; "binary")]
    #[test_case("-3 * 3", true ; "binary with rhs")]
    #[test_case("-(4 + (3 * 3)", false ; "incomplete delimiter")]
    #[test_case("-(4 + (3 * 3))", true ; "complete delimiter")]
    #[test_case("3,", true ; "binary with optional rhs")]
    fn is_complete(input: &str, is_complete: bool) -> anyhow::Result<()> {
        let mut tokens = SimpleTokenizer::new(StrSource::new(input));
        let mut state = ParseState::<_, _, _, _, Vec<_>>::new(SimpleExprContext);
        while let Some(t) = tokens.next_token() {
            state.parse_result(t);
        }
        assert_eq!(state.has_parsed_expression(), is_complete);
        if is_complete {
            state.finish()?;
        } else {
            state.finish().unwrap_err();
        }
        Ok(())
    }

    #[test_case("1 )", &[(ParseErrorKind::UnmatchedClosingDelimiter, 2..3)] ; "unmatched right paren" )]
    #[test_case("1 +", &[(ParseErrorKind::EndOfInput { expected: EXPECT_TERM }, 3..3)] ; "end of input" )]
    #[test_case("(5 5 +", &[
        (ParseErrorKind::UnexpectedToken { expected: EXPECT_OPERATOR }, 3..4),
        (ParseErrorKind::UnexpectedToken { expected: EXPECT_TERM }, 5..6),
        (ParseErrorKind::EndOfInput { expected: EXPECT_TERM }, 6..6),
        (ParseErrorKind::UnmatchedOpeningDelimiter, 0..1),
    ] ; "multiple errors")]
    #[test_case("[ 1 )", &[
        (ParseErrorKind::MismatchedDelimiter { opening: (0..1).into() }, 4..5),
    ] ; "mismatched delimiters" )]
    #[test_case("( [ 1 )", &[
        (ParseErrorKind::MismatchedDelimiter { opening: (2..3).into() }, 6..7),
        (ParseErrorKind::UnmatchedOpeningDelimiter, 0..1),
    ] ; "mismatched delimiters 2" )]
    #[test_case("[ 1 + )", &[
        (ParseErrorKind::UnexpectedToken { expected: EXPECT_TERM }, 6..7),
        (ParseErrorKind::MismatchedDelimiter { opening: (0..1).into() }, 6..7),
    ] ; "mismatched delimiters with missing rhs" )]
    #[test_case("1 + * 2", &[
        (ParseErrorKind::UnexpectedToken { expected: EXPECT_TERM }, 4..5),
    ] ; "extra operator" )]
    #[test_case("* 3", &[
        (ParseErrorKind::UnexpectedToken { expected: EXPECT_TERM }, 0..1),
    ] ; "initial operator")]
    #[test_case("[ * 3", &[
        (ParseErrorKind::UnexpectedToken { expected: EXPECT_TERM }, 2..3),
        (ParseErrorKind::UnmatchedOpeningDelimiter, 0..1),
    ] ; "operator after brackets")]
    fn parse_expression_fail(
        input: &str,
        expected: &[(ParseErrorKind<Infallible, usize>, Range<usize>)],
    ) {
        let actual = parse::<_, _, Vec<_>>(
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
