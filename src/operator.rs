#[derive(Clone, Copy, Debug)]
pub enum Fixity<P> {
    Left(P),
    Right(P),
}

impl<P> Fixity<P> {
    pub fn precedence(&self) -> &P {
        match self {
            Fixity::Left(prec) | Fixity::Right(prec) => prec,
        }
    }

    pub fn into_precedence(self) -> P {
        match self {
            Fixity::Left(prec) | Fixity::Right(prec) => prec,
        }
    }
}
