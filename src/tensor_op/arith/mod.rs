pub(crate) mod add;
pub(crate) mod mean;
pub(crate) mod multiply;
pub(crate) mod pow;
pub(crate) mod scal;
pub(crate) mod sqrt;
pub(crate) mod sum;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}
