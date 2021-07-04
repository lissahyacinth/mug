pub(crate) mod add;
pub(crate) mod agg;
pub(crate) mod error;
pub(crate) mod ethereal;
pub(crate) mod exp;
pub(crate) mod hadamard;
pub(crate) mod mul;
pub(crate) mod scalar_mul;
pub(crate) mod sub;
pub(crate) mod util;

fn transpose_shape(shape: &[usize]) -> Vec<usize> {
    shape.into_iter().rev().cloned().collect()
}
