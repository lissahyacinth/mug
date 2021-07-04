pub(crate) mod ir;
pub(crate) mod tensor;
pub(crate) mod tensor_ir;

use co::{plugin::numeric_helpers::NumCast, prelude::*};
use coaster as co;
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

pub use crate::tensor_grad::tensor::Tensor;

use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::TensorOp,
};

pub trait TensorOperationIR<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>,
    F: IFramework + Clone + 'static,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    fn output(&self) -> TensorOp<T, F>;
    fn output_shape(&self, lhs_shape: &[usize], rhs_shape: Option<&[usize]>) -> Vec<usize>;
}
