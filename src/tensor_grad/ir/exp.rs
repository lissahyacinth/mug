use std::sync::Arc;

use co::{plugin::numeric_helpers::NumCast, prelude::*};
use coaster as co;
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::tensor_op::arith::pow::TensorPow;

use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::{TensorIRStruct, TensorInput, TensorOp},
};

impl<T: 'static, F> TensorOp<T, F>
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
    pub fn pow(self, exp: T) -> TensorOp<T, F> {
        TensorOp::new(
            TensorInput::Op(Box::new(self)),
            TensorInput::None,
            Box::new(TensorPow::new(exp)),
        )
    }
}

impl<T: 'static, F> ArcTensor<T, F>
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
    pub fn pow(self, exp: T) -> TensorIRStruct<T, F> {
        TensorIRStruct::new(self, None, Box::new(TensorPow { exp }))
    }
}

impl<T, F> TensorIRStruct<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>
        + 'static,
    F: IFramework + Clone + 'static,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    pub fn pow(self, exp: T) -> TensorIRStruct<T, F> {
        self.scalar_op(Box::new(TensorPow::new(exp)))
    }
}
