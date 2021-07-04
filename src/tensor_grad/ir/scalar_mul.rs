use co::{plugin::numeric_helpers::NumCast, prelude::*};
use coaster as co;
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::tensor_op::arith::scal::TensorScal;

use crate::tensor_op::{TensorInput, TensorOp};

use crate::{
    tensor_grad::{
        tensor::{ArcTensor, ReadTensor},
        tensor_ir::TensorIR,
    },
    tensor_op::TensorIRStruct,
};

use super::ethereal::EtherealTensor;

impl<T, F> std::ops::Mul<T> for ArcTensor<T, F>
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
    type Output = TensorIRStruct<T, F>;
    fn mul(self, other: T) -> Self::Output {
        TensorIRStruct::new(self, None, Box::new(TensorScal::new(other)))
    }
}

impl<T, F> std::ops::Mul<T> for &ArcTensor<T, F>
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
    type Output = TensorIRStruct<T, F>;
    fn mul(self, other: T) -> Self::Output {
        TensorIRStruct::new(self.clone(), None, Box::new(TensorScal::new(other)))
    }
}

impl<T, F> std::ops::Mul<T> for &mut ArcTensor<T, F>
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
    type Output = TensorIRStruct<T, F>;
    fn mul(self, other: T) -> Self::Output {
        TensorIRStruct::new(self.clone(), None, Box::new(TensorScal::new(other)))
    }
}

impl<T, F> std::ops::Mul<T> for TensorIR<T, F>
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
    type Output = TensorOp<T, F>;
    fn mul(self, other: T) -> Self::Output {
        TensorOp::new(
            TensorInput::Tensor(self),
            TensorInput::None,
            Box::new(TensorScal::new(other)),
        )
    }
}

impl<T, F> std::ops::Mul<T> for TensorIRStruct<T, F>
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
    type Output = TensorIRStruct<T, F>;
    fn mul(self, other: T) -> Self::Output {
        self.scalar_op(Box::new(TensorScal::new(other)))
    }
}

pub struct ScalarMultiplication<T> {
    pub value: T,
}

impl<T, F> std::ops::Mul<TensorIRStruct<T, F>> for ScalarMultiplication<T>
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
    type Output = TensorIRStruct<T, F>;
    fn mul(self, other: TensorIRStruct<T, F>) -> Self::Output {
        other.scalar_op(Box::new(TensorScal::new(self.value)))
    }
}

impl<T, F> std::ops::Mul<T> for Box<TensorOp<T, F>>
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
    type Output = TensorOp<T, F>;
    fn mul(self, other: T) -> Self::Output {
        TensorOp::new(
            TensorInput::Op(self),
            TensorInput::None,
            Box::new(TensorScal::new(other)),
        )
    }
}

impl<T, F> std::ops::Mul<T> for EtherealTensor<T, F>
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
    type Output = TensorOp<T, F>;
    fn mul(self, other: T) -> Self::Output {
        TensorOp::new(
            TensorInput::EtherealTensor(self),
            TensorInput::None,
            Box::new(TensorScal::new(other)),
        )
    }
}
