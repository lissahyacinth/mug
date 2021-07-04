use co::{plugin::numeric_helpers::NumCast, prelude::*};
use coaster as co;
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::tensor_grad::tensor_ir::TensorIR;
use crate::tensor_op::util::{
    squeeze::TensorSqueeze, transpose::TensorTranspose, unsqueeze::TensorUnSqueeze,
};

use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::{TensorIRStruct, TensorInput, TensorOp},
};

use super::ethereal::EtherealTensor;

impl<T, F> TensorOp<T, F>
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
    pub fn t(&self) -> TensorOp<T, F> {
        TensorOp::new(
            TensorInput::Op(Box::new(self.clone())),
            TensorInput::None,
            Box::new(TensorTranspose {}),
        )
    }
}

impl<T, F> ArcTensor<T, F>
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
    pub fn t(&self) -> TensorIRStruct<T, F> {
        TensorIRStruct::new(self.clone(), None, Box::new(TensorTranspose {}))
    }
}

impl<T, F> TensorIR<T, F>
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
    pub fn t(&self) -> TensorOp<T, F> {
        TensorOp::new(
            TensorInput::Tensor(self.clone()),
            TensorInput::None,
            Box::new(TensorTranspose {}),
        )
    }
}

impl<T, F> EtherealTensor<T, F>
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
    pub fn t(&self) -> TensorOp<T, F> {
        TensorOp::new(
            TensorInput::EtherealTensor(self.clone()),
            TensorInput::None,
            Box::new(TensorTranspose {}),
        )
    }
}

impl<T, F> TensorOp<T, F>
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
    pub fn squeeze(&self, dimension: usize) -> TensorOp<T, F> {
        TensorOp::new(
            TensorInput::Op(Box::new(self.clone())),
            TensorInput::None,
            Box::new(TensorSqueeze { dimension }),
        )
    }
}

impl<T, F> ArcTensor<T, F>
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
    pub fn squeeze(&self, dimension: usize) -> TensorIRStruct<T, F> {
        TensorIRStruct::new(self.clone(), None, Box::new(TensorSqueeze::new(dimension)))
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
    pub fn unsqueeze(&self, dimension: usize) -> TensorIRStruct<T, F> {
        self.clone()
            .scalar_op(Box::new(TensorUnSqueeze::new(dimension)))
    }
}

impl<T, F> ArcTensor<T, F>
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
    pub fn unsqueeze(&self, dimension: usize) -> TensorIRStruct<T, F> {
        TensorIRStruct::new(
            self.clone(),
            None,
            Box::new(TensorUnSqueeze::new(dimension)),
        )
    }
}
