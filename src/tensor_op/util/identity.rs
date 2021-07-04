use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::{arith::Side, chain_gradient::ChainGradient, TensorIRStruct, TensorInput},
    NumCast,
};
use coaster::{IBackend, IFramework};
use coaster_blas::plugin::{Asum, Axpy, Gemm};

use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::tensor_op::operation::TensorOperation;
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct TensorIdentity {}

impl TensorIdentity {
    pub(crate) fn new() -> Self {
        TensorIdentity {}
    }
}

impl<T, F> TensorOperation<T, F> for TensorIdentity
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>
        + Copy
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>
        + 'static,
    F: IFramework + Clone + 'static + 'static,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    fn to_string(&self) -> String {
        "Identity".to_string()
    }

    fn output_shape(&self, lhs_shape: Option<&[usize]>, rhs_shape: Option<&[usize]>) -> Vec<usize> {
        match lhs_shape {
            Some(lhs_shape) => match rhs_shape {
                None => lhs_shape.to_vec(),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    fn evaluate(
        &self,
        lhs: Option<ArcTensor<T, F>>,
        _rhs: Option<ArcTensor<T, F>>,
        _backend: &Rc<coaster::Backend<F>>,
    ) -> ArcTensor<T, F> {
        lhs.unwrap()
    }

    fn grad(
        &self,
        seed: TensorIRStruct<T, F>,
        lhs: TensorInput<T, F>,
        other: TensorInput<T, F>,
        side: Side,
    ) -> TensorIRStruct<T, F> {
        seed
    }
}

#[cfg(test)]
mod test {}
