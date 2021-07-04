use std::rc::Rc;

use coaster::{IBackend, IFramework};
use coaster_blas::plugin::{Asum, Axpy, Gemm};

use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    NumCast,
};

use super::{arith::Side, chain_gradient::ChainGradient, TensorIRStruct, TensorInput};

#[macro_export]
macro_rules! call_function {
    ($lhs_tensor:expr, $rhs_tensor: expr, $Func: expr) => {
        match $lhs_tensor {
            Some(l_tensor) => match $rhs_tensor {
                Some(r_tensor) => $Func(l_tensor, r_tensor),
                None => unreachable!(),
            },
            None => unreachable!(),
        }
    };
}

pub trait TensorOperation<T, F>: CloneableOps<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + Clone
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
    fn to_string(&self) -> String;
    fn output_shape(&self, lhs_shape: Option<&[usize]>, rhs_shape: Option<&[usize]>) -> Vec<usize>;
    fn evaluate(
        &self,
        lhs: Option<ArcTensor<T, F>>,
        rhs: Option<ArcTensor<T, F>>,
        backend: &Rc<coaster::Backend<F>>,
    ) -> ArcTensor<T, F>;
    fn grad(
        &self,
        seed: TensorIRStruct<T, F>,
        lhs: TensorInput<T, F>,
        other: TensorInput<T, F>,
        side: Side,
    ) -> TensorIRStruct<T, F>;
}

pub trait CloneableOps<T, F> {
    fn internal_clone(&self) -> Box<dyn TensorOperation<T, F>>;
}

impl<T, F, G> CloneableOps<T, F> for G
where
    G: TensorOperation<T, F> + Clone + 'static,
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
    fn internal_clone<'a>(&self) -> Box<dyn TensorOperation<T, F>> {
        Box::new(self.clone())
    }
}

impl<T, F> Clone for Box<dyn TensorOperation<T, F>> {
    fn clone(&self) -> Self {
        self.internal_clone()
    }
}
