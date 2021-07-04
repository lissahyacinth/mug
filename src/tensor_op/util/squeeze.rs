use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::{arith::Side, chain_gradient::ChainGradient, TensorIRStruct, TensorInput},
    utility::copy_tensor,
    NumCast,
};
use coaster::{IBackend, IFramework};
use coaster_blas::plugin::{Asum, Axpy, Gemm};

use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{tensor_grad::Tensor, tensor_op::operation::TensorOperation};
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct TensorSqueeze {
    pub(crate) dimension: usize,
}

impl TensorSqueeze {
    pub(crate) fn new(dimension: usize) -> Self {
        TensorSqueeze { dimension }
    }
}

impl<T, F> TensorOperation<T, F> for TensorSqueeze
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
        + num::traits::Pow<T, Output = T>,
    F: IFramework + Clone + 'static,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    fn to_string(&self) -> String {
        "Squeeze".to_string()
    }

    fn output_shape(&self, lhs_shape: Option<&[usize]>, rhs_shape: Option<&[usize]>) -> Vec<usize> {
        match lhs_shape {
            Some(lhs_shape) => match rhs_shape {
                None => {
                    assert!(lhs_shape[self.dimension] == 1);
                    let mut lhs_shape = lhs_shape.to_vec();
                    lhs_shape.remove(self.dimension);
                    lhs_shape
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    fn evaluate(
        &self,
        lhs: Option<ArcTensor<T, F>>,
        _rhs: Option<ArcTensor<T, F>>,
        backend: &Rc<coaster::Backend<F>>,
    ) -> ArcTensor<T, F> {
        let lhs = lhs.unwrap();
        let output_shape = self.output_shape(Some(lhs.shape()), None);
        let mut new_tensor = copy_tensor(backend, lhs.shape(), &lhs.get_tensor().tensor);
        new_tensor.reshape(&output_shape).unwrap();
        Tensor::new(new_tensor, output_shape, backend)
    }

    fn grad(
        &self,
        seed: TensorIRStruct<T, F>,
        lhs: TensorInput<T, F>,
        other: TensorInput<T, F>,
        side: Side,
    ) -> TensorIRStruct<T, F> {
        todo!()
    }
}

#[cfg(test)]
mod test {}
