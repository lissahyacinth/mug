use std::marker::PhantomData;

use super::{
    tensor::{ReadTensor, TensorID},
    ArcTensor,
};
use co::{plugin::numeric_helpers::NumCast, prelude::*};
use coaster as co;

use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use rand::{distributions::Standard, prelude::Distribution};

use num_traits::real::Real;

#[derive(Debug, Clone)]
pub struct TensorIR<T, F> {
    pub(crate) id: TensorID,
    shape: Vec<usize>,
    value_type: PhantomData<T>,
    backend_type: PhantomData<F>,
    pub(crate) has_gradient: bool,
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
        + num::traits::Pow<T, Output = T>,
    Standard: Distribution<T>,
    F: IFramework + Clone + 'static,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    pub fn shape(&self) -> &[usize] {
        &self.shape
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
        + num::traits::Pow<T, Output = T>,
    Standard: Distribution<T>,
    F: IFramework + Clone + 'static,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    pub(crate) fn ir(&self) -> TensorIR<T, F> {
        //println!(
        //    "Generating new Tensor IR from Tensor with ID {}",
        //    self.tensor_id
        //);
        TensorIR {
            id: self.tensor_id,
            shape: self.shape().to_vec(),
            value_type: PhantomData,
            backend_type: PhantomData,
            has_gradient: self.has_gradient,
        }
    }
}
