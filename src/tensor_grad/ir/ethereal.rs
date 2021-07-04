use std::marker::PhantomData;

use co::{plugin::numeric_helpers::NumCast, prelude::*};
use coaster as co;
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::tensor_grad::Tensor;

use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::TensorIRStruct,
    utility::filled_tensor,
};

#[derive(Clone, Debug)]
/// Temporary Tensor for Resizing or Modifying other Operations
///
/// Designed to allow for simpler operations in backprop, without
/// requiring a new tensor to be initialised before it's known what
/// backend is going to be used.
pub struct EtherealTensor<T, F> {
    pub(crate) id: uuid::Uuid,
    data: Vec<T>,
    phantom_backend: PhantomData<F>,
    shape: Vec<usize>,
}

impl<T, F> EtherealTensor<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>
        + Copy,
    F: IFramework + Clone + 'static,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    pub(crate) fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let id = uuid::Uuid::new_v4();
        println!(
            "Creating Ethereal Tensor with ID {} and Shape {:?}",
            id, shape
        );
        EtherealTensor {
            id,
            data,
            phantom_backend: PhantomData,
            shape,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn to_tensor(&self, backend: &::std::rc::Rc<Backend<F>>) -> ArcTensor<T, F>
    where
        F: IFramework + Clone + 'static,
        coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
        ArcTensor<T, F>: ReadTensor<T>,
    {
        println!(
            "Converting Ethereal Tensor {} to Tensor with Shape {:?}",
            self.id,
            self.shape()
        );
        Tensor::with_id(
            filled_tensor(backend, self.shape(), &self.data),
            self.shape.to_vec(),
            backend,
            self.id,
        )
    }
}
