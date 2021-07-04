use crate::{
    tensor_grad::{
        ir::{ethereal::EtherealTensor, scalar_mul::ScalarMultiplication},
        tensor::{ArcTensor, ReadTensor},
    },
    tensor_op::{chain_gradient::ChainGradient, TensorIRStruct, TensorInput},
    NumCast,
};
use coaster::{IBackend, IFramework, SharedTensor};
use coaster_blas::plugin::{Asum, Axpy, Gemm};

use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{tensor_grad::Tensor, tensor_op::operation::TensorOperation};

use std::rc::Rc;

use super::Side;

#[derive(Clone)]
pub(crate) struct TensorSum {}

impl TensorSum {
    pub(crate) fn new() -> Self {
        TensorSum {}
    }
}

impl<T, F> TensorOperation<T, F> for TensorSum
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>
        + Copy
        + 'static,
    F: IFramework + Clone + 'static,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    fn to_string(&self) -> String {
        "Sum".to_string()
    }

    fn output_shape(
        &self,
        _lhs_shape: Option<&[usize]>,
        _rhs_shape: Option<&[usize]>,
    ) -> Vec<usize> {
        return vec![1];
    }

    fn evaluate(
        &self,
        lhs: Option<ArcTensor<T, F>>,
        _rhs: Option<ArcTensor<T, F>>,
        backend: &Rc<coaster::Backend<F>>,
    ) -> ArcTensor<T, F> {
        let lhs = lhs.unwrap();
        let output_shape = self.output_shape(Some(lhs.shape()), None);
        let mut new_tensor: SharedTensor<T> = SharedTensor::new(&output_shape);
        (*backend)
            .asum(&lhs.get_tensor().tensor, &mut new_tensor)
            .unwrap();
        Tensor::new(new_tensor, output_shape, backend)
    }

    fn grad(
        &self,
        seed: TensorIRStruct<T, F>,
        lhs: TensorInput<T, F>,
        other: TensorInput<T, F>,
        side: Side,
    ) -> TensorIRStruct<T, F> {
        let grad = match lhs {
            TensorInput::Op(op) => {
                let output_shape = op.output_shape;
                sum_gradient_ethereal(output_shape)
            }
            TensorInput::Tensor(tensor) => {
                let output_shape = tensor.shape();
                sum_gradient_ethereal(output_shape.to_vec())
            }
            TensorInput::None => unreachable!(),
            TensorInput::EtherealTensor(tensor) => {
                let output_shape = tensor.shape();
                sum_gradient_ethereal(output_shape.to_vec())
            }
        };
        match side {
            Side::Left => grad * seed,
            Side::Right => seed * grad,
        }
    }
}

/// Create a Row Vector of 1s for the Gradient of a Sum or Mean Operation
///
// This is at least validated - it looks right from the maths.
fn sum_gradient_ethereal<T, F>(output_shape: Vec<usize>) -> EtherealTensor<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>
        + Copy
        + 'static,
    F: IFramework + Clone + 'static,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    dbg!(&output_shape);
    dbg!(output_shape.len());
    match output_shape.len() {
        2 => {
            let mut non_one_size = output_shape
                .clone()
                .into_iter()
                .filter(|x| *x > 1)
                .collect::<Vec<usize>>();
            dbg!(&non_one_size);
            let remaining_size = match non_one_size.pop() {
                Some(size) => size,
                None => 1_usize,
            };
            dbg!(remaining_size);
            let shape = if output_shape[0] == 1 {
                vec![remaining_size, 1]
            } else {
                vec![remaining_size, 1]
            };
            EtherealTensor::new(vec![T::one(); remaining_size], shape)
        }
        1 => {
            let remaining_size = output_shape.clone().pop().unwrap();
            let shape = if output_shape[0] == 1 {
                vec![remaining_size, 1]
            } else {
                vec![remaining_size, 1]
            };
            EtherealTensor::new(vec![T::one(); remaining_size], shape)
        }
        _ => panic!("Gradient of Tensor Sum is only defined for Input Tensors of Order 2 or less"),
    }
}

#[cfg(test)]
mod test {
    use crate::utility::get_native_backend;

    use super::*;
    use crate::tensor_grad::tensor::ArrayOutput;
    use coaster::frameworks::cuda::get_cuda_backend;
    use ndarray::prelude::*;

    fn create_2d_array() -> Array2<f32> {
        return array![[1., 2., 3.], [4., 5., 6.],];
    }

    #[test]
    fn check_summation() {
        let backend = Rc::new(get_native_backend());
        let input = create_2d_array();
        let input_sum = input.clone().sum();
        let tensor = Tensor::from_array(input, &backend);
        let mut tensor_res = tensor.sum();
        let tensor_sum = match tensor_res.evaluate(&backend).to_ndarray() {
            ArrayOutput::Tensor0D(_) => 0.0f32,
            ArrayOutput::Tensor1D(array) => array.sum(),
            ArrayOutput::Tensor2D(array) => array.sum(),
        };
        assert_eq!(input_sum, tensor_sum);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn check_summation_cuda() {
        let backend = Rc::new(get_cuda_backend());
        let input = create_2d_array();
        let input_sum = input.sum();
        let tensor_a = Tensor::from_array(input.clone(), &backend);
        let mut tensor_res = tensor_a.sum();
        let tensor_sum = match tensor_res.evaluate(&backend).to_ndarray() {
            ArrayOutput::Tensor0D(_) => 0.0f32,
            ArrayOutput::Tensor1D(array) => array.sum(),
            ArrayOutput::Tensor2D(array) => array.sum(),
        };
        assert_eq!(input_sum, tensor_sum);
    }
}
