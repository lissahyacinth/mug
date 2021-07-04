use crate::{
    tensor_grad::{
        ir::{ethereal::EtherealTensor, scalar_mul::ScalarMultiplication},
        tensor::{ArcTensor, ReadTensor},
    },
    tensor_op::{chain_gradient::ChainGradient, TensorIRStruct, TensorInput},
    utility::{element_tensor, ones, zeros},
    NumCast,
};
use coaster::{IBackend, IFramework};
use coaster_blas::{
    plugin::{Asum, Axpy, Gemm},
    transpose::Transpose,
};

use super::Side;
use log::error;
use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{tensor_grad::Tensor, tensor_op::operation::TensorOperation};
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct TensorMean {}

impl TensorMean {
    pub(crate) fn new() -> Self {
        TensorMean {}
    }
}

impl<T, F> TensorOperation<T, F> for TensorMean
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
        + 'static,
    Standard: Distribution<T>,
    F: IFramework + Clone + 'static,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
    T: std::ops::Div<Output = T>,
{
    fn to_string(&self) -> String {
        "Mean".to_string()
    }

    fn output_shape(
        &self,
        lhs_shape: Option<&[usize]>,
        _rhs_shape: Option<&[usize]>,
    ) -> Vec<usize> {
        match lhs_shape {
            Some(lhs_shape) => {
                if lhs_shape.iter().filter(|x| **x != 1).count() > 1 {
                    error!("Tensor provided to Mean of shape {:?}", &lhs_shape);
                    panic!("Mean does not currently support Tensors with rank greater than 1")
                }
                return vec![1];
            }
            None => unreachable!(),
        }
    }

    fn evaluate(
        &self,
        lhs: Option<ArcTensor<T, F>>,
        _rhs: Option<ArcTensor<T, F>>,
        backend: &Rc<coaster::Backend<F>>,
    ) -> ArcTensor<T, F> {
        let lhs = lhs.unwrap();
        let _lhs_shape = lhs.shape();
        let lhs_tensor = lhs.get_tensor();
        let output_shape = self.output_shape(Some(lhs.shape()), None);
        let mut new_tensor = element_tensor(backend, T::zero());
        let inv_n_elements: T = T::one() / num::cast::<usize, T>(lhs.shape()[0]).unwrap();
        let ones_shape = if lhs.shape().len() == 1 {
            vec![1, lhs.shape()[0]]
        } else {
            vec![lhs.shape()[1], lhs.shape()[0]]
        };
        (**backend)
            .gemm(
                &element_tensor(backend, inv_n_elements),
                Transpose::NoTrans,
                &ones(backend, &ones_shape).get_tensor().tensor,
                Transpose::NoTrans,
                &lhs_tensor.tensor,
                &zeros(backend, &[1]),
                &mut new_tensor,
            )
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
        let output_shape = match lhs {
            TensorInput::Op(op) => op.output_shape,
            TensorInput::Tensor(tensor) => tensor.shape().to_vec(),
            TensorInput::None => unreachable!(),
            TensorInput::EtherealTensor(tensor) => tensor.shape().to_vec(),
        };
        let grad = mean_gradient_ethereal(output_shape);
        match side {
            Side::Left => grad * seed,
            Side::Right => seed * grad,
        }
    }
}

fn mean_gradient_ethereal<T, F>(output_shape: Vec<usize>) -> EtherealTensor<T, F>
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
    match output_shape.len() {
        2 => {
            let mut non_one_size = output_shape
                .clone()
                .into_iter()
                .filter(|x| *x > 1)
                .collect::<Vec<usize>>();
            let remaining_size = match non_one_size.pop() {
                Some(size) => size,
                None => 1_usize,
            };
            let shape = if output_shape[0] == 1 {
                vec![1, remaining_size]
            } else {
                vec![remaining_size, 1]
            };
            EtherealTensor::new(
                vec![T::one() / num::cast::<usize, T>(remaining_size).unwrap(); remaining_size],
                shape,
            )
        }
        1 => {
            let remaining_size = output_shape.clone().pop().unwrap();
            let shape = if output_shape[0] == 1 {
                vec![1, remaining_size]
            } else {
                vec![remaining_size, 1]
            };
            EtherealTensor::new(
                vec![T::one() / num::cast::<usize, T>(remaining_size).unwrap(); remaining_size],
                shape,
            )
        }
        _ => panic!("Gradient of Tensor Sum is only defined for Input Tensors of Order 2 or less"),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utility::{filled_tensor, get_native_backend};

    #[test]
    fn check_mean() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(&backend, &[4], &[4.0, 16.0, 64.0, 81.0]),
            vec![4],
            &backend,
        );
        let mut tensor_res = input.mean();
        println!("{}", tensor_res.graph());
        let output = tensor_res.evaluate(&backend).read();
        assert_eq!(vec![41.25], output);
    }
}
