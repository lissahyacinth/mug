use crate::{
    tensor_grad::{
        ir::scalar_mul::ScalarMultiplication,
        tensor::{ArcTensor, ReadTensor},
    },
    tensor_op::{arith::Side, chain_gradient::ChainGradient, TensorIRStruct, TensorInput},
    utility::filled_tensor,
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
pub(crate) struct TensorPow<T> {
    pub(crate) exp: T,
}

impl<T> TensorPow<T> {
    pub(crate) fn new(exp: T) -> Self {
        TensorPow { exp }
    }
}

impl<T, F> TensorOperation<T, F> for TensorPow<T>
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
    Standard: Distribution<T>,
    F: IFramework + Clone + 'static,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    fn to_string(&self) -> String {
        "Pow".to_string()
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
        backend: &Rc<coaster::Backend<F>>,
    ) -> ArcTensor<T, F> {
        let lhs = lhs.unwrap();
        let lhs_shape = lhs.shape();
        let _lhs_tensor = lhs.get_tensor();
        let input = lhs.read();
        let output = input
            .into_iter()
            .map(|x| x.pow(self.exp))
            .collect::<Vec<T>>();
        Tensor::new(
            filled_tensor(backend, lhs_shape, &output),
            lhs.shape().to_vec(),
            backend,
        )
    }

    fn grad(
        &self,
        seed: TensorIRStruct<T, F>,
        lhs: TensorInput<T, F>,
        other: TensorInput<T, F>,
        side: Side,
    ) -> TensorIRStruct<T, F> {
        let scalar_mult = Box::new(ScalarMultiplication { value: self.exp });
        match lhs {
            TensorInput::Op(op) => {
                let grad = Box::new(op * self.exp);
                match side {
                    Side::Left => grad * seed,
                    Side::Right => seed * grad,
                }
            }
            TensorInput::Tensor(tensor) => {
                let grad = Box::new(tensor * self.exp);
                match side {
                    Side::Left => grad * seed,
                    Side::Right => seed * grad,
                }
            }
            TensorInput::EtherealTensor(e_tensor) => {
                let grad = Box::new(e_tensor * self.exp);
                match side {
                    Side::Left => grad * seed,
                    Side::Right => seed * grad,
                }
            }
            TensorInput::None => unreachable!(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utility::get_native_backend;

    #[test]
    fn check_pow() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(&backend, &[4, 1], &[2.0_f64, 4.0, 8.0, 9.0]),
            vec![4, 1],
            &backend,
        );
        let mut tensor_res = input.pow(2.0);
        println!("{}", tensor_res.graph());
        let output = tensor_res.evaluate(&backend).read();
        assert_eq!(vec![4.0_f64, 16.0, 64.0, 81.0], output);
    }
}
