use crate::{
    tensor_grad::{
        ir::scalar_mul::ScalarMultiplication,
        tensor::{ArcTensor, ReadTensor},
    },
    tensor_op::{chain_gradient::ChainGradient, TensorIRStruct, TensorInput},
    utility::{filled_tensor, identity, zeros},
    NumCast,
};
use coaster::{IBackend, IFramework, SharedTensor};
use coaster_blas::{
    plugin::{Asum, Axpy, Gemm},
    transpose::Transpose,
};

use super::Side;
use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{tensor_grad::Tensor, tensor_op::operation::TensorOperation};
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct TensorScal<T> {
    scale: T,
}

impl<T> TensorScal<T> {
    pub(crate) fn new(scale: T) -> Self {
        TensorScal { scale }
    }
}

impl<T, F> TensorOperation<T, F> for TensorScal<T>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + num::traits::Pow<T, Output = T>
        + Copy
        + One
        + Zero
        + Real
        + 'static,
    F: IFramework + Clone + 'static,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    fn to_string(&self) -> String {
        format!("Scale {}", self.scale)
    }

    fn output_shape(&self, lhs_shape: Option<&[usize]>, rhs_shape: Option<&[usize]>) -> Vec<usize> {
        match lhs_shape {
            Some(lhs_shape) => match rhs_shape {
                None => lhs_shape.to_vec(),
                Some(_rhs_shape) => unreachable!(),
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
        let mut new_tensor = SharedTensor::new(&output_shape);
        let identity_shape = match output_shape.len() {
            2 => *output_shape.iter().last().unwrap(),
            1 => 1,
            _ => unimplemented!(),
        };
        (**backend)
            .gemm(
                &filled_tensor(backend, &[1], &[self.scale]),
                Transpose::NoTrans,
                &lhs.get_tensor().tensor,
                Transpose::NoTrans,
                &identity(identity_shape, &backend).get_tensor().tensor,
                &zeros(backend, &[1]),
                &mut new_tensor,
            )
            .unwrap();
        let output = Tensor::new(new_tensor, output_shape, backend);
        output
    }

    fn grad(
        &self,
        seed: TensorIRStruct<T, F>,
        lhs: TensorInput<T, F>,
        other: TensorInput<T, F>,
        _side: Side,
    ) -> TensorIRStruct<T, F> {
        seed * self.scale
    }
}

#[cfg(test)]
mod test {
    use coaster::Native;

    use crate::utility::{get_native_backend, ones};

    use super::*;

    #[test]
    fn check_scale() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(&backend, &[4], &[1.0, 2.0, 3.0, 4.0]),
            vec![4],
            &backend,
        );
        let mut op = input * 2.0_f64;
        let tensor_scaled = op.evaluate(&backend).read();
        let real_output = vec![2.0, 4.0, 6.0, 8.0_f64];
        assert_eq!(tensor_scaled, real_output);
    }

    #[test]
    fn check_1x1_scale() {
        let backend = Rc::new(get_native_backend());
        let input = ones::<f64, Native>(&backend, &[1]);
        let mut op = input * 1.0_f64;
        let tensor_scaled = op.evaluate(&backend).read();
        let real_output = vec![1.0];
        assert_eq!(tensor_scaled, real_output);
    }

    #[test]
    fn check_2d_scale() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
            vec![10, 1],
            &backend,
        )
        .with_grad();
        let mut op = input * 2.0;
        assert_eq!(op.evaluate(&backend).read()[1], 0.2);
    }

    #[test]
    fn check_2d_scale_trans() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[1, 10],
                &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
            vec![1, 10],
            &backend,
        )
        .with_grad();
        let mut op = input * 2.0;
        assert_eq!(
            op.evaluate(&backend).read(),
            vec![0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
        )
    }
}
