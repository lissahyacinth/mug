use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::{arith::Side, chain_gradient::ChainGradient, TensorIRStruct, TensorInput},
    utility::{identity, ones, zeros},
    NumCast,
};
use coaster::{IBackend, IFramework, SharedTensor};
use coaster_blas::{
    plugin::{Asum, Axpy, Gemm},
    transpose::Transpose,
};

use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};
type TensorDims = Vec<usize>;

use crate::{tensor_grad::Tensor, tensor_op::operation::TensorOperation};
use std::rc::Rc;
#[derive(Clone)]
pub(crate) struct TensorTranspose {}

impl TensorTranspose {
    pub(crate) fn new() -> Self {
        TensorTranspose {}
    }
}

impl<T, F> TensorOperation<T, F> for TensorTranspose
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
        "Transpose".to_string()
    }

    fn output_shape(&self, lhs_shape: Option<&[usize]>, rhs_shape: Option<&[usize]>) -> Vec<usize> {
        match lhs_shape {
            Some(lhs_shape) => match rhs_shape {
                None => match lhs_shape.len() {
                    1 => lhs_shape.to_vec(),
                    2 => {
                        let mut lhs_internal = lhs_shape.to_vec();
                        lhs_internal.reverse();
                        lhs_internal
                    }
                    3 => {
                        let mut lhs_internal = lhs_shape.to_vec();
                        lhs_internal.swap(2, 3);
                        lhs_internal
                    }
                    _ => {
                        panic!("Haven't decided how to represent transpose for 4D Matrices yet.")
                    }
                },
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
        let mut new_tensor = SharedTensor::new(&output_shape);
        let identity_shape = output_shape.iter().last().unwrap();
        (**backend)
            .gemm(
                &ones(backend, &[1]).get_tensor().tensor,
                Transpose::Trans,
                &lhs.get_tensor().tensor,
                Transpose::NoTrans,
                &identity(*identity_shape, backend).get_tensor().tensor,
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
        side: Side,
    ) -> TensorIRStruct<T, F> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::utility::{filled_tensor, get_native_backend};

    #[test]
    fn transpose_2d() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(&backend, &[2, 3], &[1.0_f32, 2., 3., 4., 5., 6.]),
            vec![2, 3],
            &backend,
        );
        println!("{}", input.t().evaluate(&backend));
        assert_eq!(
            input.t().evaluate(&backend).read(),
            vec![1.0_f32, 4., 2., 5., 3., 6.],
        );
        assert_eq!(input.t().output_shape(), vec![3, 2])
    }

    #[test]
    fn transpose_failing() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0.0_f32, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            ),
            vec![10, 1],
            &backend,
        );
        println!("{}", input.t().evaluate(&backend));
        assert_eq!(
            input.t().evaluate(&backend).read(),
            vec![0.0_f32, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
        );
        assert_eq!(input.t().output_shape(), vec![1, 10])
    }

    #[test]
    fn transpose_1d() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(&backend, &[1, 3], &[1.0_f32, 2., 3.]),
            vec![1, 3],
            &backend,
        );
        println!("{}", input.t().evaluate(&backend));
        assert_eq!(input.t().evaluate(&backend).read(), vec![1.0_f32, 2., 3.],);
        assert_eq!(input.t().output_shape(), vec![3, 1])
    }
}
