use crate::{
    tensor_grad::{
        ir::mul::{suitable_for_broadcast, suitable_for_dot, suitable_for_hadamard},
        tensor::{ArcTensor, ReadTensor},
    },
    tensor_op::{chain_gradient::ChainGradient, TensorIRStruct, TensorInput},
    utility::{filled_tensor, ones, zeros},
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
use rust_blas::math::Trans;

use crate::{tensor_grad::Tensor, tensor_op::operation::TensorOperation};
use std::rc::Rc;

use super::Side;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub(crate) enum MultiplicationType {
    Dot,
    Broadcast(Side),
    Hadamard,
}

#[derive(Clone)]
pub(crate) struct TensorMul<T> {
    scale_output: T,
    transpose_lhs: bool,
    scale_input: T,
    transpose_rhs: bool,
    multiplication_type: MultiplicationType,
}

impl<T> TensorMul<T> {
    pub(crate) fn new(
        scale_output: T,
        transpose_lhs: bool,
        scale_input: T,
        transpose_rhs: bool,
        multiplication_type: MultiplicationType,
    ) -> Self {
        TensorMul {
            scale_output,
            transpose_lhs,
            scale_input,
            transpose_rhs,
            multiplication_type,
        }
    }
}

fn hadamard<T, F>(
    lhs: ArcTensor<T, F>,
    rhs: ArcTensor<T, F>,
    backend: &Rc<coaster::Backend<F>>,
) -> ArcTensor<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + num::traits::Pow<T, Output = T>
        + Copy
        + Real
        + 'static,
    F: Clone + IFramework,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    let raw_res = lhs
        .read()
        .into_iter()
        .zip(rhs.read().into_iter())
        .map(|(a, b)| a * b)
        .collect::<Vec<T>>();
    dbg!(lhs.read().into_iter());
    dbg!(rhs.read().into_iter());
    dbg!(&raw_res);
    Tensor::new(
        filled_tensor(backend, lhs.shape(), &raw_res),
        lhs.shape().to_vec(),
        backend,
    )
}

/// Multiply LHS by a Broadcasted RHS
fn broadcast_dot<T, F>(
    lhs: ArcTensor<T, F>,
    rhs: ArcTensor<T, F>,
    backend: &Rc<coaster::Backend<F>>,
    output_shape: Vec<usize>,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> ArcTensor<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + num::traits::Pow<T, Output = T>
        + Copy
        + Real,
    F: Clone + IFramework,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    let mut new_tensor = SharedTensor::new(&output_shape);
    (**backend)
        .gemm(
            &ones(backend, &[1]).get_tensor().tensor,
            if transpose_lhs {
                Transpose::Trans
            } else {
                Transpose::NoTrans
            },
            &lhs.get_tensor().tensor,
            if transpose_rhs {
                Transpose::Trans
            } else {
                Transpose::NoTrans
            },
            &rhs.get_tensor().tensor,
            &zeros(backend, &[1]),
            &mut new_tensor,
        )
        .unwrap();
    Tensor::new(new_tensor, output_shape, backend)
}

fn dot<T, F>(
    lhs: ArcTensor<T, F>,
    transpose_lhs: bool,
    rhs: ArcTensor<T, F>,
    transpose_rhs: bool,
    backend: &Rc<coaster::Backend<F>>,
    output_shape: Vec<usize>,
) -> ArcTensor<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + num::traits::Pow<T, Output = T>
        + Copy
        + Real,
    F: Clone + IFramework,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    if lhs.shape().last().unwrap() != rhs.shape().first().unwrap() {
        panic!(
            "LHS Tensor ({:?}) cannot be multiplied by RHS Tensor ({:?})",
            lhs.shape(),
            rhs.shape()
        )
    }
    let mut new_tensor = SharedTensor::new(&output_shape);
    (**backend)
        .gemm(
            &ones(backend, &[1]).get_tensor().tensor,
            if transpose_lhs {
                Transpose::Trans
            } else {
                Transpose::NoTrans
            },
            &lhs.get_tensor().tensor,
            if transpose_rhs {
                Transpose::Trans
            } else {
                Transpose::NoTrans
            },
            &rhs.get_tensor().tensor,
            &zeros(backend, &[1]),
            &mut new_tensor,
        )
        .unwrap();
    let output = Tensor::new(new_tensor, output_shape, backend);
    output
}

impl<T, F> TensorOperation<T, F> for TensorMul<T>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + num::traits::Pow<T, Output = T>
        + Copy
        + Real
        + 'static,
    F: Clone + IFramework + 'static,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    fn to_string(&self) -> String {
        format!(
            "A{} Mul B{}",
            if self.transpose_lhs { "ᵀ" } else { "" },
            if self.transpose_rhs { "ᵀ" } else { "" }
        )
    }

    fn output_shape(&self, lhs_shape: Option<&[usize]>, rhs_shape: Option<&[usize]>) -> Vec<usize> {
        match lhs_shape {
            Some(lhs_shape) => match rhs_shape {
                Some(rhs_shape) => {
                    let lhs_shape = if self.transpose_lhs {
                        lhs_shape.into_iter().rev().cloned().collect()
                    } else {
                        lhs_shape.to_vec()
                    };
                    let rhs_shape = if self.transpose_rhs {
                        rhs_shape.into_iter().rev().cloned().collect()
                    } else {
                        rhs_shape.to_vec()
                    };
                    if suitable_for_dot(&lhs_shape, &rhs_shape) {
                        let mut output_shape: Vec<usize> = lhs_shape.to_vec();
                        if output_shape.len() > 1 {
                            let _ = output_shape.pop().unwrap();
                        }
                        match rhs_shape.len() {
                            2 => {
                                output_shape.push(*rhs_shape.iter().skip(1).next().unwrap());
                            }
                            1 => output_shape.push(1),
                            _ => unimplemented!(),
                        }
                        output_shape
                    } else if suitable_for_hadamard(&lhs_shape, &rhs_shape) {
                        lhs_shape.to_vec()
                    } else if suitable_for_broadcast(&lhs_shape, &rhs_shape) {
                        if lhs_shape.clone().into_iter().product::<usize>() == 1 {
                            rhs_shape.to_vec()
                        } else {
                            lhs_shape.to_vec()
                        }
                    } else {
                        unreachable!()
                    }
                }
                None => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    fn evaluate(
        &self,
        lhs: Option<ArcTensor<T, F>>,
        rhs: Option<ArcTensor<T, F>>,
        backend: &Rc<coaster::Backend<F>>,
    ) -> ArcTensor<T, F> {
        let lhs = lhs.unwrap();
        let rhs = rhs.unwrap();
        let lhs_shape = lhs.shape().to_vec();
        let rhs_shape = rhs.shape().to_vec();
        dbg!(&lhs_shape);
        dbg!(&rhs_shape);
        dbg!(self.multiplication_type);
        let output_shape = self.output_shape(Some(&lhs_shape), Some(&rhs_shape));
        dbg!(&output_shape);
        match self.multiplication_type {
            MultiplicationType::Dot => {
                assert_eq!(lhs_shape.last().unwrap(), rhs_shape.first().unwrap());
                dot(
                    lhs,
                    self.transpose_lhs,
                    rhs,
                    self.transpose_rhs,
                    backend,
                    output_shape,
                )
            }
            MultiplicationType::Broadcast(side) => match side {
                Side::Left => {
                    let output_shape = self.output_shape(Some(&lhs_shape), Some(&rhs_shape));
                    broadcast_dot(
                        lhs,
                        rhs,
                        backend,
                        output_shape,
                        self.transpose_lhs,
                        self.transpose_rhs,
                    )
                }
                Side::Right => {
                    let output_shape = self.output_shape(Some(&rhs_shape), Some(&lhs_shape));
                    broadcast_dot(
                        rhs,
                        lhs,
                        backend,
                        output_shape,
                        self.transpose_rhs,
                        self.transpose_lhs,
                    )
                }
            },
            MultiplicationType::Hadamard => {
                assert_eq!(lhs_shape, rhs_shape);
                hadamard(lhs, rhs, backend)
            }
        }
    }

    fn grad(
        &self,
        seed: TensorIRStruct<T, F>,
        lhs: TensorInput<T, F>,
        other: TensorInput<T, F>,
        side: Side,
    ) -> TensorIRStruct<T, F> {
        let grad = match lhs {
            TensorInput::Op(_lhs_op) => match other {
                TensorInput::Op(other_op) => Box::new(Box::new(*other_op).t()),
                TensorInput::Tensor(other_tensor) => Box::new(other_tensor.t()),
                TensorInput::EtherealTensor(other_e_tensor) => Box::new(other_e_tensor.t()),
                TensorInput::None => unreachable!(),
            },
            TensorInput::Tensor(_lhs_tensor) => match other {
                TensorInput::Op(other_op) => Box::new(Box::new(*other_op).t()),
                TensorInput::Tensor(other_tensor) => Box::new(other_tensor.t()),
                TensorInput::EtherealTensor(other_e_tensor) => Box::new(other_e_tensor.t()),
                TensorInput::None => unreachable!(),
            },
            TensorInput::EtherealTensor(_e_tensor) => match other {
                TensorInput::Op(other_op) => Box::new(Box::new(*other_op).t()),
                TensorInput::Tensor(other_tensor) => Box::new(other_tensor.t()),
                TensorInput::EtherealTensor(other_e_tensor) => Box::new(other_e_tensor.t()),
                TensorInput::None => unreachable!(),
            },
            TensorInput::None => unreachable!(),
        };
        match side {
            Side::Left => grad * seed,
            Side::Right => seed * grad,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{tensor_grad::ir::hadamard::Hadamard, utility::get_native_backend};

    use coaster::frameworks::cuda::get_cuda_backend;
    use ndarray::prelude::*;

    fn create_2d_array() -> Array2<f32> {
        return array![[1., 2., 3.], [4., 5., 6.],];
    }

    #[test]
    fn check_multiply_self() {
        let backend = Rc::new(get_native_backend());
        let input = create_2d_array();
        let inner_product = input.clone().dot(&input.clone().t()).into_raw_vec();
        let tensor_a = Tensor::from_array(input.clone(), &backend);
        let mut tensor_res = &tensor_a * (tensor_a.t());
        let tensor_sum = tensor_res.evaluate(&backend).read();
        assert_eq!(inner_product, tensor_sum);
    }

    #[test]
    fn check_multiply_dims() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(filled_tensor(&backend, &[1], &[1.0]), vec![1], &backend);
        let weight = Tensor::new(
            filled_tensor(
                &backend,
                &[1, 10],
                &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
            vec![1, 10],
            &backend,
        );
        let mut tensor_res = input * weight;
        println!("{}", tensor_res.evaluate(&backend));
    }

    #[test]
    fn check_multiply_hadamard() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(&backend, &[2, 2], &[1.0_f32, 2., 3., 4.]),
            vec![2, 2],
            &backend,
        );
        let mut res = (&input).hadamard(&input);
        let tensor_sum = res.evaluate(&backend).read();
        assert_eq!(vec![1.0_f32, 4., 9., 16.], tensor_sum);
    }

    #[test]
    fn check_multiply_2x2_broadcast() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            ),
            vec![10, 1],
            &backend,
        );
        let weight = Tensor::new(
            filled_tensor(&backend, &[1, 1], &[2.0_f32]),
            vec![1, 1],
            &backend,
        );
        let mut res = (&input) * (&weight);
        let res = res.evaluate(&backend).read();
        assert_eq!(
            vec![1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.]
                .into_iter()
                .map(|x| 2.0 * x)
                .collect::<Vec<f32>>(),
            res
        );
    }

    #[test]
    fn check_multiply_2x1_broadcast() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            ),
            vec![10, 1],
            &backend,
        );
        let weight = Tensor::new(
            filled_tensor(&backend, &[1, 1], &[2.0_f32]),
            vec![1],
            &backend,
        );
        let mut res = (&input) * (&weight);
        let res = res.evaluate(&backend).read();
        assert_eq!(
            vec![1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.]
                .into_iter()
                .map(|x| 2.0 * x)
                .collect::<Vec<f32>>(),
            res
        );
    }

    #[test]
    fn check_multiply_1x2_transpose_broadcast() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[1, 10],
                &[1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            ),
            vec![1, 10],
            &backend,
        )
        .t();
        let weight = Tensor::new(
            filled_tensor(&backend, &[1, 1], &[2.0_f32]),
            vec![1],
            &backend,
        );
        let mut res = (&input) * (&weight);
        let res = res.evaluate(&backend).read();
        assert_eq!(
            vec![1.0_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.]
                .into_iter()
                .map(|x| 2.0 * x)
                .collect::<Vec<f32>>(),
            res
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn check_multiply_cuda() {
        let backend = Rc::new(get_cuda_backend());
        let input = array![[1., 2., 3.], [4., 5., 6.],];
        let input_2 = array![[1., 2., 3.],];
        let inner_product = input.clone().dot(&input_2.clone().t()).into_raw_vec();
        let tensor_a = Tensor::from_array(input.clone(), &backend);
        let tensor_b = Tensor::from_array(input_2.clone(), &backend);
        println!("{}", tensor_b);
        println!("{:?}", tensor_b.shape());
        let mut tensor_res = &tensor_a * (tensor_b.t());
        let tensor_sum = tensor_res.evaluate(&backend).read();
        println!("{}", tensor_res.evaluate(&backend));
        assert_eq!(inner_product, tensor_sum);
    }
}
