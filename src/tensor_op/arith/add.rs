use crate::{
    tensor_grad::{
        ir::{ethereal::EtherealTensor, scalar_mul::ScalarMultiplication},
        tensor::{ArcTensor, ReadTensor, Tensor},
    },
    tensor_op::{chain_gradient::ChainGradient, TensorIRStruct, TensorInput},
    utility::{copy_tensor, element_tensor, filled_tensor, identity, ones},
    NumCast,
};
use coaster::{IBackend, IFramework};
use coaster_blas::{
    plugin::{Asum, Axpy, Gemm},
    transpose::Transpose,
};

use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::tensor_op::operation::TensorOperation;
use std::{iter::repeat, mem::swap, ops::Add, rc::Rc};

use super::Side;

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub(crate) enum AddType {
    BroadcastColumn(Side),
    BroadcastRow(Side),
    Singleton(Side),
    EqualSize,
}

#[derive(Clone, Debug)]
pub(crate) struct TensorAdd<T> {
    lhs_scale: T,
    rhs_scale: T,
    add_type: AddType,
}

impl<T> TensorAdd<T> {
    pub(crate) fn new(lhs_scale: T, rhs_scale: T, add_type: AddType) -> Self {
        TensorAdd {
            lhs_scale,
            rhs_scale,
            add_type,
        }
    }
}

enum BroadcastDirection {
    Column,
    Row,
}

fn broadcast_add<T, F>(
    lhs: ArcTensor<T, F>,
    rhs: ArcTensor<T, F>,
    backend: &Rc<coaster::Backend<F>>,
    output_shape: Vec<usize>,
    direction: BroadcastDirection,
) -> ArcTensor<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>
        + Copy,
    Standard: Distribution<T>,
    F: IFramework + Clone + 'static,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    let rhs_raw = &rhs.get_tensor().tensor;
    let column_shape = match lhs.shape().len() {
        2 => lhs.shape()[1],
        1 => lhs.shape()[0],
        _ => unimplemented!(),
    };

    dbg!(&output_shape);
    dbg!(lhs.shape());
    println!("{}", lhs);

    let mut output = match direction {
        BroadcastDirection::Column => {
            let mut output = copy_tensor(&backend, &output_shape, &lhs.get_tensor().tensor);
            (**backend)
                .gemm(
                    &ones::<T, F>(backend, &[1]).get_tensor().tensor,
                    Transpose::NoTrans,
                    rhs_raw,
                    Transpose::NoTrans,
                    &ones::<T, F>(&backend, &[1, column_shape])
                        .get_tensor()
                        .tensor,
                    &ones::<T, F>(backend, &[1]).get_tensor().tensor,
                    &mut output,
                )
                .unwrap();
            output
        }
        BroadcastDirection::Row => unimplemented!(),
    };
    Tensor::new(output, output_shape, backend)
}

fn add_singleton<T, F>(
    lhs: ArcTensor<T, F>,
    singleton: ArcTensor<T, F>,
    _rhs_scaler: T,
    backend: &Rc<coaster::Backend<F>>,
) -> ArcTensor<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>
        + Copy,
    Standard: Distribution<T>,
    F: IFramework + Clone + 'static,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    let output_shape = lhs.shape();
    let column_shape = match lhs.shape().len() {
        2 => lhs.shape()[1],
        1 => lhs.shape()[0],
        _ => unimplemented!(),
    };
    let datapoint = singleton.read().into_iter().next().unwrap();
    let mut output = filled_tensor(
        &backend,
        &output_shape,
        &vec![datapoint; output_shape.iter().product()],
    );
    (**backend)
        .gemm(
            &ones::<T, F>(backend, &[1]).get_tensor().tensor,
            Transpose::NoTrans,
            &lhs.get_tensor().tensor,
            Transpose::NoTrans,
            &identity::<T, F>(column_shape, &backend).get_tensor().tensor,
            &ones::<T, F>(backend, &[1]).get_tensor().tensor,
            &mut output,
        )
        .unwrap();
    Tensor::new(output, lhs.shape().to_vec(), backend)
}

fn add_equal_sized<T, F>(
    lhs: ArcTensor<T, F>,
    rhs: ArcTensor<T, F>,
    rhs_scaler: T,
    backend: &Rc<coaster::Backend<F>>,
) -> ArcTensor<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>
        + Copy,
    Standard: Distribution<T>,
    F: IFramework + Clone + 'static,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    let mut new_tensor = copy_tensor(backend, lhs.shape(), &lhs.get_tensor().tensor);
    (*backend)
        .axpy(
            &element_tensor(backend, rhs_scaler),
            &rhs.get_tensor().tensor,
            &mut new_tensor,
        )
        .unwrap();
    Tensor::new(new_tensor, lhs.shape().to_vec(), backend)
}

impl<T, F> TensorOperation<T, F> for TensorAdd<T>
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
        if self.rhs_scale >= T::zero() {
            "Add".to_string()
        } else {
            "Sub".to_string()
        }
    }

    fn output_shape(&self, lhs_shape: Option<&[usize]>, rhs_shape: Option<&[usize]>) -> Vec<usize> {
        match self.add_type {
            AddType::BroadcastColumn(side) | AddType::BroadcastRow(side) => match side {
                Side::Left => rhs_shape.unwrap().to_vec(),
                Side::Right => lhs_shape.unwrap().to_vec(),
            },
            AddType::EqualSize => lhs_shape.unwrap().to_vec(),
            AddType::Singleton(side) => match side {
                Side::Left => rhs_shape.unwrap().to_vec(),
                Side::Right => lhs_shape.unwrap().to_vec(),
            },
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
        let output_shape = self
            .output_shape(Some(&lhs_shape), Some(&rhs_shape))
            .to_vec();
        match self.add_type {
            AddType::BroadcastColumn(side) => match side {
                Side::Left => {
                    broadcast_add(rhs, lhs, &backend, output_shape, BroadcastDirection::Column)
                }
                Side::Right => {
                    broadcast_add(lhs, rhs, &backend, output_shape, BroadcastDirection::Column)
                }
            },
            AddType::BroadcastRow(side) => match side {
                Side::Left => {
                    broadcast_add(rhs, lhs, &backend, output_shape, BroadcastDirection::Row)
                }
                Side::Right => {
                    broadcast_add(lhs, rhs, &backend, output_shape, BroadcastDirection::Row)
                }
            },
            AddType::EqualSize => add_equal_sized(lhs, rhs, self.rhs_scale, backend),
            AddType::Singleton(side) => match side {
                Side::Left => add_singleton(rhs, lhs, T::one(), backend),
                Side::Right => add_singleton(lhs, rhs, T::one(), backend),
            },
        }
    }

    fn grad(
        &self,
        seed: TensorIRStruct<T, F>,
        lhs: TensorInput<T, F>,
        other: TensorInput<T, F>,
        side: Side,
    ) -> TensorIRStruct<T, F> {
        let lhs_shape = match lhs {
            TensorInput::Op(op) => op.output_shape,
            TensorInput::Tensor(tensor) => tensor.shape().to_vec(),
            TensorInput::EtherealTensor(tensor) => tensor.shape().to_vec(),
            TensorInput::None => unreachable!(),
        };
        let other_shape = match other {
            TensorInput::Op(op) => op.output_shape,
            TensorInput::Tensor(tensor) => tensor.shape().to_vec(),
            TensorInput::EtherealTensor(tensor) => tensor.shape().to_vec(),
            TensorInput::None => unreachable!(),
        };

        dbg!(self.add_type);
        dbg!(side);
        dbg!(&lhs_shape);
        dbg!(&other_shape);
        match self.add_type {
            AddType::BroadcastColumn(broadcast_side) => {
                if side == broadcast_side {
                    let tensor = EtherealTensor::new(
                        vec![T::one(); other_shape.iter().product::<usize>()],
                        vec![other_shape[1], other_shape[0]],
                    );
                    match side {
                        Side::Left => tensor * seed,
                        Side::Right => seed * tensor,
                    }
                } else {
                    seed
                }
            }
            AddType::Singleton(broadcast_side) => {
                if side == broadcast_side {
                    let other_shape = if other_shape.len() == 1 {
                        [other_shape[0], 1].to_vec()
                    } else {
                        other_shape
                    };
                    let tensor = EtherealTensor::new(
                        vec![T::one(); other_shape[1]],
                        vec![other_shape[1], 1],
                    );
                    let tensor_2 = EtherealTensor::new(
                        vec![T::one(); other_shape[0]],
                        vec![other_shape[0], 1],
                    );
                    ((seed * tensor).t()) * tensor_2
                } else {
                    seed
                }
            }
            _ => seed,
        }
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
    fn check_addition_a_b() {
        let backend = Rc::new(get_native_backend());
        let input = create_2d_array();
        let input_sum = (input.clone() + input.clone()).sum();
        let tensor_a = Tensor::from_array(input.clone(), &backend);
        let tensor_b = Tensor::from_array(input.clone(), &backend);
        let mut tensor_res = (tensor_a + tensor_b).sum();
        print!("Graph Output: \n{}", tensor_res.graph());
        let tensor_sum = match tensor_res.evaluate(&backend).to_ndarray() {
            ArrayOutput::Tensor0D(_) => 0.0f32,
            ArrayOutput::Tensor1D(array) => array.sum(),
            ArrayOutput::Tensor2D(array) => array.sum(),
        };
        assert_eq!(input_sum, tensor_sum);
    }

    #[test]
    fn check_sub_a_b() {
        let backend = Rc::new(get_native_backend());
        let res = (Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[1.0_f32, 0., 1.0, 0., 1., 1.0_f32, 0., 1.0, 0., 1.],
            ),
            vec![10, 1],
            &backend,
        ) - Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[1.0_f32, 0., 1.0, 0., 1., 1.0_f32, 0., 1.0, 0., 1.],
            ),
            vec![10, 1],
            &backend,
        ))
        .evaluate(&backend);
        assert_eq!(res.read(), vec![0.0_f32; 10]);
    }

    #[test]
    fn check_addition_self() {
        let backend = Rc::new(get_native_backend());
        let input = create_2d_array();
        let input_sum = (input.clone() + input.clone()).sum();
        let tensor_a = Tensor::from_array(input.clone(), &backend);
        let mut tensor_res = (&tensor_a + &tensor_a).sum();
        let tensor_sum = match tensor_res.evaluate(&backend).to_ndarray() {
            ArrayOutput::Tensor0D(_) => 0.0f32,
            ArrayOutput::Tensor1D(array) => array.sum(),
            ArrayOutput::Tensor2D(array) => array.sum(),
        };
        assert_eq!(input_sum, tensor_sum);
    }

    #[test]
    fn check_addition_broadcast() {
        let backend = Rc::new(get_native_backend());
        let lhs_matrix = Tensor::new(
            filled_tensor(&backend, &[2, 3], &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
            &backend,
        );
        let rhs_vector = Tensor::new(
            filled_tensor(&backend, &[2, 1], &[1.0_f32, 2.0]),
            vec![2],
            &backend,
        );
        let res = broadcast_add(
            lhs_matrix,
            rhs_vector,
            &backend,
            vec![2, 3],
            BroadcastDirection::Column,
        );
        assert_eq!(res.read(), vec![2.0f32, 3.0, 4.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn check_1x1_addition_broadcast() {
        let backend = Rc::new(get_native_backend());
        let lhs_matrix = Tensor::new(
            filled_tensor(&backend, &[2, 3], &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
            &backend,
        );
        let rhs_vector = Tensor::new(
            filled_tensor(&backend, &[1, 1], &[1.0_f32]),
            vec![1, 1],
            &backend,
        );
        let mut res = &lhs_matrix + &rhs_vector;
        println!("LHS Input {}", lhs_matrix);
        println!("RHS Input {}", rhs_vector);
        println!("Output: {}", res.evaluate(&backend));
        assert_eq!(
            res.evaluate(&backend).read(),
            vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn check_addition_cuda() {
        let backend = Rc::new(get_cuda_backend());
        let input = create_2d_array();
        let input_sum = (input.clone() + input.clone()).sum();
        let tensor_a = Tensor::from_array(input.clone(), &backend);
        let mut tensor_res = (&tensor_a + &tensor_a).sum();
        let tensor_sum = match tensor_res.evaluate(&backend).to_ndarray() {
            ArrayOutput::Tensor0D(_) => 0.0f32,
            ArrayOutput::Tensor1D(array) => array.sum(),
            ArrayOutput::Tensor2D(array) => array.sum(),
        };
        assert_eq!(input_sum, tensor_sum);
    }
}
