use co::{plugin::numeric_helpers::NumCast, prelude::*};
use coaster as co;
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::{
        arith::{
            add::{AddType, TensorAdd},
            Side,
        },
        TensorIRStruct,
    },
};

use super::error::IRError;
use crate::tensor_op::OpSide;

macro_rules! impl_sub_for_op {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Sub<$rhs_type> for $lhs_type
        where
            T: std::fmt::Debug
                + std::fmt::Display
                + Copy
                + NumCast
                + One
                + Zero
                + Real
                + num::traits::Pow<T, Output = T>
                + 'static,
            F: IFramework + Clone + 'static,
            Standard: Distribution<T>,
            coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
            ArcTensor<T, F>: ReadTensor<T>,
        {
            type Output = TensorIRStruct<T, F>;

            fn sub(self, rhs: $rhs_type) -> Self::Output {
                let sub_type: AddType =
                    find_sub_type(self.output_shape(), rhs.output_shape()).unwrap();
                self.clone().merge(
                    rhs.clone(),
                    Box::new(TensorAdd::new(T::one(), -T::one(), sub_type)),
                )
            }
        }
    };
}

macro_rules! impl_sub_for_op_tensor {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Sub<$rhs_type> for $lhs_type
        where
            T: std::fmt::Debug
                + std::fmt::Display
                + Copy
                + NumCast
                + One
                + Zero
                + Real
                + num::traits::Pow<T, Output = T>
                + 'static,
            F: IFramework + Clone + 'static,
            Standard: Distribution<T>,
            coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
            ArcTensor<T, F>: ReadTensor<T>,
        {
            type Output = TensorIRStruct<T, F>;

            fn sub(self, rhs: $rhs_type) -> Self::Output {
                let sub_type: AddType = find_sub_type(&self.shape(), rhs.output_shape()).unwrap();
                rhs.clone().op_tensor(
                    self.clone(),
                    Box::new(TensorAdd::new(T::one(), -T::one(), sub_type)),
                    OpSide::Right,
                )
            }
        }

        impl<T, F> ::std::ops::Sub<$lhs_type> for $rhs_type
        where
            T: std::fmt::Debug
                + std::fmt::Display
                + Copy
                + NumCast
                + One
                + Zero
                + Real
                + num::traits::Pow<T, Output = T>
                + 'static,
            F: IFramework + Clone + 'static,
            Standard: Distribution<T>,
            coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
            ArcTensor<T, F>: ReadTensor<T>,
        {
            type Output = TensorIRStruct<T, F>;

            fn sub(self, rhs: $lhs_type) -> Self::Output {
                let sub_type: AddType = find_sub_type(&self.output_shape(), &rhs.shape()).unwrap();
                self.clone().op_tensor(
                    rhs.clone(),
                    Box::new(TensorAdd::new(T::one(), -T::one(), sub_type)),
                    OpSide::Right,
                )
            }
        }
    };
}

macro_rules! impl_sub_for_tensor {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Sub<$rhs_type> for $lhs_type
        where
            T: std::fmt::Debug
                + std::fmt::Display
                + Copy
                + NumCast
                + One
                + Zero
                + Real
                + num::traits::Pow<T, Output = T>
                + 'static,
            F: IFramework + Clone + 'static,
            Standard: Distribution<T>,
            coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
            ArcTensor<T, F>: ReadTensor<T>,
        {
            type Output = TensorIRStruct<T, F>;

            fn sub(self, rhs: $rhs_type) -> Self::Output {
                let sub_type: AddType = find_sub_type(&self.shape(), &rhs.shape()).unwrap();
                TensorIRStruct::new(
                    self.clone(),
                    Some(rhs.clone()),
                    Box::new(TensorAdd::new(T::one(), -T::one(), sub_type)),
                )
            }
        }
    };
}

impl_sub_for_op!(TensorIRStruct<T, F>, TensorIRStruct<T, F>);
impl_sub_for_op!(&TensorIRStruct<T, F>, TensorIRStruct<T, F>);
impl_sub_for_op!(TensorIRStruct<T, F>, &TensorIRStruct<T, F>);
impl_sub_for_op!(&TensorIRStruct<T, F>, &TensorIRStruct<T, F>);

impl_sub_for_tensor!(ArcTensor<T, F>, ArcTensor<T, F>);
impl_sub_for_tensor!(&ArcTensor<T, F>, ArcTensor<T, F>);
impl_sub_for_tensor!(ArcTensor<T, F>, &ArcTensor<T, F>);
impl_sub_for_tensor!(&ArcTensor<T, F>, &ArcTensor<T, F>);

impl_sub_for_op_tensor!(ArcTensor<T, F>, TensorIRStruct<T, F>);
impl_sub_for_op_tensor!(&ArcTensor<T, F>, TensorIRStruct<T, F>);
impl_sub_for_op_tensor!(ArcTensor<T, F>, &TensorIRStruct<T, F>);
impl_sub_for_op_tensor!(&ArcTensor<T, F>, &TensorIRStruct<T, F>);
impl_sub_for_op_tensor!(&mut ArcTensor<T, F>, TensorIRStruct<T, F>);
impl_sub_for_op_tensor!(&mut ArcTensor<T, F>, &TensorIRStruct<T, F>);

fn find_sub_type(lhs_shape: &[usize], rhs_shape: &[usize]) -> Result<AddType, IRError> {
    if lhs_shape.len() != rhs_shape.len() {
        Err(IRError::InvalidInputs {
            ir_name: "Sub".to_string(),
            inputs: (lhs_shape.to_vec(), rhs_shape.to_vec()),
        })
    } else if lhs_shape == rhs_shape {
        Ok(AddType::EqualSize)
    } else if lhs_shape
        .iter()
        .zip(rhs_shape.iter())
        .filter(|(a, b)| a != b)
        .filter(|(a, b)| (**a != 1) & (**b != 1))
        .count()
        == 0
    {
        if lhs_shape
            .iter()
            .zip(rhs_shape.iter())
            .filter(|(a, b)| a != b)
            .filter(|(a, _)| **a != 1)
            .count()
            > 0
        {
            match lhs_shape
                .iter()
                .enumerate()
                .zip(rhs_shape.iter())
                .filter(|((_, a), b)| a != b)
                .map(|((index, _), _)| index)
                .next()
                .unwrap()
            {
                0 => Ok(AddType::BroadcastColumn(Side::Right)),
                1 => Ok(AddType::BroadcastRow(Side::Right)),
                _ => unimplemented!(),
            }
        } else {
            match rhs_shape
                .iter()
                .enumerate()
                .zip(lhs_shape.iter())
                .filter(|((_, a), b)| a != b)
                .map(|((index, _), _)| index)
                .next()
                .unwrap()
            {
                0 => Ok(AddType::BroadcastColumn(Side::Left)),
                1 => Ok(AddType::BroadcastRow(Side::Left)),
                _ => unimplemented!(),
            }
        }
    } else {
        Err(IRError::InvalidInputs {
            ir_name: "Sub".to_string(),
            inputs: (lhs_shape.to_vec(), rhs_shape.to_vec()),
        })
    }
}
