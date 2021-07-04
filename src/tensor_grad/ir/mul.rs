use std::usize;

use co::{plugin::numeric_helpers::NumCast, prelude::*};
use coaster as co;
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    tensor_grad::{
        tensor::{ArcTensor, ReadTensor},
        tensor_ir::TensorIR,
    },
    tensor_op::{
        arith::{multiply::MultiplicationType, Side},
        TensorIRStruct, TensorOp,
    },
};

use crate::{
    tensor_grad::ir::error::IRError,
    tensor_op::{arith::multiply::TensorMul, OpSide, TensorInput},
};

use super::{ethereal::EtherealTensor, transpose_shape};

pub(crate) fn find_mul_type(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    lhs_transpose: bool,
    rhs_transpose: bool,
) -> Result<MultiplicationType, IRError> {
    let lhs_shape = if lhs_transpose {
        transpose_shape(lhs_shape)
    } else {
        lhs_shape.to_vec()
    };
    let rhs_shape = if rhs_transpose {
        transpose_shape(rhs_shape)
    } else {
        rhs_shape.to_vec()
    };
    dbg!(&lhs_shape);
    dbg!(&rhs_shape);
    dbg!(suitable_for_dot(&lhs_shape, &rhs_shape));
    dbg!(suitable_for_hadamard(&lhs_shape, &rhs_shape));
    dbg!(suitable_for_broadcast(&lhs_shape, &rhs_shape));
    if suitable_for_dot(&lhs_shape, &rhs_shape) {
        Ok(MultiplicationType::Dot)
    } else if suitable_for_hadamard(&lhs_shape, &rhs_shape) {
        Ok(MultiplicationType::Hadamard)
    } else if suitable_for_broadcast(&lhs_shape, &rhs_shape) {
        if lhs_shape.len() < rhs_shape.len() {
            Ok(MultiplicationType::Broadcast(Side::Left))
        } else if rhs_shape.len() < lhs_shape.len() {
            Ok(MultiplicationType::Broadcast(Side::Right))
        } else {
            let modified_dimension = lhs_shape
                .iter()
                .enumerate()
                .zip(rhs_shape.iter())
                .filter(|((_, lhs), rhs)| lhs != rhs)
                .map(|((idx, _), _)| idx)
                .next()
                .unwrap();
            if lhs_shape[modified_dimension] == 1 {
                Ok(MultiplicationType::Broadcast(Side::Left))
            } else {
                Ok(MultiplicationType::Broadcast(Side::Right))
            }
        }
    } else {
        Err(IRError::InvalidInputs {
            ir_name: "Multiplication".to_string(),
            inputs: (lhs_shape.to_vec(), rhs_shape.to_vec()),
        })
    }
}

pub(crate) fn suitable_for_hadamard(lhs_shape: &[usize], rhs_shape: &[usize]) -> bool {
    (lhs_shape.len() == rhs_shape.len())
        && (lhs_shape.iter().zip(rhs_shape.iter()).all(|(a, b)| a == b))
}

/// Broadcasting is possible when;
/// * One Shape is a singleton - effectively a scaling operation
/// * One Shape fits neatly within the other - i.e. [3x3] * [1x3]
pub(crate) fn suitable_for_broadcast(lhs_shape: &[usize], rhs_shape: &[usize]) -> bool {
    (lhs_shape != rhs_shape)
        & (
            // Singleton Broadcast
            ((lhs_shape.len() == 1) | (rhs_shape.len() == 1)) |
        // Match in all but one dimension
        (
            (lhs_shape
                .iter()
                .filter(|x| !rhs_shape.contains(*x))
                .count() == rhs_shape.len() - 1)
            |
            (rhs_shape
                .iter()
                .filter(|x| !lhs_shape.contains(*x))
                .count() == lhs_shape.len() - 1)
        )
        )
}

/// Identify if the dot product of two matrices is calcuable
pub(crate) fn suitable_for_dot(lhs_shape: &[usize], rhs_shape: &[usize]) -> bool {
    (lhs_shape.len() == rhs_shape.len()) & (lhs_shape.last().unwrap() == rhs_shape.first().unwrap())
}

macro_rules! impl_mul_for_op {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Mul<$rhs_type> for $lhs_type
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

            fn mul(self, rhs: $rhs_type) -> Self::Output {
                let mul_type: MultiplicationType =
                    find_mul_type(self.output_shape(), rhs.output_shape(), false, false).unwrap();
                self.clone().merge(
                    rhs.clone(),
                    Box::new(TensorMul::new(T::one(), false, T::one(), false, mul_type)),
                )
            }
        }
    };
}

macro_rules! impl_mul_for_op_tensor {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Mul<$rhs_type> for $lhs_type
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

            fn mul(self, rhs: $rhs_type) -> Self::Output {
                let mul_type: MultiplicationType =
                    find_mul_type(self.shape(), rhs.output_shape(), false, false).unwrap();
                rhs.clone().op_tensor(
                    self.clone(),
                    Box::new(TensorMul::new(T::one(), false, T::one(), false, mul_type)),
                    OpSide::Left,
                )
            }
        }

        impl<T, F> ::std::ops::Mul<$lhs_type> for $rhs_type
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

            fn mul(self, rhs: $lhs_type) -> Self::Output {
                let mul_type: MultiplicationType =
                    find_mul_type(&self.output_shape(), &rhs.shape(), false, false).unwrap();
                self.clone().op_tensor(
                    rhs.clone(),
                    Box::new(TensorMul::new(T::one(), false, T::one(), false, mul_type)),
                    OpSide::Right,
                )
            }
        }
    };
}

macro_rules! impl_mul_for_tensor {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Mul<$rhs_type> for $lhs_type
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

            fn mul(self, rhs: $rhs_type) -> Self::Output {
                let mul_type: MultiplicationType =
                    find_mul_type(&self.shape(), &rhs.shape(), false, false).unwrap();
                TensorIRStruct::new(
                    self.clone(),
                    Some(rhs.clone()),
                    Box::new(TensorMul::new(T::one(), false, T::one(), false, mul_type)),
                )
            }
        }
    };
}

macro_rules! impl_mul_for_op_ir_struct {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Mul<$rhs_type> for $lhs_type
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

            fn mul(self, rhs: $rhs_type) -> Self::Output {
                let mul_type: MultiplicationType =
                    find_mul_type(&self.output_shape(), &rhs.output_shape, false, false).unwrap();
                self.clone().op(
                    TensorInput::Op(rhs.clone()),
                    Box::new(TensorMul::new(T::one(), false, T::one(), false, mul_type)),
                    OpSide::Left,
                )
            }
        }

        impl<T, F> ::std::ops::Mul<$lhs_type> for $rhs_type
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

            fn mul(self, rhs: $lhs_type) -> Self::Output {
                let mul_type: MultiplicationType =
                    find_mul_type(&self.output_shape, &rhs.output_shape(), false, false).unwrap();
                rhs.clone().op(
                    TensorInput::Op(self.clone()),
                    Box::new(TensorMul::new(T::one(), false, T::one(), false, mul_type)),
                    OpSide::Right,
                )
            }
        }
    };
}

macro_rules! impl_mul_for_tensor_ir_struct {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Mul<$rhs_type> for $lhs_type
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

            fn mul(self, rhs: $rhs_type) -> Self::Output {
                let mul_type: MultiplicationType =
                    find_mul_type(&self.output_shape(), &rhs.shape(), false, false).unwrap();
                self.clone().op(
                    TensorInput::Tensor(rhs.clone()),
                    Box::new(TensorMul::new(T::one(), false, T::one(), false, mul_type)),
                    OpSide::Left,
                )
            }
        }

        impl<T, F> ::std::ops::Mul<$lhs_type> for $rhs_type
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

            fn mul(self, rhs: $lhs_type) -> Self::Output {
                let mul_type: MultiplicationType =
                    find_mul_type(&self.shape(), &rhs.output_shape(), false, false).unwrap();
                rhs.clone().op(
                    TensorInput::Tensor(self.clone()),
                    Box::new(TensorMul::new(T::one(), false, T::one(), false, mul_type)),
                    OpSide::Right,
                )
            }
        }
    };
}

macro_rules! impl_mul_for_eth_tensor_ir_struct {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Mul<$rhs_type> for $lhs_type
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

            fn mul(self, rhs: $rhs_type) -> Self::Output {
                let mul_type: MultiplicationType =
                    find_mul_type(&self.output_shape(), &rhs.shape(), false, false).unwrap();
                self.clone().op(
                    TensorInput::EtherealTensor(rhs.clone()),
                    Box::new(TensorMul::new(T::one(), false, T::one(), false, mul_type)),
                    OpSide::Right,
                )
            }
        }

        impl<T, F> ::std::ops::Mul<$lhs_type> for $rhs_type
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

            fn mul(self, rhs: $lhs_type) -> Self::Output {
                println!("Mul LHS Shape {:?}", &self.shape());
                println!("Mul RHS Shape {:?}", &rhs.output_shape());
                let mul_type: MultiplicationType =
                    find_mul_type(&self.shape(), &rhs.output_shape(), false, false).unwrap();
                dbg!(mul_type);

                rhs.clone().op(
                    TensorInput::EtherealTensor(self.clone()),
                    Box::new(TensorMul::new(T::one(), false, T::one(), false, mul_type)),
                    OpSide::Left,
                )
            }
        }
    };
}

impl_mul_for_op!(TensorIRStruct<T, F>, TensorIRStruct<T, F>);
impl_mul_for_op!(&TensorIRStruct<T, F>, TensorIRStruct<T, F>);
impl_mul_for_op!(TensorIRStruct<T, F>, &TensorIRStruct<T, F>);
impl_mul_for_op!(&TensorIRStruct<T, F>, &TensorIRStruct<T, F>);

impl_mul_for_tensor!(ArcTensor<T, F>, ArcTensor<T, F>);
impl_mul_for_tensor!(&ArcTensor<T, F>, ArcTensor<T, F>);
impl_mul_for_tensor!(ArcTensor<T, F>, &ArcTensor<T, F>);
impl_mul_for_tensor!(&ArcTensor<T, F>, &ArcTensor<T, F>);

impl_mul_for_op_tensor!(ArcTensor<T, F>, TensorIRStruct<T, F>);
impl_mul_for_op_tensor!(&ArcTensor<T, F>, TensorIRStruct<T, F>);
impl_mul_for_op_tensor!(ArcTensor<T, F>, &TensorIRStruct<T, F>);
impl_mul_for_op_tensor!(&ArcTensor<T, F>, &TensorIRStruct<T, F>);

impl_mul_for_op_ir_struct!(TensorIRStruct<T, F>, Box<TensorOp<T, F>>);
impl_mul_for_op_ir_struct!(&TensorIRStruct<T, F>, Box<TensorOp<T, F>>);
impl_mul_for_op_ir_struct!(TensorIRStruct<T, F>, &Box<TensorOp<T, F>>);
impl_mul_for_op_ir_struct!(&TensorIRStruct<T, F>, &Box<TensorOp<T, F>>);

impl_mul_for_tensor_ir_struct!(TensorIRStruct<T, F>, TensorIR<T, F>);
impl_mul_for_tensor_ir_struct!(&TensorIRStruct<T, F>, TensorIR<T, F>);
impl_mul_for_tensor_ir_struct!(TensorIRStruct<T, F>, &TensorIR<T, F>);
impl_mul_for_tensor_ir_struct!(&TensorIRStruct<T, F>, &TensorIR<T, F>);

impl_mul_for_eth_tensor_ir_struct!(TensorIRStruct<T, F>, EtherealTensor<T, F>);
impl_mul_for_eth_tensor_ir_struct!(&TensorIRStruct<T, F>, EtherealTensor<T, F>);
impl_mul_for_eth_tensor_ir_struct!(TensorIRStruct<T, F>, &EtherealTensor<T, F>);
impl_mul_for_eth_tensor_ir_struct!(&TensorIRStruct<T, F>, &EtherealTensor<T, F>);

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn check_mul_type_hadamard() {
        assert_eq!(
            find_mul_type(&[10, 7], &[10, 7], false, false).unwrap(),
            MultiplicationType::Hadamard
        );
    }

    #[test]
    fn check_mul_type_broadcast() {
        assert_eq!(
            find_mul_type(&[10, 7], &[10, 1], false, false).unwrap(),
            MultiplicationType::Broadcast(Side::Right)
        );
    }

    #[test]
    fn check_mul_type_dot() {
        assert_eq!(
            find_mul_type(&[10, 7], &[10, 1], false, false).unwrap(),
            MultiplicationType::Broadcast(Side::Right)
        );
    }

    #[test]
    fn check_mul_type_dot_broadcast() {
        assert_eq!(
            find_mul_type(&[1, 10], &[1, 1], false, false).unwrap(),
            MultiplicationType::Broadcast(Side::Right)
        );
    }

    #[test]
    fn check_mul_type_singleton() {
        assert_eq!(
            find_mul_type(&[10, 7], &[1], false, false).unwrap(),
            MultiplicationType::Broadcast(Side::Right)
        );
    }
}
