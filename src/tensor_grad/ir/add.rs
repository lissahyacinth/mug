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
        TensorIRStruct, TensorOp,
    },
};

use super::error::IRError;
use crate::tensor_op::{OpSide, TensorInput};

macro_rules! impl_add_for_ir_struct {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Add<$rhs_type> for $lhs_type
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

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let add_type: AddType =
                    find_add_type(self.output_shape(), rhs.output_shape()).unwrap();
                self.clone().merge(
                    rhs.clone(),
                    Box::new(TensorAdd::new(T::one(), T::one(), add_type)),
                )
            }
        }
    };
}

macro_rules! impl_add_for_ir_struct_tensor {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Add<$rhs_type> for $lhs_type
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

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let add_type: AddType = find_add_type(&self.shape(), rhs.output_shape()).unwrap();
                rhs.clone().op_tensor(
                    self.clone(),
                    Box::new(TensorAdd::new(T::one(), T::one(), add_type)),
                    OpSide::Right,
                )
            }
        }
    };
}

macro_rules! impl_add_for_tensor {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Add<$rhs_type> for $lhs_type
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

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let add_type: AddType = find_add_type(&self.shape(), &rhs.shape()).unwrap();
                TensorIRStruct::new(
                    self.clone(),
                    Some(rhs.clone()),
                    Box::new(TensorAdd::new(T::one(), T::one(), add_type)),
                )
            }
        }
    };
}

macro_rules! impl_add_for_tensor_ir_struct {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Add<$rhs_type> for $lhs_type
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

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let add_type: AddType = find_add_type(&self.output_shape(), &rhs.shape()).unwrap();
                self.clone().op_tensor(
                    rhs.clone(),
                    Box::new(TensorAdd::new(T::one(), T::one(), add_type)),
                    OpSide::Left,
                )
            }
        }
    };
}

macro_rules! impl_add_for_op_ir_struct {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> ::std::ops::Add<$rhs_type> for $lhs_type
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

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let add_type: AddType =
                    find_add_type(&self.output_shape(), &rhs.output_shape).unwrap();
                self.clone().op(
                    TensorInput::Op(Box::new(rhs.clone())),
                    Box::new(TensorAdd::new(T::one(), T::one(), add_type)),
                    OpSide::Left,
                )
            }
        }

        impl<T, F> ::std::ops::Add<$lhs_type> for $rhs_type
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

            fn add(self, rhs: $lhs_type) -> Self::Output {
                let add_type: AddType =
                    find_add_type(&self.output_shape, &rhs.output_shape()).unwrap();
                rhs.clone().op(
                    TensorInput::Op(Box::new(self.clone())),
                    Box::new(TensorAdd::new(T::one(), T::one(), add_type)),
                    OpSide::Right,
                )
            }
        }
    };
}

impl_add_for_ir_struct!(TensorIRStruct<T, F>, TensorIRStruct<T, F>);
impl_add_for_ir_struct!(&TensorIRStruct<T, F>, TensorIRStruct<T, F>);
impl_add_for_ir_struct!(TensorIRStruct<T, F>, &TensorIRStruct<T, F>);
impl_add_for_ir_struct!(&TensorIRStruct<T, F>, &TensorIRStruct<T, F>);

impl_add_for_tensor!(ArcTensor<T, F>, ArcTensor<T, F>);
impl_add_for_tensor!(&ArcTensor<T, F>, ArcTensor<T, F>);
impl_add_for_tensor!(ArcTensor<T, F>, &ArcTensor<T, F>);
impl_add_for_tensor!(&ArcTensor<T, F>, &ArcTensor<T, F>);

impl_add_for_ir_struct_tensor!(ArcTensor<T, F>, TensorIRStruct<T, F>);
impl_add_for_ir_struct_tensor!(&ArcTensor<T, F>, TensorIRStruct<T, F>);
impl_add_for_ir_struct_tensor!(ArcTensor<T, F>, &TensorIRStruct<T, F>);
impl_add_for_ir_struct_tensor!(&ArcTensor<T, F>, &TensorIRStruct<T, F>);

impl_add_for_tensor_ir_struct!(TensorIRStruct<T, F>, ArcTensor<T, F>);
impl_add_for_tensor_ir_struct!(&TensorIRStruct<T, F>, ArcTensor<T, F>);
impl_add_for_tensor_ir_struct!(TensorIRStruct<T, F>, &ArcTensor<T, F>);
impl_add_for_tensor_ir_struct!(&TensorIRStruct<T, F>, &ArcTensor<T, F>);

impl_add_for_op_ir_struct!(TensorIRStruct<T, F>, TensorOp<T, F>);
impl_add_for_op_ir_struct!(&TensorIRStruct<T, F>, TensorOp<T, F>);
impl_add_for_op_ir_struct!(TensorIRStruct<T, F>, &TensorOp<T, F>);
impl_add_for_op_ir_struct!(&TensorIRStruct<T, F>, &TensorOp<T, F>);

pub(crate) fn find_add_type(lhs_shape: &[usize], rhs_shape: &[usize]) -> Result<AddType, IRError> {
    if lhs_shape == rhs_shape {
        Ok(AddType::EqualSize)
    } else if lhs_shape.iter().product::<usize>() == 1 {
        Ok(AddType::Singleton(Side::Left))
    } else if rhs_shape.iter().product::<usize>() == 1 {
        Ok(AddType::Singleton(Side::Right))
    } else if lhs_shape.len() != rhs_shape.len() {
        Err(IRError::InvalidInputs {
            ir_name: "Add".to_string(),
            inputs: (lhs_shape.to_vec(), rhs_shape.to_vec()),
        })
    // Likely to be Broadcastable
    } else if lhs_shape
        .iter()
        .zip(rhs_shape.iter())
        .filter(|(a, b)| a != b)
        .filter(|(a, b)| (**a != 1) & (**b != 1))
        .count()
        == 0
    {
        let index = lhs_shape
            .iter()
            .enumerate()
            .zip(rhs_shape.iter())
            .filter(|((_, a), b)| a != b)
            .map(|((index, _), _)| index)
            .next()
            .unwrap();
        if (lhs_shape[index] == 1) & (index == 0) {
            Ok(AddType::BroadcastColumn(Side::Left))
        } else if lhs_shape[index] == 1 {
            Ok(AddType::BroadcastRow(Side::Left))
        } else if index == 0 {
            Ok(AddType::BroadcastColumn(Side::Right))
        } else {
            Ok(AddType::BroadcastRow(Side::Right))
        }
    } else {
        Err(IRError::InvalidInputs {
            ir_name: "Add".to_string(),
            inputs: (lhs_shape.to_vec(), rhs_shape.to_vec()),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn check_add_type_singleton_col() {
        assert_eq!(
            find_add_type(&[1, 1], &[10, 1]).unwrap(),
            AddType::Singleton(Side::Left)
        );
        assert_eq!(
            find_add_type(&[10, 1], &[1, 1]).unwrap(),
            AddType::Singleton(Side::Right)
        );
    }

    #[test]
    fn check_add_type_singleton_row() {
        assert_eq!(
            find_add_type(&[1, 1], &[1, 10]).unwrap(),
            AddType::Singleton(Side::Left)
        );
        assert_eq!(
            find_add_type(&[1, 10], &[1, 1]).unwrap(),
            AddType::Singleton(Side::Right)
        );
    }
}
