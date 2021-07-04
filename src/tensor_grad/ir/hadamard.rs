use co::{plugin::numeric_helpers::NumCast, prelude::*};
use coaster as co;
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::{
        arith::multiply::{MultiplicationType, TensorMul},
        TensorIRStruct,
    },
};

use crate::tensor_op::OpSide;

pub trait Hadamard<T, F, Rhs = Self>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>,
    F: IFramework + Clone + 'static,
    Standard: Distribution<T>,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    type Output;
    fn hadamard(self, other: Rhs) -> TensorIRStruct<T, F>;
}

macro_rules! impl_hadamard_for_op {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> Hadamard<T, F, $rhs_type> for $lhs_type
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

            fn hadamard(self, rhs: $rhs_type) -> Self::Output {
                assert!(suitable_for_hadamard(
                    self.output_shape(),
                    rhs.output_shape()
                ));
                self.clone().merge(
                    rhs.clone(),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        MultiplicationType::Hadamard,
                    )),
                )
            }
        }
    };
}

macro_rules! impl_hadamard_for_op_tensor {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> Hadamard<T, F, $rhs_type> for $lhs_type
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

            fn hadamard(self, rhs: $rhs_type) -> Self::Output {
                assert!(suitable_for_hadamard(&self.shape(), rhs.output_shape()));
                rhs.clone().op_tensor(
                    self.clone(),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        MultiplicationType::Hadamard,
                    )),
                    OpSide::Right,
                )
            }
        }
    };
}

macro_rules! impl_hadamard_for_tensor {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> Hadamard<T, F, $rhs_type> for $lhs_type
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

            fn hadamard(self, rhs: $rhs_type) -> Self::Output {
                assert!(suitable_for_hadamard(&self.shape(), &rhs.shape()));
                TensorIRStruct::new(
                    self.clone(),
                    Some(rhs.clone()),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        MultiplicationType::Hadamard,
                    )),
                )
            }
        }
    };
}

macro_rules! impl_hadamard_for_tensor_op {
    ($lhs_type:ty, $rhs_type:ty) => {
        impl<T, F> Hadamard<T, F, $rhs_type> for $lhs_type
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

            fn hadamard(self, rhs: $rhs_type) -> Self::Output {
                assert!(suitable_for_hadamard(&self.output_shape(), &rhs.shape()));
                self.clone().op_tensor(
                    rhs.clone(),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        MultiplicationType::Hadamard,
                    )),
                    OpSide::Left,
                )
            }
        }
    };
}

impl_hadamard_for_op!(TensorIRStruct<T, F>, TensorIRStruct<T, F>);
impl_hadamard_for_op!(&TensorIRStruct<T, F>, TensorIRStruct<T, F>);
impl_hadamard_for_op!(TensorIRStruct<T, F>, &TensorIRStruct<T, F>);
impl_hadamard_for_op!(&TensorIRStruct<T, F>, &TensorIRStruct<T, F>);

impl_hadamard_for_tensor!(ArcTensor<T, F>, ArcTensor<T, F>);
impl_hadamard_for_tensor!(&ArcTensor<T, F>, ArcTensor<T, F>);
impl_hadamard_for_tensor!(ArcTensor<T, F>, &ArcTensor<T, F>);
impl_hadamard_for_tensor!(&ArcTensor<T, F>, &ArcTensor<T, F>);

impl_hadamard_for_op_tensor!(ArcTensor<T, F>, TensorIRStruct<T, F>);
impl_hadamard_for_op_tensor!(&ArcTensor<T, F>, TensorIRStruct<T, F>);
impl_hadamard_for_op_tensor!(ArcTensor<T, F>, &TensorIRStruct<T, F>);
impl_hadamard_for_op_tensor!(&ArcTensor<T, F>, &TensorIRStruct<T, F>);

impl_hadamard_for_tensor_op!(TensorIRStruct<T, F>, ArcTensor<T, F>);
impl_hadamard_for_tensor_op!(&TensorIRStruct<T, F>, ArcTensor<T, F>);
impl_hadamard_for_tensor_op!(TensorIRStruct<T, F>, &ArcTensor<T, F>);
impl_hadamard_for_tensor_op!(&TensorIRStruct<T, F>, &ArcTensor<T, F>);

fn suitable_for_hadamard(lhs_size: &[usize], rhs_size: &[usize]) -> bool {
    lhs_size == rhs_size
}
