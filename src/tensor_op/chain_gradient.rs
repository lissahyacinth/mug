use coaster::{IBackend, IFramework};
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{NumCast, One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::tensor_grad::{
    ir::{ethereal::EtherealTensor, mul::find_mul_type, scalar_mul::ScalarMultiplication},
    tensor::{ArcTensor, ReadTensor},
    tensor_ir::TensorIR,
};

use super::{
    arith::{
        multiply::{MultiplicationType, TensorMul},
        scal::TensorScal,
        Side,
    },
    OpSide, TensorIRStruct, TensorInput, TensorOp,
};

/// Reference form of Multiplication
/// Allows for multiplication of boxed trait objects without needing sized.
pub trait ChainGradient<Lhs> {
    type Output;
    fn chain_from(&self, tensor_value: &Lhs, side: Side) -> Self::Output;
}

impl<T, F> ChainGradient<TensorOp<T, F>> for ScalarMultiplication<T>
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
    type Output = TensorOp<T, F>;

    fn chain_from(&self, tensor_value: &TensorOp<T, F>, _side: Side) -> Self::Output {
        TensorOp::new(
            TensorInput::Op(Box::new(tensor_value.clone())),
            TensorInput::None,
            Box::new(TensorScal::new(self.value)),
        )
    }
}

impl<T, F> ChainGradient<TensorOp<T, F>> for TensorOp<T, F>
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
    type Output = TensorOp<T, F>;

    fn chain_from(&self, tensor_value: &TensorOp<T, F>, side: Side) -> Self::Output {
        match side {
            Side::Left => {
                let multiplication_type: MultiplicationType =
                    find_mul_type(&self.output_shape, &tensor_value.output_shape, false, false)
                        .unwrap();
                TensorOp::new(
                    TensorInput::Op(Box::new(self.clone())),
                    TensorInput::Op(Box::new(tensor_value.clone())),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                )
            }
            Side::Right => {
                let multiplication_type: MultiplicationType =
                    find_mul_type(&tensor_value.output_shape, &self.output_shape, false, false)
                        .unwrap();
                TensorOp::new(
                    TensorInput::Op(Box::new(tensor_value.clone())),
                    TensorInput::Op(Box::new(self.clone())),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                )
            }
        }
    }
}

impl<T, F> ChainGradient<TensorIRStruct<T, F>> for TensorIRStruct<T, F>
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

    fn chain_from(&self, tensor_value: &TensorIRStruct<T, F>, side: Side) -> Self::Output {
        match side {
            Side::Left => {
                let multiplication_type: MultiplicationType = find_mul_type(
                    &tensor_value.output_shape(),
                    &self.output_shape(),
                    false,
                    false,
                )
                .unwrap();
                tensor_value.clone().merge(
                    self.clone(),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                )
            }
            Side::Right => {
                let multiplication_type: MultiplicationType = find_mul_type(
                    &self.output_shape(),
                    &tensor_value.output_shape(),
                    false,
                    false,
                )
                .unwrap();
                self.clone().merge(
                    tensor_value.clone(),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                )
            }
        }
    }
}

impl<T, F> ChainGradient<TensorIRStruct<T, F>> for TensorOp<T, F>
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

    fn chain_from(&self, tensor_value: &TensorIRStruct<T, F>, side: Side) -> Self::Output {
        match side {
            Side::Left => {
                let multiplication_type: MultiplicationType = find_mul_type(
                    &tensor_value.output_shape(),
                    &self.output_shape,
                    false,
                    false,
                )
                .unwrap();
                tensor_value.clone().op(
                    TensorInput::Op(Box::new(self.clone())),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                    OpSide::Right,
                )
            }
            Side::Right => {
                let multiplication_type: MultiplicationType = find_mul_type(
                    &self.output_shape,
                    &tensor_value.output_shape(),
                    false,
                    false,
                )
                .unwrap();
                tensor_value.clone().op(
                    TensorInput::Op(Box::new(self.clone())),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                    OpSide::Left,
                )
            }
        }
    }
}

impl<T, F> ChainGradient<TensorIRStruct<T, F>> for TensorIR<T, F>
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

    fn chain_from(&self, tensor_value: &TensorIRStruct<T, F>, side: Side) -> Self::Output {
        match side {
            Side::Left => {
                let multiplication_type: MultiplicationType =
                    find_mul_type(&tensor_value.output_shape(), &self.shape(), false, false)
                        .unwrap();

                tensor_value.clone().op_ir(
                    self.clone(),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                    OpSide::Right,
                )
            }
            Side::Right => {
                let multiplication_type: MultiplicationType =
                    find_mul_type(&self.shape(), &tensor_value.output_shape(), false, false)
                        .unwrap();
                tensor_value.clone().op_ir(
                    self.clone(),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                    OpSide::Left,
                )
            }
        }
    }
}

impl<T, F> ChainGradient<TensorIRStruct<T, F>> for EtherealTensor<T, F>
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

    fn chain_from(&self, tensor_value: &TensorIRStruct<T, F>, side: Side) -> Self::Output {
        match side {
            Side::Left => {
                let multiplication_type: MultiplicationType =
                    find_mul_type(&tensor_value.output_shape(), &self.shape(), false, false)
                        .unwrap();

                tensor_value.clone().op_ethereal(
                    self.clone(),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                    OpSide::Right,
                )
            }
            Side::Right => {
                let multiplication_type: MultiplicationType =
                    find_mul_type(&self.shape(), &tensor_value.output_shape(), false, false)
                        .unwrap();

                tensor_value.clone().op_ethereal(
                    self.clone(),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                    OpSide::Left,
                )
            }
        }
    }
}

impl<T, F> ChainGradient<EtherealTensor<T, F>> for ScalarMultiplication<T>
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

    fn chain_from(&self, tensor_value: &EtherealTensor<T, F>, side: Side) -> Self::Output {
        TensorIRStruct::from_ethereal(
            tensor_value.clone(),
            None,
            Box::new(TensorScal::new(self.value)),
        )
    }
}

impl<T, F> ChainGradient<TensorIR<T, F>> for ScalarMultiplication<T>
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

    fn chain_from(&self, tensor_value: &TensorIR<T, F>, side: Side) -> Self::Output {
        TensorIRStruct::from_ir(
            tensor_value.clone(),
            None,
            Box::new(TensorScal::new(self.value)),
        )
    }
}

impl<T, F> ChainGradient<TensorIRStruct<T, F>> for ScalarMultiplication<T>
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

    fn chain_from(&self, tensor_value: &TensorIRStruct<T, F>, side: Side) -> Self::Output {
        tensor_value
            .clone()
            .scalar_op(Box::new(TensorScal::new(self.value)))
    }
}

impl<T, F> ChainGradient<TensorIRStruct<T, F>> for Box<TensorOp<T, F>>
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

    fn chain_from(&self, tensor_value: &TensorIRStruct<T, F>, side: Side) -> Self::Output {
        match side {
            Side::Left => {
                let multiplication_type: MultiplicationType = find_mul_type(
                    &tensor_value.output_shape(),
                    &self.output_shape,
                    false,
                    false,
                )
                .unwrap();

                tensor_value.clone().op(
                    TensorInput::Op(self.clone()),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                    OpSide::Right,
                )
            }
            Side::Right => {
                let multiplication_type: MultiplicationType = find_mul_type(
                    &self.output_shape,
                    &tensor_value.output_shape(),
                    false,
                    false,
                )
                .unwrap();

                tensor_value.clone().op(
                    TensorInput::Op(self.clone()),
                    Box::new(TensorMul::new(
                        T::one(),
                        false,
                        T::one(),
                        false,
                        multiplication_type,
                    )),
                    OpSide::Right,
                )
            }
        }
    }
}
