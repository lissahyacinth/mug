use crate::tensor_op::{arith::pow::TensorPow, TensorIRStruct};
use co::{plugin::numeric_helpers::NumCast, prelude::*};
use coaster as co;
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::arith::{mean::TensorMean, sqrt::TensorSqrt, sum::TensorSum},
};

macro_rules! create_tensor_ir {
    ($op_func:ident, $op_ir:ident, $op_struct:ident) => {
        impl<T, F> TensorIRStruct<T, F>
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
            pub fn $op_func(self) -> TensorIRStruct<T, F> {
                self.scalar_op(Box::new($op_struct::new()))
            }
        }

        impl<T, F> ArcTensor<T, F>
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
            pub fn $op_func(&self) -> TensorIRStruct<T, F> {
                TensorIRStruct::new(self.clone(), None, Box::new($op_struct::new()))
            }
        }
    };
}

create_tensor_ir!(mean, Mean, TensorMean);
create_tensor_ir!(sqrt, Sqrt, TensorSqrt);
create_tensor_ir!(sum, Sum, TensorSum);
