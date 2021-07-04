use std::rc::Rc;

use ndarray::prelude::*;
use num_traits::real::Real;

use coaster as co;

use co::{plugin::numeric_helpers::NumCast, prelude::*};

pub use num::{One, Zero};

use crate::utility::filled_tensor;

use coaster_blas::plugin::{Asum, Axpy, Gemm};

pub fn convert_to_tensor<T, F, D>(backend: &Rc<Backend<F>>, array: Array<T, D>) -> SharedTensor<T>
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
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    D: Dimension,
{
    filled_tensor(backend, &array.shape(), array.as_slice().unwrap())
}

#[cfg(test)]
mod tests {

    use super::*;

    fn create_2d_array() -> Array2<f32> {
        return array![[1., 2., 3.], [4., 5., 6.],];
    }

    #[test]
    fn convert_array_to_tensor() {
        use crate::utility::get_native_backend;
        let backend = Rc::new(get_native_backend());
        let input = create_2d_array();
        let _tensor = convert_to_tensor(&backend, input);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn convert_cuda_array_to_tensor() {
        use crate::utility::get_cuda_backend;
        let backend = Rc::new(get_cuda_backend());
        let input = create_2d_array();
        let _tensor = convert_to_tensor(&backend, input);
    }
}
