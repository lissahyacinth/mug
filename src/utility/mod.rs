use std::rc::Rc;

use coaster_blas::plugin::{Asum, Axpy, Gemm};

use coaster as co;

use co::{
    plugin::numeric_helpers::{cast, NumCast},
    prelude::*,
};

pub use num::{One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::tensor_grad::{
    tensor::{ArcTensor, ReadTensor},
    Tensor,
};

pub fn get_native_backend() -> Backend<Native> {
    Backend::<Native>::default().unwrap()
}

pub fn get_cuda_backend() -> Backend<Cuda> {
    Backend::<Cuda>::default().unwrap()
}

/// Write an array of elements directly to the memory of a Tensor
/// Must be done on Native, as writing to CUDA or OpenCL directly isn't
pub fn write_to_tensor<T, F>(_backend: &Rc<Backend<F>>, xs: &mut SharedTensor<T>, data: &[T])
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
{
    assert_eq!(xs.desc().size(), data.len());
    let native = get_native_backend();
    let native_dev = native.device();
    {
        let mem = xs.write_only(native_dev).unwrap();
        let mem_buffer = mem.as_mut_slice::<T>();
        for (i, x) in data.iter().enumerate() {
            mem_buffer[i] = cast::<_, T>(*x).unwrap();
        }
    }
}

pub fn write_to_native<T>(xs: &mut SharedTensor<T>, data: &[T])
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>,
{
    assert_eq!(xs.desc().size(), data.len());
    let native = get_native_backend();
    let native_dev = native.device();
    {
        let mem = xs.write_only(native_dev).unwrap();
        let mem_buffer = mem.as_mut_slice::<T>();
        for (i, x) in data.iter().enumerate() {
            mem_buffer[i] = cast::<_, T>(*x).unwrap();
        }
    }
}

pub fn filled_tensor<T, F>(_backend: &Rc<Backend<F>>, dims: &[usize], data: &[T]) -> SharedTensor<T>
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
    Backend<F>: IBackend,
{
    let mut x = SharedTensor::new(&dims);
    write_to_native(&mut x, data);
    x
}

pub fn filled_tensor_native<T>(dims: &[usize], data: &[T]) -> SharedTensor<T>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>,
{
    let mut x = SharedTensor::new(&dims);
    write_to_native(&mut x, data);
    x
}

pub fn copy_tensor<T, F>(
    backend: &Rc<Backend<F>>,
    dims: &[usize],
    tensor: &SharedTensor<T>,
) -> SharedTensor<T>
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
    Backend<F>: IBackend,
{
    let native_backend = get_native_backend();
    let data: Vec<T> = tensor
        .read(native_backend.device())
        .unwrap()
        .as_slice()
        .to_vec();

    filled_tensor(backend, dims, &data)
}

pub fn element_tensor<T, F>(backend: &Rc<Backend<F>>, data: T) -> SharedTensor<T>
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
    Backend<F>: IBackend,
{
    let mut x = SharedTensor::new(&[1]);
    write_to_tensor(backend, &mut x, &[data]);
    x
}

pub fn identity<T, F>(length: usize, backend: &Rc<Backend<F>>) -> ArcTensor<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>
        + Zero,
    F: IFramework + Clone + 'static,
    Standard: Distribution<T>,
    Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    let mut input: Vec<T> = vec![Zero::zero(); length * length];
    for i in 0..length {
        input[i + length * i] = One::one();
    }
    Tensor::new(
        filled_tensor(&backend, &[length, length], &input),
        vec![length, length],
        &backend,
    )
}

pub fn ones<T, F>(backend: &Rc<Backend<F>>, dims: &[usize]) -> ArcTensor<T, F>
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
    Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    let total_elements = dims.iter().product();
    let data: Vec<T> = vec![One::one(); total_elements];
    Tensor::new(filled_tensor(backend, dims, &data), dims.to_vec(), backend)
}

pub fn zeros<T, F>(backend: &Rc<Backend<F>>, dims: &[usize]) -> SharedTensor<T>
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
    Backend<F>: IBackend,
{
    let total_elements = dims.iter().product();
    let data: Vec<T> = vec![Zero::zero(); total_elements];
    filled_tensor(backend, dims, &data)
}

pub fn random_initialize<T>(dims: &[usize]) -> SharedTensor<T>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>,
    Standard: Distribution<T>,
{
    let mut rng = rand::thread_rng();
    let elements: usize = dims.iter().product();
    let input: Vec<T> = (0..elements).map(|_| rng.gen::<T>()).collect();
    filled_tensor_native(&dims, &input)
}

//pub fn rmse(input: TensorOp<f32, Native>, output: ArcTensor<f32, Native>) -> TensorOp<f32, Native> {
//    ((&input - &output).pow(2.0)).mean().sqrt()
//}

#[cfg(test)]
mod tests {
    use crate::tensor_grad::tensor::ReadTensor;

    use co::frameworks::cuda::get_cuda_backend;

    use super::*;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_ones() {
        init();
        let backend = Rc::new(get_native_backend());
        let singleton_input = ones::<f64, Native>(&backend, &[1]);
        let input_2d = ones::<f64, Native>(&backend, &[10, 1]);
        assert_eq!(singleton_input.read(), vec![1.0]);
        assert_eq!(input_2d.read(), vec![1.0; 10]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn read_element_tensor_cuda() {
        init();
        let backend = Rc::new(get_cuda_backend());
        let input = element_tensor(&backend, 1.);
        dbg!(input
            .read(get_native_backend().device())
            .unwrap()
            .as_slice::<f32>()
            .to_vec());
    }

    #[cfg(feature = "native")]
    #[test]
    fn read_filled_tensor_native() {
        init();
        let backend = Rc::new(get_native_backend());
        let input = filled_tensor(&backend, &[5], &[1_f32, 2., 3., 4., 5.]);
        assert_eq!(
            input
                .read(get_native_backend().device())
                .unwrap()
                .as_slice::<f32>()
                .to_vec(),
            vec![1_f32, 2., 3., 4., 5.]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn read_filled_tensor_cuda() {
        let backend = Rc::new(get_cuda_backend());
        let input = filled_tensor(&backend, &[5], &[1_f32, 2., 3., 4., 5.]);
        assert_eq!(
            input
                .read(get_native_backend().device())
                .unwrap()
                .as_slice::<f32>()
                .to_vec(),
            vec![1_f32, 2., 3., 4., 5.]
        );
    }

    #[test]
    fn test_identity() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(&backend, &[2, 2], &[1_f32, 2., 3., 4.]),
            vec![2, 2],
            &backend,
        );
        let rhs: ArcTensor<f32, Native> = identity(2, &backend);
        let mut output = input * rhs;
        assert_eq!(output.evaluate(&backend).read(), vec![1_f32, 2., 3., 4.]);
    }
}
