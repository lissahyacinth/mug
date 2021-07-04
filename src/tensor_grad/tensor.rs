use co::{plugin::numeric_helpers::NumCast, prelude::*, Backend};
use coaster as co;

use coaster_blas::plugin::Gemm;
use crossbeam::sync::ShardedLockReadGuard;
use log::warn;
use ndarray::prelude::*;
use num::{One, Zero};
use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    utility::{get_native_backend, random_initialize},
    ArcLock,
};
use num_traits::real::Real;

use crossbeam_utils::sync::ShardedLock;

use std::{rc::Rc, sync::Arc};

use crate::coaster_ndarray::convert_to_tensor;

use super::{Asum, Axpy};

#[derive(Clone)]
pub struct ArcTensor<T, F>
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
{
    pub(crate) tensor_id: TensorID,
    dims: Vec<usize>,
    tensor: ArcLock<Tensor<T, F>>,
    backend: Rc<Backend<F>>,
    pub(crate) has_gradient: bool,
    pub gradient: Option<Box<ArcTensor<T, F>>>,
}

struct TensorReadIterator<'a, T, I: Iterator<Item = T>> {
    dims: &'a [usize],
    data: I,
    index: Vec<usize>,
}

fn create_tensor_read_iterator<I, T>(iter: I, dims: &[usize]) -> TensorReadIterator<T, I::IntoIter>
where
    I: IntoIterator<Item = T>,
{
    let mut index = vec![1; dims.len()];
    index[dims.len() - 1] = 0;
    TensorReadIterator {
        dims,
        data: iter.into_iter(),
        index,
    }
}

fn check_diff(index: &[usize], dimensions: &[usize]) -> (Vec<bool>, Vec<usize>) {
    let mut overflow = false;
    let mut local_index = index.to_vec();
    let mut modified: Vec<bool> = vec![false; dimensions.len()];
    for i in 1..dimensions.len() {
        let j = dimensions.len() - i;
        if overflow {
            local_index[j] += 1;
        }
        overflow = local_index[j] > dimensions[j];
        if overflow {
            local_index[j] = 1;
        }
        modified[j] = overflow
    }
    if overflow {
        local_index[0] += 1;
    }
    (modified, local_index)
}

impl<'a, T, I> std::iter::Iterator for TensorReadIterator<'a, T, I>
where
    I: Iterator<Item = T>,
{
    type Item = (Vec<usize>, Vec<bool>, T);

    fn next(&mut self) -> Option<Self::Item> {
        match self.data.next() {
            Some(element) => {
                self.index[self.dims.len() - 1] += 1;
                let (updates, updated_index) = check_diff(&self.index, self.dims);
                self.index = updated_index;
                Some((self.index.clone(), updates, element))
            }
            None => None,
        }
    }
}

fn padding(depth: usize) -> String {
    std::iter::repeat(" ").take(depth).collect()
}

impl<T, F> ::std::fmt::Display for ArcTensor<T, F>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>
        + std::fmt::Display,
    F: IFramework + Clone + 'static,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let raw_vec = self.read();
        let max_depth = self.dims.len();
        let mut new_line: bool = true;
        // TODO: Filter out Dim == 0
        for i in 0..(max_depth - 1) {
            write!(f, "{}[\n", padding(i))?;
        }
        for (_indices, updates, element) in
            create_tensor_read_iterator(raw_vec.iter(), self.shape())
        {
            //dbg!(indices);
            //dbg!(&updates);
            if new_line {
                write!(f, "{}[", padding(max_depth))?;
                new_line = false;
            }
            let closes = updates.iter().filter(|x| **x).count();
            if closes > 0 {
                write!(f, "{}],\n", padding(max_depth - 1))?;
                for reduction in 1..closes {
                    write!(f, "{}]\n", padding(max_depth - reduction - 1))?;
                }
                for opening in 1..closes {
                    write!(f, "{}[\n", padding(opening))?;
                }
                write!(f, "{}[ {}, ", padding(max_depth), element)?;
            } else {
                write!(f, " {}, ", element)?;
            }
        }
        for i in (0..(max_depth)).rev() {
            write!(f, "{}]\n", padding(i))?;
        }
        Ok(())
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
        + num::traits::Pow<T, Output = T>,
    F: IFramework + Clone + 'static,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    pub fn get_tensor(&self) -> ShardedLockReadGuard<Tensor<T, F>> {
        self.tensor.read().unwrap()
    }

    pub fn get_tensor_mut(&self) -> crossbeam::sync::ShardedLockWriteGuard<Tensor<T, F>> {
        self.tensor.write().unwrap()
    }

    pub fn gradient(&self) -> Vec<T> {
        match self.gradient.clone() {
            Some(gradient) => (*gradient).read(),
            None => panic!("Expected Gradient"),
        }
    }
}

impl<T, F> PartialEq for ArcTensor<T, F>
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
    ArcTensor<T, F>: ReadTensor<T>,
{
    fn eq(&self, rhs: &ArcTensor<T, F>) -> bool {
        let lhs_ptr = &self.get_tensor().tensor;
        let rhs_ptr = &rhs.get_tensor().tensor;
        ::std::ptr::eq(lhs_ptr, rhs_ptr)
    }
}

pub(crate) type TensorID = uuid::Uuid;
pub struct Tensor<T, F>
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
{
    pub(crate) tensor: SharedTensor<T>,
    dims: Vec<usize>,
    backend: Rc<Backend<F>>,
    has_gradient: bool,
}

pub enum ArrayOutput<T>
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
    Tensor0D(Array0<T>),
    Tensor1D(Array1<T>),
    Tensor2D(Array2<T>),
}

pub trait ReadTensor<T> {
    fn read(&self) -> Vec<T>;
}

pub trait NumericTraits<T>:
    Copy + NumCast + One + Zero + Real + num::traits::Pow<T, Output = T>
{
}
pub trait TensorOpTraits: IFramework + Clone {}

impl<T> ArcTensor<T, Native>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>,
    Native: IFramework + Clone,
    Backend<Native>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
{
    pub fn to_ndarray(self) -> ArrayOutput<T> {
        let dimensions = self.shape();
        let data = self.read();
        match dimensions.len() {
            1 => ArrayOutput::Tensor1D(Array::from_shape_vec(dimensions[0], data).unwrap()),
            2 => ArrayOutput::Tensor2D(
                Array::from_shape_vec((dimensions[0], dimensions[1]), data).unwrap(),
            ),
            _ => unimplemented!(),
        }
    }
}

impl<T> ArcTensor<T, Cuda>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>,
    Cuda: IFramework + Clone,
    Backend<Cuda>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
{
    pub fn to_ndarray(self) -> ArrayOutput<T> {
        let dimensions = self.shape();
        let data = self.read();
        match dimensions.len() {
            1 => ArrayOutput::Tensor1D(Array::from_shape_vec(dimensions[0], data).unwrap()),
            2 => ArrayOutput::Tensor2D(
                Array::from_shape_vec((dimensions[0], dimensions[1]), data).unwrap(),
            ),
            _ => unimplemented!(),
        }
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
        + num::traits::Pow<T, Output = T>,
    F: IFramework + Clone + 'static,
    Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
{
    pub(crate) fn shape(&self) -> &[usize] {
        &self.dims
    }

    pub fn with_grad(mut self) -> Self {
        self.has_gradient = true;
        self
    }

    pub(crate) fn set_gradient(&mut self, gradient: ArcTensor<T, F>) {
        if gradient.shape() == self.shape() {
            self.gradient = Some(Box::new(gradient));
        } else {
            warn!(
                "Gradient Shape {:?} does not match Tensor Shape {:?}",
                gradient.shape(),
                self.shape()
            )
        }
    }

    pub(crate) fn set_data(&mut self, data: ArcTensor<T, F>) {
        if data.shape() == self.shape() {
            self.tensor = data.tensor;
        } else {
            warn!(
                "Data Shape {:?} does not match Tensor Shape {:?}",
                data.shape(),
                self.shape()
            )
        }
    }
}

impl<T> ReadTensor<T> for ArcTensor<T, Native>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>,
    Native: IFramework + Clone,
    Backend<Native>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
{
    fn read(&self) -> Vec<T> {
        let native_backend = get_native_backend();
        self.tensor
            .read()
            .unwrap()
            .tensor
            .read(native_backend.device())
            .unwrap()
            .as_slice()
            .to_vec()
    }
}

impl<T> ReadTensor<T> for ArcTensor<T, Cuda>
where
    T: std::fmt::Debug
        + std::fmt::Display
        + Copy
        + NumCast
        + One
        + Zero
        + Real
        + num::traits::Pow<T, Output = T>,
    Native: IFramework + Clone,
    Backend<Cuda>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
{
    fn read(&self) -> Vec<T> {
        self.tensor
            .read()
            .unwrap()
            .tensor
            .read(get_native_backend().device())
            .unwrap()
            .as_slice::<T>()
            .to_vec()
    }
}

impl<T, F> Tensor<T, F>
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
    F: IFramework + Clone + 'static,
    coaster::Backend<F>: IBackend + Axpy<T> + Asum<T> + Gemm<T>,
    ArcTensor<T, F>: ReadTensor<T>,
{
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        tensor: SharedTensor<T>,
        dims: Vec<usize>,
        backend: &Rc<Backend<F>>,
    ) -> ArcTensor<T, F> {
        ArcTensor {
            tensor_id: uuid::Uuid::new_v4(),
            tensor: Arc::new(ShardedLock::new(Tensor {
                tensor,
                dims: dims.clone(),
                backend: Rc::clone(backend),
                has_gradient: false,
            })),
            dims,
            backend: Rc::clone(backend),
            has_gradient: false,
            gradient: None,
        }
    }

    #[allow(clippy::new_ret_no_self)]
    pub fn with_id(
        tensor: SharedTensor<T>,
        dims: Vec<usize>,
        backend: &Rc<Backend<F>>,
        tensor_id: uuid::Uuid,
    ) -> ArcTensor<T, F> {
        ArcTensor {
            tensor_id,
            tensor: Arc::new(ShardedLock::new(Tensor {
                tensor,
                dims: dims.clone(),
                backend: Rc::clone(backend),
                has_gradient: false,
            })),
            dims,
            backend: Rc::clone(backend),
            has_gradient: false,
            gradient: None,
        }
    }

    pub fn initialize_random(dims: Vec<usize>, backend: &Rc<Backend<F>>) -> ArcTensor<T, F> {
        Tensor::new(random_initialize::<T>(&dims), dims, backend)
    }

    pub fn from_array<D>(input: Array<T, D>, backend: &Rc<Backend<F>>) -> ArcTensor<T, F>
    where
        D: Dimension,
    {
        let input_shape = input.shape().to_vec();
        let tensor = convert_to_tensor(&*backend, input);
        assert_eq!(tensor.desc(), &input_shape);
        Tensor::new(tensor, input_shape, backend)
    }
}

#[cfg(test)]
mod test {
    use co::frameworks::cuda::get_cuda_backend;

    use crate::utility::{filled_tensor, ones};

    use super::*;

    fn create_2d_array() -> Array2<f32> {
        return array![[1., 2., 3.], [4., 5., 6.],];
    }

    #[test]
    fn create_tensor_with_gradient() {
        let backend = Rc::new(get_native_backend());
        let input = create_2d_array();
        Tensor::from_array(input, &backend);
    }

    #[test]
    fn check_tensor_conversion() {
        let backend = Rc::new(get_native_backend());
        let input = create_2d_array();
        let tensor = Tensor::from_array(input, &backend);
        assert_eq!(tensor.read(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn check_diff_test() {
        let (diff, new_index) = check_diff(&[2, 3], &[3, 2]);
        assert_eq!(diff, vec![false, true]);
        assert_eq!(new_index, vec![3, 1]);

        let (diff, new_index) = check_diff(&[0, 3], &[3, 2]);
        assert_eq!(diff, vec![false, true]);
        assert_eq!(new_index, vec![1, 1]);

        let (diff, new_index) = check_diff(&[3, 2], &[3, 2]);
        assert_eq!(diff, vec![false, false]);
        assert_eq!(new_index, vec![3, 2]);

        let (diff, new_index) = check_diff(&[1, 1, 2], &[3, 1, 1]);
        assert_eq!(diff, vec![false, true, true]);
        assert_eq!(new_index, vec![2, 1, 1]);
    }

    #[test]
    fn print_tensor() {
        let backend = Rc::new(get_native_backend());
        let tensor = ones::<f32, Native>(&backend, &[2, 10, 5]);
        print!("{}", tensor);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn check_cuda_tensor_conversion() {
        let backend = Rc::new(get_cuda_backend());
        let input = filled_tensor(&backend, &[6], &[1_f32, 2., 3., 4., 5., 6.]);
        let tensor = Tensor::new(input, vec![2, 3], &backend);
        assert_eq!(tensor.read(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn check_cuda_tensor_conversion_from_array() {
        let backend = Rc::new(get_cuda_backend());
        let input = create_2d_array();
        dbg!(input.shape());
        dbg!(input.clone().into_raw_vec());
        let _data = filled_tensor(&backend, &input.shape(), input.as_slice().unwrap());
        let tensor = Tensor::from_array(input, &backend);
        assert_eq!(tensor.read(), vec![1., 2., 3., 4., 5., 6.]);
    }
}
