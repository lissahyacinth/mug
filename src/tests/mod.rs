use std::{rc::Rc, sync::Arc};

use crate::{
    tensor_grad::{
        tensor::{ArcTensor, ReadTensor},
        Tensor,
    },
    tensor_op::{TensorIRStruct, TensorOp, TensorOperation},
    utility::{filled_tensor, random_initialize},
};
use coaster::{Backend, IBackend, IFramework, Native};

use coaster_blas::plugin::{Asum, Axpy, Gemm};
use ndarray::prelude::*;
use num::{NumCast, One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

struct LinearNetwork {
    weights: ArcTensor<f32, Native>,
    bias: ArcTensor<f32, Native>,
}

impl LinearNetwork {
    pub fn new(input_size: usize, output_size: usize, backend: &Rc<Backend<Native>>) -> Self {
        let weights = Tensor::initialize_random(vec![input_size, output_size], backend).with_grad();
        let bias = Tensor::initialize_random(vec![output_size], backend).with_grad();
        println!(
            "Initialised Linear Network with Weight ID {} and Bias ID {}",
            weights.tensor_id, bias.tensor_id
        );
        println!("Bias Initial {}", bias);
        LinearNetwork { weights, bias }
    }

    pub fn forward(&self, input: ArcTensor<f32, Native>) -> TensorIRStruct<f32, Native> {
        input * &self.weights + &self.bias
    }
}

fn mse(
    input: TensorIRStruct<f32, Native>,
    output: ArcTensor<f32, Native>,
) -> TensorIRStruct<f32, Native> {
    ((&input - &output).pow(2.0)).mean().pow(2.0)
}

#[test]
fn forward_pass_native() {
    use crate::utility::get_native_backend;
    let backend = Rc::new(get_native_backend());
    let input = Tensor::new(
        filled_tensor(
            &backend,
            &[10, 1],
            &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ),
        vec![10, 1],
        &backend,
    );
    let labels = Tensor::new(
        filled_tensor(
            &backend,
            &[10, 1],
            &[0_f32, 0., 0., 0., 0., 1., 1., 1., 1., 1.],
        ),
        vec![10, 1],
        &backend,
    );
    let network = LinearNetwork::new(1, 1, &backend);
    let output = network.forward(input);
    let mut cost = mse(output, labels);
    println!("{}", cost.evaluate(&backend));
}

fn sgd<T, F>(tensor: &mut ArcTensor<T, F>, backend: &Rc<Backend<F>>, learning_rate: T)
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
    println!("Updating Gradient for Tensor with ID {}", tensor.tensor_id);
    println!("Tensor Pre {}", tensor);
    println!("Gradient {}", *tensor.clone().gradient.unwrap());
    let mut updated_gradient = tensor.clone() - (*tensor.clone().gradient.unwrap() * learning_rate);
    println!("Updated Tensor {}", (updated_gradient).evaluate(&backend));
    tensor.set_data(updated_gradient.evaluate(backend));
    println!("Tensor Post {}", tensor);
}

#[test]
fn backward_pass_native() {
    use crate::utility::get_native_backend;
    let backend = Rc::new(get_native_backend());
    let input = Tensor::new(
        filled_tensor(
            &backend,
            &[10, 1],
            &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ),
        vec![10, 1],
        &backend,
    );
    let labels = Tensor::new(
        filled_tensor(
            &backend,
            &[10, 1],
            &[0_f32, 0., 0., 0., 0., 1., 1., 1., 1., 1.],
        ),
        vec![10, 1],
        &backend,
    );
    let network = LinearNetwork::new(1, 1, &backend);
    let mut output = mse(network.forward(input.clone()), labels);
    for _ in 0..5 {
        //println!("Input {}", &input);
        // FIXME: Output is growing - despite it not making any sense to.
        println!("Network Output {}", output.evaluate(&backend));
        println!("Weights {}", network.weights);
        println!("Bias {}", network.bias);
        output.backward(&backend);
        output.update_gradients(&sgd, &backend, 0.1);
    }
}
