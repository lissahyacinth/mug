use coaster::{Backend, IBackend, IDevice, IFramework};
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    tensor_grad::{
        ir::ethereal::EtherealTensor,
        tensor::{ArcTensor, ReadTensor, TensorID},
        tensor_ir::TensorIR,
    },
    tensor_op::{arith::Side, graph::viz::GraphViz},
    utility::ones,
    NumCast,
};
use num_traits::real::Real;

use std::{collections::HashMap, rc::Rc};

pub mod arith;
pub(crate) mod chain_gradient;
pub(crate) mod graph;
pub(crate) mod operation;
pub(crate) mod operation_ir;
pub mod util;

pub use operation::TensorOperation;

#[derive(Clone)]
pub enum TensorInput<T, F>
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
    Op(Box<TensorOp<T, F>>),
    Tensor(TensorIR<T, F>),
    EtherealTensor(EtherealTensor<T, F>),
    None,
}

pub(crate) enum OpSide {
    Left,
    Right,
}

#[derive(Clone)]
/// Unevaluated Container for Tensor Operations
///
/// Contains Forward/Backward operations for Tensors, as well as a link to the underlying tensors itself.
/// In a very real way, this structure 'owns' the actual tensors, and during operations with other structures,
/// i.e. Tensors, TensorOPs, or other TensorIRStructs, the tensors are subsumed into this object.
pub struct TensorIRStruct<T, F>
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
    tensors: HashMap<TensorID, ArcTensor<T, F>>,
    ethereal_tensors: HashMap<TensorID, ArcTensor<T, F>>,
    forward_op: TensorOp<T, F>,
    backward_op: Option<TensorOp<T, F>>,
}

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
    pub(crate) fn new(
        lhs: ArcTensor<T, F>,
        rhs: Option<ArcTensor<T, F>>,
        op: Box<dyn TensorOperation<T, F>>,
    ) -> Self {
        let mut tensors: HashMap<TensorID, ArcTensor<T, F>> = HashMap::new();
        tensors.insert(lhs.tensor_id, lhs.clone());
        let lhs_ir = lhs.ir();
        match rhs {
            Some(rhs_tensor) => {
                let rhs_ir = rhs_tensor.ir();
                tensors.insert(rhs_tensor.tensor_id, rhs_tensor);
                TensorIRStruct {
                    tensors,
                    ethereal_tensors: HashMap::new(),
                    forward_op: TensorOp::new(
                        TensorInput::Tensor(lhs_ir),
                        TensorInput::Tensor(rhs_ir),
                        op,
                    ),
                    backward_op: None,
                }
            }
            None => TensorIRStruct {
                tensors,
                ethereal_tensors: HashMap::new(),
                forward_op: TensorOp::new(TensorInput::Tensor(lhs_ir), TensorInput::None, op),
                backward_op: None,
            },
        }
    }

    pub(crate) fn from_ethereal(
        lhs: EtherealTensor<T, F>,
        rhs: Option<EtherealTensor<T, F>>,
        op: Box<dyn TensorOperation<T, F>>,
    ) -> Self {
        let mut tensors: HashMap<TensorID, ArcTensor<T, F>> = HashMap::new();
        match rhs {
            Some(rhs_tensor) => TensorIRStruct {
                tensors,
                ethereal_tensors: HashMap::new(),
                forward_op: TensorOp::new(
                    TensorInput::EtherealTensor(lhs),
                    TensorInput::EtherealTensor(rhs_tensor),
                    op,
                ),
                backward_op: None,
            },
            None => TensorIRStruct {
                tensors,
                ethereal_tensors: HashMap::new(),
                forward_op: TensorOp::new(TensorInput::EtherealTensor(lhs), TensorInput::None, op),
                backward_op: None,
            },
        }
    }

    pub(crate) fn from_ir(
        lhs: TensorIR<T, F>,
        rhs: Option<TensorIR<T, F>>,
        op: Box<dyn TensorOperation<T, F>>,
    ) -> Self {
        let mut tensors: HashMap<TensorID, ArcTensor<T, F>> = HashMap::new();
        match rhs {
            Some(rhs_tensor) => TensorIRStruct {
                tensors,
                ethereal_tensors: HashMap::new(),
                forward_op: TensorOp::new(
                    TensorInput::Tensor(lhs),
                    TensorInput::Tensor(rhs_tensor),
                    op,
                ),
                backward_op: None,
            },
            None => TensorIRStruct {
                tensors,
                ethereal_tensors: HashMap::new(),
                forward_op: TensorOp::new(TensorInput::Tensor(lhs), TensorInput::None, op),
                backward_op: None,
            },
        }
    }

    pub(crate) fn op(
        mut self,
        rhs: TensorInput<T, F>,
        op: Box<dyn TensorOperation<T, F>>,
        side: OpSide,
    ) -> Self {
        match rhs {
            TensorInput::Op(rhs_op) => match side {
                OpSide::Left => {
                    self.forward_op = TensorOp::new(
                        TensorInput::Op(rhs_op),
                        TensorInput::Op(Box::new(self.forward_op)),
                        op,
                    );
                }
                OpSide::Right => {
                    self.forward_op = TensorOp::new(
                        TensorInput::Op(Box::new(self.forward_op)),
                        TensorInput::Op(rhs_op),
                        op,
                    );
                }
            },
            TensorInput::Tensor(rhs_tensor) => match side {
                OpSide::Left => {
                    self.forward_op = TensorOp::new(
                        TensorInput::Tensor(rhs_tensor),
                        TensorInput::Op(Box::new(self.forward_op)),
                        op,
                    );
                }
                OpSide::Right => {
                    self.forward_op = TensorOp::new(
                        TensorInput::Op(Box::new(self.forward_op)),
                        TensorInput::Tensor(rhs_tensor),
                        op,
                    );
                }
            },
            TensorInput::None => match side {
                OpSide::Left => {
                    self.forward_op = TensorOp::new(
                        TensorInput::None,
                        TensorInput::Op(Box::new(self.forward_op)),
                        op,
                    );
                }
                OpSide::Right => {
                    self.forward_op = TensorOp::new(
                        TensorInput::Op(Box::new(self.forward_op)),
                        TensorInput::None,
                        op,
                    );
                }
            },
            TensorInput::EtherealTensor(e_tensor) => match side {
                OpSide::Left => {
                    self.forward_op = TensorOp::new(
                        TensorInput::EtherealTensor(e_tensor),
                        TensorInput::Op(Box::new(self.forward_op)),
                        op,
                    );
                }
                OpSide::Right => {
                    self.forward_op = TensorOp::new(
                        TensorInput::Op(Box::new(self.forward_op)),
                        TensorInput::EtherealTensor(e_tensor),
                        op,
                    );
                }
            },
        }
        self
    }

    pub(crate) fn merge(mut self, rhs: Self, op: Box<dyn TensorOperation<T, F>>) -> Self {
        for (tensor_id, tensor) in rhs.tensors {
            self.tensors.entry(tensor_id).or_insert(tensor);
        }
        self.forward_op = TensorOp::new(
            TensorInput::Op(Box::new(self.forward_op)),
            TensorInput::Op(Box::new(rhs.forward_op)),
            op,
        );
        self
    }

    pub(crate) fn op_tensor(
        mut self,
        tensor: ArcTensor<T, F>,
        op: Box<dyn TensorOperation<T, F>>,
        side: OpSide,
    ) -> Self {
        match side {
            OpSide::Left => {
                let tensor_ir = tensor.ir();
                self.tensors.insert(tensor.tensor_id, tensor);
                self.forward_op = TensorOp::new(
                    TensorInput::Tensor(tensor_ir),
                    TensorInput::Op(Box::new(self.forward_op)),
                    op,
                );
            }
            OpSide::Right => {
                let tensor_ir = tensor.ir();
                self.tensors.insert(tensor.tensor_id, tensor);
                self.forward_op = TensorOp::new(
                    TensorInput::Op(Box::new(self.forward_op)),
                    TensorInput::Tensor(tensor_ir),
                    op,
                );
            }
        }
        self
    }

    pub(crate) fn scalar_op(mut self, op: Box<dyn TensorOperation<T, F>>) -> Self {
        self.forward_op = TensorOp::new(
            TensorInput::Op(Box::new(self.forward_op)),
            TensorInput::None,
            op,
        );
        self
    }

    pub(crate) fn op_ir(
        mut self,
        tensor_ir: TensorIR<T, F>,
        op: Box<dyn TensorOperation<T, F>>,
        side: OpSide,
    ) -> Self {
        assert!(self.tensors.contains_key(&tensor_ir.id));
        match side {
            OpSide::Left => {
                // assert tensor is in hashmap
                self.forward_op = TensorOp::new(
                    TensorInput::Tensor(tensor_ir),
                    TensorInput::Op(Box::new(self.forward_op)),
                    op,
                );
            }
            OpSide::Right => {
                self.forward_op = TensorOp::new(
                    TensorInput::Op(Box::new(self.forward_op)),
                    TensorInput::Tensor(tensor_ir),
                    op,
                );
            }
        }
        self
    }

    pub(crate) fn op_ethereal(
        mut self,
        e_tensor: EtherealTensor<T, F>,
        op: Box<dyn TensorOperation<T, F>>,
        side: OpSide,
    ) -> Self {
        match side {
            OpSide::Left => {
                // assert tensor is in hashmap
                self.forward_op = TensorOp::new(
                    TensorInput::EtherealTensor(e_tensor),
                    TensorInput::Op(Box::new(self.forward_op)),
                    op,
                );
            }
            OpSide::Right => {
                self.forward_op = TensorOp::new(
                    TensorInput::Op(Box::new(self.forward_op)),
                    TensorInput::EtherealTensor(e_tensor),
                    op,
                );
            }
        }
        self
    }

    pub fn graph(&self) -> GraphViz<T, F> {
        self.forward_op.graph(None)
    }

    pub fn evaluate(&mut self, backend: &Rc<coaster::Backend<F>>) -> ArcTensor<T, F> {
        self.forward_op
            .evaluate(backend, &self.tensors, &mut self.ethereal_tensors)
    }

    pub fn retrieve_gradients(&mut self) -> Vec<ArcTensor<T, F>> {
        self.tensors
            .values()
            .clone()
            .into_iter()
            .filter(|x| x.has_gradient)
            .cloned()
            .collect()
    }

    pub fn update_gradients(
        &mut self,
        update_fn: &dyn Fn(&mut ArcTensor<T, F>, &Rc<Backend<F>>, f32),
        backend: &Rc<coaster::Backend<F>>,
        learning_rate: f32,
    ) {
        for tensor in self.tensors.values_mut().filter(|x| x.has_gradient) {
            update_fn(tensor, backend, learning_rate);
        }
    }

    pub(crate) fn output_shape(&self) -> &[usize] {
        &self.forward_op.output_shape
    }

    /// Calculate the Gradients of the Inputs that were labelled as having a Gradient
    pub fn backward(&mut self, backend: &Rc<coaster::Backend<F>>) {
        // [x] Enforce Output
        // [x] Identify Inputs that Require a Gradient
        // [x] Identify routes from the output to the tensors with gradients
        // [-] Calculate - as cheaply as possible - the gradients required for each part
        let mut seed = ones(backend, &[1, 1]) * T::one();
        // TODO: Add in case for when Backward Graph has already been calculated
        for (key, value) in seed.clone().tensors {
            println!("Inserting Seed with ID {}", key);
            self.tensors.insert(key, value);
        }
        seed.tensors = self.tensors.clone();
        let gradient_operations = self.forward_op.clone().graph_with_backward(seed);
        for (tensor_id, mut tensor_gradient) in gradient_operations {
            tensor_gradient.tensors = self.tensors.clone();
            let tensor = self.tensors.get_mut(&tensor_id).unwrap();
            println!("Calculate Gradient for {}", tensor.tensor_id);
            println!("Gradient Calculation is {}", tensor_gradient.graph());
            tensor.gradient = Some(Box::new(tensor_gradient.evaluate(backend)));
            println!("Evaluation Succeeds");
            tensor.has_gradient = true;
        }
        self.backward_op = None;
    }
}

#[derive(Clone)]
/// Unevaluated Representation of a Tensor Operation
///
/// This is a recursive struct that can either contain itself, or a IR form of a Tensor, without any attachment
/// to the data underneath. To evaluate the Operation, and produce an ArcTensor, a Hashmap of Tensors must be
/// provided to an evaluator.
pub struct TensorOp<T, F>
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
    left_tensor: TensorInput<T, F>,
    right_tensor: TensorInput<T, F>,
    operation_ir: Box<dyn TensorOperation<T, F>>,
    pub(crate) output_shape: Vec<usize>,
}

impl<T, F> TensorOp<T, F>
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
    pub(crate) fn new(
        lhs: TensorInput<T, F>,
        rhs: TensorInput<T, F>,
        operation_ir: Box<dyn TensorOperation<T, F>>,
    ) -> Self {
        let lhs_shape = match lhs {
            TensorInput::Op(ref lhs_op) => Some(lhs_op.output_shape.as_slice()),
            TensorInput::Tensor(ref lhs_tensor) => Some(lhs_tensor.shape()),
            TensorInput::None => None,
            TensorInput::EtherealTensor(ref e_tensor) => Some(e_tensor.shape()),
        };
        let rhs_shape = match rhs {
            TensorInput::Op(ref rhs_op) => Some(rhs_op.output_shape.as_slice()),
            TensorInput::Tensor(ref rhs_tensor) => Some(rhs_tensor.shape()),
            TensorInput::None => None,
            TensorInput::EtherealTensor(ref e_tensor) => Some(e_tensor.shape()),
        };
        let output_shape = operation_ir.output_shape(lhs_shape, rhs_shape).to_vec();
        TensorOp {
            left_tensor: lhs.clone(),
            right_tensor: rhs.clone(),
            operation_ir,
            output_shape,
        }
    }

    /// Evaluate the Tensor OperationIRs in chain
    pub fn evaluate(
        &self,
        backend: &Rc<coaster::Backend<F>>,
        tensor_map: &HashMap<TensorID, ArcTensor<T, F>>,
        ethereal_map: &mut HashMap<TensorID, ArcTensor<T, F>>,
    ) -> ArcTensor<T, F> {
        let lhs = match self.left_tensor.clone() {
            TensorInput::Op(left_op) => {
                Some((*left_op).evaluate(backend, tensor_map, ethereal_map))
            }
            TensorInput::Tensor(t) => match tensor_map.get(&t.id) {
                Some(tensor) => Some(tensor.clone()),
                None => panic!("Could not find Tensor for TID {}", t.id),
            },
            TensorInput::None => None,
            TensorInput::EtherealTensor(e_tensor) => Some(
                ethereal_map
                    .entry(e_tensor.id)
                    .or_insert(e_tensor.to_tensor(backend))
                    .clone(),
            ),
        };
        let rhs = match self.right_tensor.clone() {
            TensorInput::Op(left_op) => Some(left_op.evaluate(backend, tensor_map, ethereal_map)),
            TensorInput::Tensor(t) => Some(tensor_map[&t.id].clone()),
            TensorInput::None => None,
            TensorInput::EtherealTensor(e_tensor) => Some(
                ethereal_map
                    .entry(e_tensor.id)
                    .or_insert(e_tensor.to_tensor(backend))
                    .clone(),
            ),
        };
        self.operation_ir.evaluate(lhs, rhs, backend)
    }

    pub(crate) fn to_ir_struct(
        &self,
        tensors: HashMap<TensorID, ArcTensor<T, F>>,
    ) -> TensorIRStruct<T, F> {
        TensorIRStruct {
            tensors,
            ethereal_tensors: HashMap::new(),
            forward_op: self.clone(),
            backward_op: None,
        }
    }

    pub fn backward(self, seed: TensorIRStruct<T, F>) -> Vec<(TensorID, TensorIRStruct<T, F>)> {
        println!("Backward - Current Seed Shape {:?}", seed.output_shape());
        println!("Backward - Current Graph {}", seed.graph());
        let mut local_gradients: Vec<(TensorID, TensorIRStruct<T, F>)> = Vec::new();
        match self.left_tensor.clone() {
            TensorInput::Op(lhs_tensor) => {
                let gradient_at_point = self.operation_ir.grad(
                    seed.clone(),
                    self.left_tensor.clone(),
                    self.right_tensor.clone(),
                    Side::Left,
                );
                let mut path_gradients = lhs_tensor.backward(gradient_at_point);
                local_gradients.append(&mut path_gradients);
            }
            TensorInput::Tensor(ref lhs_tensor) => {
                if lhs_tensor.has_gradient {
                    let gradient_at_point = self.operation_ir.grad(
                        seed.clone(),
                        self.left_tensor.clone(),
                        self.right_tensor.clone(),
                        Side::Left,
                    );
                    local_gradients.push((lhs_tensor.id, gradient_at_point));
                }
            }
            TensorInput::EtherealTensor(_) | TensorInput::None => {}
        }
        match self.right_tensor.clone() {
            TensorInput::Op(rhs_tensor) => {
                let gradient_at_point = self.operation_ir.grad(
                    seed,
                    self.right_tensor,
                    self.left_tensor.clone(),
                    Side::Right,
                );
                let mut path_gradients = rhs_tensor.backward(gradient_at_point);
                local_gradients.append(&mut path_gradients);
            }
            TensorInput::Tensor(ref rhs_tensor) => {
                if rhs_tensor.has_gradient {
                    let gradient_at_point = self.operation_ir.grad(
                        seed,
                        self.right_tensor.clone(),
                        self.left_tensor.clone(),
                        Side::Right,
                    );
                    local_gradients.push((rhs_tensor.id, gradient_at_point));
                }
            }
            TensorInput::EtherealTensor(_) | TensorInput::None => {}
        }
        local_gradients
    }

    fn graph_with_backward(
        self,
        seed: TensorIRStruct<T, F>,
    ) -> Vec<(TensorID, TensorIRStruct<T, F>)> {
        let mut gradients: Vec<(TensorID, TensorIRStruct<T, F>)> = Vec::new();
        println!("Backward - Current Seed Shape {:?}", seed.output_shape());
        match self.left_tensor.clone() {
            TensorInput::Op(lhs_op) => {
                let gradient_at_point = self.operation_ir.grad(
                    seed.clone(),
                    self.left_tensor.clone(),
                    self.right_tensor.clone(),
                    Side::Left,
                );
                gradients.append(&mut lhs_op.backward(gradient_at_point));
            }
            TensorInput::Tensor(ref lhs_tensor) => {
                let has_gradient = lhs_tensor.has_gradient;
                let gradient_at_point = self.operation_ir.grad(
                    seed.clone(),
                    self.left_tensor.clone(),
                    self.right_tensor.clone(),
                    Side::Left,
                );
                if has_gradient {
                    gradients.push((lhs_tensor.id.clone(), gradient_at_point));
                }
            }
            TensorInput::EtherealTensor(_) | TensorInput::None => {}
        }
        match self.right_tensor.clone() {
            TensorInput::Op(rhs_op) => {
                let gradient_at_point =
                    self.operation_ir
                        .grad(seed, self.right_tensor, self.left_tensor, Side::Right);
                gradients.append(&mut rhs_op.backward(gradient_at_point));
            }
            TensorInput::Tensor(ref rhs_tensor) => {
                let has_gradient = rhs_tensor.has_gradient;
                let gradient_at_point =
                    self.operation_ir
                        .grad(seed, self.right_tensor, self.left_tensor, Side::Right);
                if has_gradient {
                    gradients.push((rhs_tensor.id.clone(), gradient_at_point));
                }
            }
            TensorInput::EtherealTensor(_) | TensorInput::None => {}
        }
        gradients
    }
}

#[cfg(test)]
mod test {

    use coaster::frameworks::cuda::get_cuda_backend;

    use crate::{
        tensor_grad::Tensor,
        utility::{filled_tensor, get_native_backend, random_initialize},
    };

    use super::*;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn scalar_gradient() {
        init();
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
            vec![10, 1],
            &backend,
        )
        .with_grad();
        let mut op = input * 2.0;
        op.backward(&backend);
        let tensor = op.retrieve_gradients().pop().unwrap();
        assert_eq!(vec![2.0_f32], tensor.gradient.unwrap().read())
    }

    #[test]
    fn gradient_broadcast_add_bias() {
        init();
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
        let bias = Tensor::new(
            filled_tensor(&backend, &[1, 1], &[0.1]),
            vec![1, 1],
            &backend,
        )
        .with_grad();
        let mut op = (input + bias).sum();
        println!("Graph {}", op.graph());
        op.backward(&backend);
        let tensor = op.retrieve_gradients().pop().unwrap();
        println!("Tensor {}", tensor.clone());
        assert_eq!(tensor.read().len(), 1);
        println!("Tensor Gradient {}", tensor.clone().gradient.unwrap());
        assert_eq!(tensor.clone().gradient.unwrap().read().len(), 1);
        assert_eq!(vec![10.0], tensor.gradient.unwrap().read())
    }

    #[test]
    fn gradient_broadcast_add_input() {
        init();
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
            vec![10, 1],
            &backend,
        )
        .with_grad();
        let bias = Tensor::new(
            filled_tensor(&backend, &[1, 1], &[0.1]),
            vec![1, 1],
            &backend,
        );
        let mut op = (input + bias).sum();
        println!("Graph {}", op.graph());
        op.backward(&backend);
        let tensor = op.retrieve_gradients().pop().unwrap();
        println!("Tensor {}", tensor.clone());
        assert_eq!(tensor.read().len(), 10);
        println!("Tensor Gradient {}", tensor.clone().gradient.unwrap());
        assert_eq!(tensor.clone().gradient.unwrap().read().len(), 10);
    }

    #[test]
    fn mean_gradient() {
        init();
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
            vec![10, 1],
            &backend,
        )
        .with_grad();
        let mut op = input.mean();
        op.backward(&backend);
        let tensor = op.retrieve_gradients().first().unwrap().clone();
        assert_eq!(vec![0.1; 10], tensor.gradient.unwrap().read())
    }

    #[test]
    fn mul_gradient() {
        init();
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
        let weights = Tensor::new(
            filled_tensor(&backend, &[1, 1], &[0.5]),
            vec![1, 1],
            &backend,
        )
        .with_grad();
        let mut op = (&input * &weights).sum();
        op.backward(&backend);
        let tensor = op.retrieve_gradients().first().unwrap().clone();
        assert_eq!(vec![4.5], tensor.gradient.unwrap().read())
    }

    #[test]
    fn mul_gradient_input() {
        init();
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
            vec![10, 1],
            &backend,
        )
        .with_grad();
        let weights = Tensor::new(
            filled_tensor(&backend, &[1, 1], &[0.5]),
            vec![1, 1],
            &backend,
        );
        let mut op = (&input * &weights).sum();
        op.backward(&backend);
        let tensor = op.retrieve_gradients().first().unwrap().clone();
        println!("{}", tensor);
        assert_eq!(vec![0.5; 10], tensor.gradient.unwrap().read())
    }

    #[test]
    fn mse_gradient() {
        init();
        let backend = Rc::new(get_cuda_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
            vec![10, 1],
            &backend,
        )
        .with_grad();
        let weights = Tensor::new(
            filled_tensor(&backend, &[1, 1], &[0.5]),
            vec![1, 1],
            &backend,
        );
        let target = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.0, 0.0, 0.0, 0.0, 1., 1., 1., 1., 1.],
            ),
            vec![10, 1],
            &backend,
        );
        let mut op = ((input * weights) - target).pow(2.0).mean();
        println!("{}", op.graph());
        println!("Output {}", op.evaluate(&backend));
        op.backward(&backend);
        let tensor = op.retrieve_gradients().first().unwrap().clone();
        println!("Tensor {}", tensor);
        println!("Tensor Gradient {}", tensor.clone().gradient.unwrap());
        assert!(vec![
            0.0000, 0.0050, 0.0100, 0.0150, 0.0200, -0.0750, -0.0700, -0.0650, -0.0600, -0.0550
        ]
        .into_iter()
        .zip(tensor.clone().gradient.unwrap().read().into_iter())
        .all(|(a, b)| (a - b).abs() < 0.001))
    }

    #[test]
    fn bias_gradient() {
        init();
        let backend = Rc::new(get_cuda_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
            vec![10, 1],
            &backend,
        )
        .with_grad();
        let weights = Tensor::new(
            filled_tensor(&backend, &[1, 1], &[0.5]),
            vec![1, 1],
            &backend,
        )
        .with_grad();
        println!("Weight Tensor ID is {}", input.tensor_id);
        let bias = Tensor::initialize_random(vec![1, 1], &backend);
        let target = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.0, 0.0, 0.0, 0.0, 1., 1., 1., 1., 1.],
            ),
            vec![10, 1],
            &backend,
        );
        let mut op = (input * weights).pow(2.0).mean();
        println!("{}", op.graph());
        println!("Output {}", op.evaluate(&backend));
        op.backward(&backend);
        for tensor in op.retrieve_gradients() {
            println!("Tensor {}", tensor);
            println!("Tensor Gradient {}", tensor.clone().gradient.unwrap());
        }
    }

    #[test]
    fn gradient_add_sum() {
        init();
        let backend = Rc::new(get_cuda_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
            vec![10, 1],
            &backend,
        )
        .with_grad();
        let target = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.0, 0.0, 0.0, 0.0, 1., 1., 1., 1., 1.],
            ),
            vec![10, 1],
            &backend,
        );
        let mut op = (input + target).sum();
        println!("{}", op.graph());
        println!("Output {}", op.evaluate(&backend));
        op.backward(&backend);
        let tensor = op.retrieve_gradients().first().unwrap().clone();
        println!("Tensor {}", tensor);
        println!("Tensor Gradient {}", tensor.clone().gradient.unwrap());
        assert_eq!(vec![1.0; 10], tensor.clone().gradient.unwrap().read())
    }

    #[test]
    fn gradient_add_pow() {
        init();
        let backend = Rc::new(get_cuda_backend());
        let input = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ),
            vec![10, 1],
            &backend,
        )
        .with_grad();
        let target = Tensor::new(
            filled_tensor(
                &backend,
                &[10, 1],
                &[0_f32, 0.0, 0.0, 0.0, 0.0, 1., 1., 1., 1., 1.],
            ),
            vec![10, 1],
            &backend,
        );
        let mut op = (input + target).pow(2.0).mean();
        println!("{}", op.graph());
        println!("Output {}", op.evaluate(&backend));
        op.backward(&backend);
        let tensor = op.retrieve_gradients().first().unwrap().clone();
        println!("Tensor {}", tensor);
        println!("Tensor Gradient {}", tensor.clone().gradient.unwrap());
        assert!(
            vec![0., 0.02, 0.04, 0.06, 0.08, 0.3, 0.32, 0.34, 0.36, 0.38]
                .into_iter()
                .zip(tensor.clone().gradient.unwrap().read().into_iter())
                .all(|(a, b)| (a - b).abs() < 0.001)
        )
    }
}
