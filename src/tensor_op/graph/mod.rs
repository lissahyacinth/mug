use self::viz::GraphViz;

use super::{TensorInput, TensorOp};

use coaster::{IBackend, IFramework};
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{One, Zero};
use rand::{distributions::Standard, prelude::Distribution};
use uuid::Uuid;

use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    NumCast,
};
use num_traits::real::Real;

type TensorDims = Vec<usize>;

mod traversal;
pub(crate) mod viz;
mod writer;

type NodeId = Uuid;

#[derive(Clone, Copy)]
pub(crate) enum Direction {
    Left,
    Right,
    Straight,
}

#[derive(Clone)]
pub enum VertexViz<T, F>
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
    Node(GraphViz<T, F>),
    Root(TensorDims),
    None,
}

#[derive(Clone, Debug)]
pub(crate) struct GraphNode {
    id: NodeId,
    op_string: String,
    level: usize,
    horizon_coord: i16,
    link: Option<NodeId>,
}

type Coords = (usize, i16);

impl<T, F> TensorOp<T, F>
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
    pub fn graph(&self, level: Option<usize>) -> GraphViz<T, F> {
        GraphViz {
            lhs: match self.left_tensor.clone() {
                TensorInput::None => Box::new(VertexViz::None),
                TensorInput::Op(op) => {
                    Box::new(VertexViz::Node(op.graph(if let Some(level) = level {
                        Some(level)
                    } else {
                        Some(1)
                    })))
                }
                TensorInput::Tensor(tensor) => Box::new(VertexViz::Root(tensor.shape().to_vec())),
                TensorInput::EtherealTensor(tensor) => {
                    Box::new(VertexViz::Root(tensor.shape().to_vec()))
                }
            },
            shape: self.output_shape.clone(),
            level: if let Some(level) = level { level } else { 0 },
            rhs: match self.right_tensor.clone() {
                TensorInput::None => Box::new(VertexViz::None),
                TensorInput::Op(op) => {
                    Box::new(VertexViz::Node(op.graph(if let Some(level) = level {
                        Some(level)
                    } else {
                        Some(1)
                    })))
                }
                TensorInput::Tensor(tensor) => Box::new(VertexViz::Root(tensor.shape().to_vec())),
                TensorInput::EtherealTensor(tensor) => {
                    Box::new(VertexViz::Root(tensor.shape().to_vec()))
                }
            },
            operation: self.operation_ir.clone(),
        }
    }
}

#[cfg(test)]
mod test {
    use std::rc::Rc;

    use super::*;
    use crate::{
        tensor_grad::{ir::hadamard::Hadamard, Tensor},
        utility::{filled_tensor, get_native_backend},
    };

    #[test]
    fn print_graph() {
        let backend = Rc::new(get_native_backend());
        let input = Tensor::new(
            filled_tensor(&backend, &[2, 2], &[1.0_f32, 2., 3., 4.]),
            vec![2, 2],
            &backend,
        );
        let res = ((&input.sqrt()).hadamard(&input) + &input).sum().sqrt();
        print!("Graph Output: \n{}", res.graph())
    }
}
