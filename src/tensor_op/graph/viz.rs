use coaster::{IBackend, IFramework};
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{NumCast, One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use super::{Coords, NodeId};
use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::{
        graph::{
            traversal::{merge_construct_limits, recursively_find_limits},
            writer::flush_line,
            Direction, VertexViz,
        },
        TensorOperation,
    },
};

use super::writer::recursive_writer;

#[derive(Clone)]
pub struct GraphViz<T, F>
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
    pub(crate) level: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) lhs: Box<VertexViz<T, F>>,
    pub(crate) rhs: Box<VertexViz<T, F>>,
    pub(crate) operation: Box<dyn TensorOperation<T, F>>,
}

impl<T, F> ::std::fmt::Display for GraphViz<T, F>
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
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        let construct_limits = merge_construct_limits(
            recursively_find_limits(&*self.lhs, (0, 0, vec![]), Direction::Left),
            recursively_find_limits(&*self.rhs, (0, 0, vec![]), Direction::Right),
        );
        let max_level = construct_limits.2.len();
        let min_padding = -(construct_limits.0 as i16);
        let mut output = recursive_writer(&self, (max_level, 0), None);
        output.sort_by(|lhs, rhs| lhs.level.partial_cmp(&rhs.level).unwrap());
        let coord_lookup: std::collections::HashMap<NodeId, Coords> = output
            .clone()
            .into_iter()
            .map(|x| (x.id, (x.level, x.horizon_coord)))
            .collect();
        let mut print_line: Vec<(String, i16)> = Vec::new();
        let mut link_line: Vec<(i16, i16)> = Vec::new();
        let mut previous_level = max_level;
        for node in output {
            if node.level != previous_level {
                flush_line(
                    min_padding,
                    print_line.drain(..).collect(),
                    link_line.drain(..).collect(),
                    f,
                );
            }
            print_line.push((node.op_string, node.horizon_coord));
            if let Some(ref link) = node.link {
                link_line.push((node.horizon_coord, coord_lookup[link].1));
            }
            previous_level = node.level;
        }
        flush_line(min_padding, print_line, link_line, f);
        Ok(())
    }
}

impl<T, F> GraphViz<T, F>
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
    fn invert(self) -> GraphViz<T, F> {
        match *(self.lhs) {
            VertexViz::Node(_lhs_node) => {}
            VertexViz::Root(_lhs_root) => {}
            VertexViz::None => {}
        };
        match *(self.rhs) {
            VertexViz::Node(_lhs_node) => {}
            VertexViz::Root(_lhs_root) => {}
            VertexViz::None => {}
        };
        todo!()
    }
}
