use std::collections::HashSet;

use coaster::{IBackend, IFramework};
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use num::{NumCast, One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};
use uuid::Uuid;

use super::{viz::GraphViz, Coords, GraphNode, NodeId};
use crate::{
    tensor_grad::tensor::{ArcTensor, ReadTensor},
    tensor_op::{graph::VertexViz, TensorOperation},
};

pub(crate) fn recursive_writer<T, F>(
    node: &GraphViz<T, F>,
    coords: Coords,
    link: Option<NodeId>,
) -> Vec<GraphNode>
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
    let (level, direction) = coords;
    let current_op = write_op(node.operation.clone(), coords, link);

    let lhs_match = match &*node.rhs {
        VertexViz::None => match &*node.lhs {
            VertexViz::Node(ref node_lhs) => {
                recursive_writer(node_lhs, (level - 1, direction), Some(current_op.id))
            }
            VertexViz::Root(tensor_lhs) => {
                vec![write_root_tensor_info(
                    tensor_lhs,
                    (level - 1, direction),
                    Some(current_op.id),
                )]
            }
            VertexViz::None => vec![],
        },
        _ => match &*node.lhs {
            VertexViz::Node(ref node_lhs) => {
                recursive_writer(node_lhs, (level - 1, direction - 1), Some(current_op.id))
            }
            VertexViz::Root(tensor_lhs) => {
                vec![write_root_tensor_info(
                    tensor_lhs,
                    (level - 1, direction - 1),
                    Some(current_op.id),
                )]
            }
            VertexViz::None => vec![],
        },
    };
    let mut rhs_match = match &*node.rhs {
        VertexViz::Node(ref node_lhs) => {
            recursive_writer(node_lhs, (level - 1, direction + 1), Some(current_op.id))
        }
        VertexViz::Root(tensor_lhs) => vec![write_root_tensor_info(
            tensor_lhs,
            (level - 1, direction + 1),
            Some(current_op.id),
        )],
        VertexViz::None => vec![],
    };
    let lhs_coordinates = lhs_match
        .iter()
        .map(|x| (x.level, x.horizon_coord))
        .collect::<HashSet<(usize, i16)>>();

    rhs_match.sort_by(|a, b| match a.level.cmp(&b.level) {
        std::cmp::Ordering::Less => std::cmp::Ordering::Less,
        std::cmp::Ordering::Greater => std::cmp::Ordering::Greater,
        std::cmp::Ordering::Equal => a.horizon_coord.cmp(&b.horizon_coord),
    });

    let mut shift = 0;
    for r_match in rhs_match.iter() {
        if lhs_coordinates.contains(&(r_match.level, r_match.horizon_coord)) {
            shift += 2;
        }
    }

    for r_match in rhs_match.iter_mut() {
        r_match.horizon_coord += shift;
    }

    vec![current_op]
        .into_iter()
        .chain(lhs_match.into_iter().chain(rhs_match.into_iter()))
        .collect()
}

fn write_op<T, F>(
    tensor_operation: Box<dyn TensorOperation<T, F>>,
    coords: Coords,
    link: Option<NodeId>,
) -> GraphNode
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
    GraphNode {
        id: Uuid::new_v4(),
        op_string: format!("({:^10})", format!("{:?}", tensor_operation.to_string())),
        level: coords.0,
        horizon_coord: coords.1,
        link,
    }
}

fn write_root_tensor_info(tensor: &[usize], coords: Coords, link: Option<NodeId>) -> GraphNode {
    GraphNode {
        id: Uuid::new_v4(),
        op_string: format!("({:^10})", format!("{:?}", tensor)),
        level: coords.0,
        horizon_coord: coords.1,
        link,
    }
}

/// Write a line to the Graph Formatter
///
/// `baseline_lhs` - Left margin of the writing area
/// `line` -
/// `link_line` -
pub(crate) fn flush_line(
    baseline_lhs: i16,
    line: Vec<(String, i16)>,
    link_line: Vec<(i16, i16)>,
    f: &mut ::std::fmt::Formatter,
) {
    let mut line = line;
    let mut link_line = link_line;
    // Spacing from LHS of Screen - increases by one for each element in the link line.
    let mut padding = baseline_lhs;
    line.sort_by(|(_, eh), (_, fh)| eh.partial_cmp(fh).unwrap());
    link_line.sort_by(|(eh, _), (fh, _)| eh.partial_cmp(fh).unwrap());
    for (op_string, horizon) in line {
        if horizon < padding {
            panic!("Flush line attempted to backtrack")
        }
        // Pad from Baseline LHS to Horizon
        let pad = std::iter::repeat(" ")
            .take((horizon - padding) as usize * 10usize)
            .collect::<String>();
        write!(f, "{}", pad).unwrap();
        write!(f, "{}", op_string).unwrap();
        padding = horizon + 1;
    }
    write!(f, "\n").unwrap();
    padding = baseline_lhs;
    for (s, e) in link_line {
        let link_char = match s.cmp(&e) {
            std::cmp::Ordering::Less => format!("{:>10}", r"\"),
            std::cmp::Ordering::Equal => format!("{:^10}", r"|"),
            std::cmp::Ordering::Greater => format!("{:<10}", r"/"),
        }
        .to_string();
        let pad = std::iter::repeat(" ")
            .take((s - padding) as usize * 10usize)
            .collect::<String>();
        write!(f, "{}", pad).unwrap();
        write!(f, "{}", link_char).unwrap();
        padding = s + 1;
    }
    write!(f, "\n").unwrap();
}
