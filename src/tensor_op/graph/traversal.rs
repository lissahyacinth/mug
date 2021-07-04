use coaster::{IBackend, IFramework};
use coaster_blas::plugin::{Asum, Axpy, Gemm};
use itertools::Itertools;
use num::{NumCast, One, Zero};
use num_traits::real::Real;
use rand::{distributions::Standard, prelude::Distribution};

use crate::tensor_grad::tensor::{ArcTensor, ReadTensor};

use super::{Direction, VertexViz};

type InterConstructLimits = (usize, usize, Vec<(usize, usize)>);

pub(crate) fn recursively_find_limits<T, F>(
    node: &VertexViz<T, F>,
    limits: InterConstructLimits,
    direction: Direction,
) -> InterConstructLimits
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
    let (lhs, rhs, mut limit_vec) = limits.clone();
    match node {
        VertexViz::Node(intermediary_node) => {
            let updated_lhs;
            let updated_rhs;
            match direction {
                Direction::Left => {
                    updated_lhs = lhs + 1;
                    updated_rhs = rhs;
                    limit_vec.push((updated_lhs, updated_rhs));
                }
                Direction::Right => {
                    updated_lhs = lhs;
                    updated_rhs = rhs + 1;
                    limit_vec.push((updated_lhs, updated_rhs));
                }
                Direction::Straight => {
                    updated_lhs = lhs;
                    updated_rhs = rhs;
                    limit_vec.push((updated_lhs, updated_rhs));
                }
            }
            let lhs_limit = match &*intermediary_node.rhs {
                VertexViz::None => recursively_find_limits(
                    &*intermediary_node.lhs,
                    (updated_lhs, updated_rhs, limit_vec.clone()),
                    Direction::Straight,
                ),
                _ => recursively_find_limits(
                    &*intermediary_node.lhs,
                    (updated_lhs, updated_rhs, limit_vec.clone()),
                    Direction::Left,
                ),
            };
            let rhs_limit = recursively_find_limits(
                &*intermediary_node.rhs,
                (updated_lhs, updated_rhs, limit_vec),
                Direction::Right,
            );
            merge_construct_limits(lhs_limit, rhs_limit)
        }
        VertexViz::Root(_) => match direction {
            Direction::Left => {
                limit_vec.push((lhs + 1, rhs));
                (lhs + 1, rhs, limit_vec)
            }
            Direction::Right => {
                limit_vec.push((lhs, rhs + 1));
                (lhs, rhs + 1, limit_vec)
            }
            Direction::Straight => {
                limit_vec.push((lhs, rhs));
                (lhs, rhs, limit_vec)
            }
        },
        VertexViz::None => limits,
    }
}

pub(crate) fn merge_construct_limits(
    lhs: InterConstructLimits,
    rhs: InterConstructLimits,
) -> InterConstructLimits {
    let (furthest_left_lhs, furthest_right_lhs, limits_by_depth_lhs) = lhs;
    let (furthest_left_rhs, furthest_right_rhs, limits_by_depth_rhs) = rhs;
    (
        std::cmp::max(furthest_left_lhs, furthest_left_rhs),
        std::cmp::max(furthest_right_lhs, furthest_right_rhs),
        limits_by_depth_lhs
            .iter()
            .zip_longest(limits_by_depth_rhs.iter())
            .map(|variant| match variant {
                itertools::EitherOrBoth::Both(
                    (left_align_lhs, left_align_rhs),
                    (right_align_lhs, right_align_rhs),
                ) => (
                    *std::cmp::max(left_align_lhs, left_align_rhs),
                    *std::cmp::max(right_align_lhs, right_align_rhs),
                ),
                itertools::EitherOrBoth::Left((left_align_lhs, left_align_rhs)) => {
                    (*left_align_lhs, *left_align_rhs)
                }
                itertools::EitherOrBoth::Right((right_align_lhs, right_align_rhs)) => {
                    (*right_align_lhs, *right_align_rhs)
                }
            })
            .collect::<Vec<(usize, usize)>>(),
    )
}
