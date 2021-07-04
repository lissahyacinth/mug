extern crate crossbeam_utils;
extern crate ndarray;
extern crate num_traits;
extern crate rand;
extern crate thiserror;

use std::sync::Arc;

use crossbeam::sync::ShardedLock;
use env_logger::Env;

use co::plugin::numeric_helpers::NumCast;
use coaster as co;

pub use num::{One, Zero};

mod coaster_ndarray;
pub mod prelude;
pub mod tensor_grad;
mod tests;
pub mod utility;

type ArcLock<T> = Arc<ShardedLock<T>>;

#[macro_use]
pub(crate) mod tensor_op;

#[macro_export]
macro_rules! unpack {
    ($rc_tensor:expr) => {
        match ::std::rc::Rc::try_unwrap($rc_tensor) {
            Ok(tensor) => tensor,
            Err(E) => panic!("Could not unpack Tensor"),
        }
    };
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();
}
