use thiserror::Error;

#[derive(Error, Debug)]
pub enum IRError {
    #[error("Operation {ir_name:?} received invalid inputs {inputs:?}")]
    InvalidInputs {
        ir_name: String,
        inputs: (Vec<usize>, Vec<usize>),
    },
}
