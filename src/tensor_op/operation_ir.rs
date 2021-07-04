use super::arith::{add::AddType, multiply::MultiplicationType};

#[derive(Debug, Clone)]
pub(crate) enum OperationIR<T> {
    Add(AddType),
    Sub(AddType),
    Mul(MultiplicationType),
    Transpose,
    Mean,
    Sum,
    Sqrt,
    Pow(T),
    Scal(T),
    Squeeze(usize),
    UnSqueeze(usize),
}

impl<T> OperationIR<T> {
    pub fn to_string(&self) -> String {
        match self {
            OperationIR::Add(_) => "Add IR",
            OperationIR::Sub(_) => "Sub IR",
            OperationIR::Mul(_) => "Mul IR",
            OperationIR::Transpose => "Transpose IR",
            OperationIR::Mean => "Mean IR",
            OperationIR::Sum => "Sum IR",
            OperationIR::Sqrt => "Sqrt IR",
            OperationIR::Pow(_) => "Pow IR",
            OperationIR::Scal(_) => "Scale IR",
            OperationIR::Squeeze(_) => "Squeeze IR",
            OperationIR::UnSqueeze(_) => "UnSqueeze IR",
        }
        .to_string()
    }
}
