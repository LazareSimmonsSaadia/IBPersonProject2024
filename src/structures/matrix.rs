use super::matrix_simd::SimdExcess;
use std::{
    error::Error,
    ops::Deref,
    simd::{LaneCount, Simd, SupportedLaneCount},
};
use thiserror::Error;

pub struct Matrix<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub vector_flat: Vec<Simd<f32, N>>,
    pub row_size: usize,
    pub excess: <Simd<f32, N> as SimdExcess>::Excess,
}

#[derive(Debug, Error)]
pub enum MatrixCreationError {
    #[error("Passing a 2D Vector with inconsistent row lengths to Matrix::from() results in failiure because matrices have uniform row length")]
    InconsistentRowLengthErr,
}

impl<const N: usize> Matrix<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn from(input: Vec<Vec<f32>>) -> Result<Matrix<N>, MatrixCreationError> {
        let lengths = input.iter().map(|i| i.len());
        if lengths.all(|i| i == lengths.nth(0).unwrap()) {
            let matrix = Matrix::<N> {
                vector_flat: input.concat(),
                row_size: lengths.nth(0).unwrap(),
                excess:  
            };
            Ok(matrix)
        } else {
            Err(MatrixCreationError::InconsistentRowLengthErr)
        }
    }

    pub fn guarantee(&mut self) -> bool {
        self.vector_flat
            .truncate(self.vector_flat.len() - (self.vector_flat.len() % self.row_size));
        self.vector_flat.len() % self.row_size == 0
    }

    pub fn is_square(&self) -> bool {
        self.row_size * self.row_size == self.vector_flat.len()
    }

    pub fn height(&self) -> usize {
        self.vector_flat.len() / self.row_size
    }
}
