use std::{
    error::Error,
    ops::Deref,
    simd::{LaneCount, Simd, SupportedLaneCount},
};
use thiserror::Error;

use super::matrix_simd::SimdVector;

pub struct Matrix {
    pub matrix: Vec<SimdVector>,
    pub row_size: usize,
}

#[derive(Debug, Error)]
pub enum MatrixCreationError {
    #[error("Passing a 2D Vector with inconsistent row lengths to Matrix::from() results in failiure because matrices have uniform row length")]
    InconsistentRowLengthErr,
}

impl Matrix {
    pub fn from(input: Vec<Vec<f32>>) -> Result<Matrix, MatrixCreationError> {
        let row = input.get(0).unwrap().len();
        let mut lengths = input.iter().map(Vec::<f32>::len);
        if lengths.all(|i| i == row){
            let matrix = Matrix {
                matrix: input.iter().map(|i| SimdVector::from_vector(i.to_owned())).collect(),
                row_size: row,
            };
            Ok(matrix)
        } else {
            Err(MatrixCreationError::InconsistentRowLengthErr)
        }
    }

    pub fn guarantee(&mut self) -> bool {
        self.matrix
            .iter()
            .map(|i| i.len() == self.row_size)
            .fold(true, |i, j| i && j)
    }

    pub fn is_square(&self) -> bool {
        self.row_size == self.matrix.len()
    }

    pub fn height(&self) -> usize {
        self.matrix.len()
    }

    pub fn column(&self, column: usize) -> Option<SimdVector> {
        let column_slice: Vec<f32> = self
            .matrix
            .iter()
            .filter_map(|i| i.get(column))
            .collect();
        if column_slice.is_empty() {
            None
        } else {
            Some(SimdVector::from_vector(column_slice))
        }
    }
}
