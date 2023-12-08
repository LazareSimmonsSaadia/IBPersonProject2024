use thiserror::Error;

use super::vector_simd::SimdVector;

use rayon::prelude::*;

#[derive(Debug, PartialEq)]
pub struct SimdMatrix {
    pub matrix: Vec<SimdVector>,
    pub row_size: usize,
}

pub struct SimdLineIter<'a> {
    parent: &'a SimdMatrix,
    linegetfn: fn(&SimdMatrix, usize) -> Option<SimdVector>,
    count: usize,
}

#[derive(Debug, Error)]
pub enum MatrixCreationError {
    #[error("Passing a 2D Vector with inconsistent row lengths to Matrix::from() results in failiure because matrices have uniform row length")]
    InconsistentRowLengthErr,
}

impl SimdMatrix {
    pub fn from(input: Vec<Vec<f32>>) -> Result<SimdMatrix, MatrixCreationError> {
        let row = input.get(0).unwrap().len();
        let lengths = input.par_iter().map(Vec::<f32>::len);
        if lengths.all(|i| i == row) {
            let matrix = SimdMatrix {
                matrix: input
                    .par_iter()
                    .map(|i| SimdVector::from_vector(i.to_owned()))
                    .collect(),
                row_size: row,
            };
            Ok(matrix)
        } else {
            Err(MatrixCreationError::InconsistentRowLengthErr)
        }
    }

    pub fn from_simd(input: Vec<SimdVector>) -> Result<SimdMatrix, MatrixCreationError> {
        let row = input.get(0).unwrap().len();
        let lengths = input.par_iter().map(SimdVector::len);
        if lengths.all(|i| i == row) {
            let matrix = SimdMatrix {
                matrix: input,
                row_size: row,
            };
            Ok(matrix)
        } else {
            Err(MatrixCreationError::InconsistentRowLengthErr)
        }
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
            .par_iter()
            .filter_map(|i| i.get(column))
            .collect();
        if column_slice.is_empty() {
            None
        } else {
            Some(SimdVector::from_vector(column_slice))
        }
    }

    pub fn row(&self, row: usize) -> Option<SimdVector> {
        self.matrix.get(row).cloned()
    }

    pub fn to_vector(&self) -> Vec<Vec<f32>> {
        self.matrix.par_iter().map(|i| i.to_vector()).collect()
    }

    pub fn iter_column(&self) -> SimdLineIter {
        SimdLineIter {
            parent: (&self),
            linegetfn: SimdMatrix::column,
            count: (0),
        }
    }

    pub fn iter_row(&self) -> SimdLineIter {
        SimdLineIter {
            parent: &self,
            linegetfn: SimdMatrix::row,
            count: 0,
        }
    }
}

impl<'a> Iterator for SimdLineIter<'a> {
    type Item = SimdVector;
    fn next(&mut self) -> Option<Self::Item> {
        let val = (self.linegetfn)(self.parent, self.count);
        self.count += 1;
        val
    }
}
