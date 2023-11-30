use crate::structures::{matrix_simd::{SimdMatrix, SimdLineIter}, vector_simd::SimdVector};

use super::vector::{Matrix, Vector};

impl Vector<f32> for SimdVector {
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn scale(self, scalar: f32) -> Self {
        self.scale(scalar)
    }

    fn elem_mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn partial_sum(&self) -> f32 {
        self.sum()
    }
}

impl Matrix for SimdMatrix {
    type Line = SimdVector;
    type LineIter<'a> = SimdLineIter<'a>;
    fn dimensions(&self) -> (usize, usize) {
        (self.matrix.len(), self.height())
    }

    fn from_vectors(input: Vec<Self::Line>) -> Option<Self> {
        SimdMatrix::from_simd(input).ok()
    }

    fn row(&self, index: usize) -> Option<Self::Line> {
        self.row(index)
    }

    fn column(&self, index: usize) -> Option<Self::Line> {
        self.column(index)
    }

    fn row_iter<'a>(&'a self) -> Self::LineIter<'a> {
        self.iter_row()
    }

    fn col_iter<'a>(&'a self) -> Self::LineIter<'a> {
        self.iter_column()
    }
}
