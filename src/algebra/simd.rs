use crate::structures::{
    matrix_simd::{SimdLineIter, SimdMatrix},
    vector_simd::SimdVector,
};

use super::vector::{Matrix, Vector};

impl Vector<f32> for SimdVector {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    #[inline]
    fn scale(self, scalar: f32) -> Self {
        self.scale(scalar)
    }
    #[inline]
    fn elem_mul(self, rhs: Self) -> Self {
        self * rhs
    }
    #[inline]
    fn partial_sum(&self) -> f32 {
        self.sum()
    }
    #[inline]
    fn from_vec(input: Vec<f32>) -> Self {
        Self::from_vector(input)
    }
    #[inline]
    fn to_vec(&self) -> Vec<f32> {
        self.to_vector()
    }
}

impl Matrix for SimdMatrix {
    type Line = SimdVector;
    type LineIter<'a> = SimdLineIter<'a>;
    fn dimensions(&self) -> (usize, usize) {
        (self.height(), self.row_size)
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
