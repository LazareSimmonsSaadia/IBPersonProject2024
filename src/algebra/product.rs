use crate::structures::matrix::MatrixNxN;
use std::simd::{LaneCount, SupportedLaneCount};

impl<'a, const N: usize> MatrixNxN<'a, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn dot(lhs: Vec<Vec<f32>>, rhs: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut matrix = MatrixNxN::from(lhs, rhs);
    }
}
