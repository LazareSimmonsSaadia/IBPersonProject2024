use std::ops::{Add, Div, Mul, Sub};

use num::Float;

pub trait Vector<Scalar: Float = f32> where Self: Sized + Clone{
    fn scale(self, scalar: Scalar) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn elem_mul(self, rhs: Self) -> Self;

    fn partial_sum(&self) -> Scalar;

    fn magnitude(&self) -> Scalar {
        self.clone().dot(self.clone()).sqrt()
    }

    fn dot(self, rhs: Self) -> Scalar {
        self.elem_mul(rhs).partial_sum()
    }
}

pub trait Matrix<Scalar: Float = f32>: Sized {
    type Line: Vector<Scalar>;
    type LineIter: Iterator<Item = Self::Line>;
    fn from_vectors(input: Vec<Self::Line>) -> Option<Self>;

    fn dimensions(&self) -> (usize, usize);
    fn row(&self, index: usize) -> Self::Line;
    fn column(&self, index: usize) -> Self::Line;

    fn row_iter(&self) -> Self::LineIter;
    fn col_iter(&self) -> Self::LineIter;

    fn scale(&self, rhs: Scalar) -> Self {
        Self::from_vectors(self.row_iter().map(|i| i.scale(rhs.clone())).collect()).unwrap()
    }
    fn mul(self, rhs: Self) -> Option<Self> {
        if self.dimensions().1 != rhs.dimensions().0 {
            return None;
        }
        Self::from_vectors(
            self.col_iter()
                .zip(rhs.row_iter())
                .map(|(i, j)| i.elem_mul(j))
                .collect(),
        )
    }
    fn add(self, rhs: Self) -> Option<Self> {
        if self.dimensions() != rhs.dimensions() {
            return None;
        }
        Self::from_vectors(
            self.row_iter()
                .zip(rhs.col_iter())
                .map(|(i, j)| i.add(j))
                .collect(),
        )
    }
    }

