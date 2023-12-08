use std::fmt::Debug;

use num::Float;

pub trait Vector<Scalar: Float = f32>
where
    Self: Sized + Clone,
    Scalar: Debug + std::fmt::Display,
{
    fn scale(self, scalar: Scalar) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn elem_mul(self, rhs: Self) -> Self;

    fn from_vec(input: Vec<Scalar>) -> Self;
    fn to_vec(&self) -> Vec<Scalar>;

    fn partial_sum(&self) -> Scalar;

    fn magnitude(&self) -> Scalar {
        self.clone().elem_mul(self.clone()).partial_sum().sqrt()
    }

    fn dot(self, rhs: Self) -> Scalar {
        self.elem_mul(rhs).partial_sum()
    }

    fn dot_debug(self, rhs: Self) -> Scalar {
        let muld = self.elem_mul(rhs);
        println!("{:?}", muld.to_vec());
        muld.partial_sum()
    }
}

pub trait Matrix<Scalar: Float = f32>: Sized
where
    Scalar: Debug + std::fmt::Display,
{
    type Line: Vector<Scalar> + Debug;
    type LineIter<'a>: Iterator<Item = Self::Line>
    where
        Self: 'a;
    fn from_vectors(input: Vec<Self::Line>) -> Option<Self>;

    fn dimensions(&self) -> (usize, usize);
    fn row(&self, index: usize) -> Option<Self::Line>;
    fn column(&self, index: usize) -> Option<Self::Line>;

    fn row_iter<'a>(&'a self) -> Self::LineIter<'a>;
    fn col_iter<'a>(&'a self) -> Self::LineIter<'a>;

    fn scale(&self, rhs: Scalar) -> Self {
        Self::from_vectors(self.row_iter().map(|i| i.scale(rhs.clone())).collect()).unwrap()
    }
    fn mul(self, rhs: Self) -> Option<Self> {
        if self.dimensions().1 != rhs.dimensions().0 {
            return None;
        }
        Some(Self::from_vectors(
            self.row_iter()
                .map(|i| Self::Line::from_vec(rhs.col_iter().map(|j| j.dot(i.clone())).collect()))
                .collect(),
        )?)
    }

    fn mul_debug(self, rhs: Self) -> Option<Self> {
        if self.dimensions().1 != rhs.dimensions().0 {
            return None;
        }
        let vectors: Vec<Self::Line> = self
            .row_iter()
            .map(|i| {
                let output: Vec<Scalar> = rhs
                    .col_iter()
                    .map(|j| {
                        let output = j.clone().dot_debug(i.clone());
                        println!(
                            "Dot product of {:?}, {:?} is {}",
                            j.to_vec(),
                            i.to_vec(),
                            output.clone()
                        );
                        output
                    })
                    .collect();
                println!("{:?}", output.clone());
                let out_simd = Self::Line::from_vec(output);
                out_simd
            })
            .collect();

        println!("{:?}", vectors.clone());
        Some(Self::from_vectors(vectors).unwrap())
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
