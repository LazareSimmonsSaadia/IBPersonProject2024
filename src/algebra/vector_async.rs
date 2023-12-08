use std::{async_iter::AsyncIterator, fmt::Debug};

use async_trait_fn::{async_trait, unboxed};
use futures::future::join_all;
use num::Float;

#[async_trait]
pub trait AsyncVector<Scalar: Float = f32>
where
    Self: Sized + Clone + Send + Sync,
    Scalar: Debug + std::fmt::Display + Send,
{
    async fn scale(self, scalar: Scalar) -> Self;
    async fn add(self, rhs: Self) -> Self;
    async fn elem_mul(self, rhs: Self) -> Self;

    async fn from_vec(input: Vec<Scalar>) -> Self;
    async fn to_vec(&self) -> Vec<Scalar>;

    async fn partial_sum(&self) -> Scalar;

    async fn magnitude(&self) -> Scalar {
        self.clone()
            .elem_mul(self.clone())
            .await
            .partial_sum()
            .await
            .sqrt()
    }

    async fn dot(self, rhs: Self) -> Scalar {
        self.elem_mul(rhs).await.partial_sum().await
    }
}

#[async_trait]
pub trait AsyncMatrix<Scalar: Float = f32>: Sized
where
    Scalar: Debug + std::fmt::Display + Send + Sync + Copy,
    Self: Copy,
{
    type Line: AsyncVector<Scalar> + Debug;
    type LineIter<'a>: Iterator<Item = Self::Line>
    where
        Self: 'a;
    async fn from_vectors(input: Vec<Self::Line>) -> Option<Self>;

    fn dimensions(&self) -> (usize, usize);
    async fn row(&self, index: usize) -> Option<Self::Line>;
    async fn column(&self, index: usize) -> Option<Self::Line>;

    fn row_iter<'a>(&'a self) -> Self::LineIter<'a>;
    fn col_iter<'a>(&'a self) -> Self::LineIter<'a>;

    async fn scale<'a>(&self, rhs: &'a Scalar) -> Self {
        Self::from_vectors(
            join_all(
                self.row_iter()
                    .map(|i| async { i.scale(rhs.clone()).await })
                    .collect::<Vec<_>>(),
            )
            .await,
        )
        .await
        .unwrap()
    }
    async fn mul(self, rhs: Self) -> Option<Self> {
        if self.dimensions().1 != rhs.dimensions().0 {
            return None;
        }
        Some(
            Self::from_vectors(
                join_all(
                    self.row_iter()
                        .map(|i| async move {
                            Self::Line::from_vec(
                                join_all(
                                    rhs.col_iter()
                                        .map(|j| async { j.dot(i.clone()).await })
                                        .collect::<Vec<_>>(),
                                )
                                .await,
                            )
                            .await
                        })
                        .collect::<Vec<_>>(),
                )
                .await,
            )
            .await?,
        )
    }

    async fn add(self, rhs: Self) -> Option<Self> {
        if self.dimensions() != rhs.dimensions() {
            return None;
        }
        Self::from_vectors(
            join_all(
                self.row_iter()
                    .zip(rhs.col_iter())
                    .map(|(i, j)| async { i.add(j).await })
                    .collect::<Vec<_>>(),
            )
            .await,
        )
        .await
    }
}
