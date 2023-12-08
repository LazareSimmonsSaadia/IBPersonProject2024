pub mod vector_simd {
    #[cfg(feature = "async")]
    pub mod asynchronous {
        use crate::structures::vector_simd::SimdVector;
        use rayon::prelude::*;
        use std::simd::{f32x1, f32x16, f32x2, f32x32, f32x4, f32x64, f32x8, SimdFloat};

        impl SimdVector {
            pub async fn len_async(&self) -> usize {
                async { self.size_1.map_or(0, |_i| 1) }.await
                    + async { self.size_2.map_or(0, |i| i.lanes()) }.await
                    + async { self.size_4.map_or(0, |i| i.lanes()) }.await
                    + async { self.size_8.map_or(0, |i| i.lanes()) }.await
                    + async { self.size_16.map_or(0, |i| i.lanes()) }.await
                    + async { self.size_32.map_or(0, |i| i.lanes()) }.await
                    + async { self.size_64.len() * 64 }.await
            }

            pub async fn sum_async(&self) -> f32 {
                async { self.size_1.map_or(0., |i| i.reduce_sum()) }.await
                    + async { self.size_2.map_or(0., |i| i.reduce_sum()) }.await
                    + async { self.size_4.map_or(0., |i| i.reduce_sum()) }.await
                    + async { self.size_8.map_or(0., |i| i.reduce_sum()) }.await
                    + async { self.size_16.map_or(0., |i| i.reduce_sum()) }.await
                    + async { self.size_32.map_or(0., |i| i.reduce_sum()) }.await
                    + async { self.size_64.par_iter().sum::<f32x64>().reduce_sum() }.await
            }

            pub async fn scale_async(self, rhs: f32) -> SimdVector {
                SimdVector {
                    size_64: async {
                        self.size_64
                            .par_iter()
                            .map(|i| i * f32x64::splat(rhs))
                            .collect()
                    }
                    .await,
                    size_32: async { self.size_32.map(|i| i * f32x32::splat(rhs)) }.await,
                    size_16: async { self.size_16.map(|i| i * f32x16::splat(rhs)) }.await,
                    size_8: async { self.size_8.map(|i| i * f32x8::splat(rhs)) }.await,
                    size_4: async { self.size_4.map(|i| i * f32x4::splat(rhs)) }.await,
                    size_2: async { self.size_2.map(|i| i * f32x2::splat(rhs)) }.await,
                    size_1: async { self.size_1.map(|i| i * f32x1::splat(rhs)) }.await,
                }
            }

            //TODO: fix faster get function
            // fn bitmask(&self) -> usize {
            //     self.size_32.map_or(0, |_i| 0b100000)
            //         | self.size_16.map_or(0, |_i| 0b010000)
            //         | self.size_8.map_or(0, |_i| 0b001000)
            //         | self.size_4.map_or(0, |_i| 0b000100)
            //         | self.size_2.map_or(0, |_i| 0b000010)
            //         | self.size_1.map_or(0, |_i| 0b000001)
            // }

            // fn get_broken(&self, index: usize) -> Option<f32> {
            //     if self.len() == 0 {
            //         return None;
            //     }
            //     let bitmask = self.bitmask();
            //     self.size_64
            //         .iter()
            //         .map(|i| i.as_array())
            //         .flatten()
            //         .collect::<Vec<&f32>>()
            //         .get(index)
            //         .copied()
            //         .copied()
            //         .or_else(|| {
            //             let current = index - (64 * self.size_64.len());
            //             let array = (current ^ bitmask).next_power_of_two() >> 1;
            //             match array {
            //                 32 => self.size_32.unwrap().as_array().get(current).copied(),
            //                 16 => self
            //                     .size_16
            //                     .unwrap()
            //                     .as_array()
            //                     .get(current - (current << current.leading_ones()))
            //                     .copied(),
            //                 8 => self
            //                     .size_8
            //                     .unwrap()
            //                     .as_array()
            //                     .get(current - (bitmask & 0b110000))
            //                     .copied(),
            //                 4 => self
            //                     .size_4
            //                     .unwrap()
            //                     .as_array()
            //                     .get(current - (bitmask & 0b111000))
            //                     .copied(),
            //                 2 => self
            //                     .size_2
            //                     .unwrap()
            //                     .as_array()
            //                     .get(current - (bitmask & 0b111100))
            //                     .copied(),
            //                 1 => self
            //                     .size_1
            //                     .unwrap()
            //                     .as_array()
            //                     .get(current - (bitmask & 0b111110))
            //                     .copied(),
            //                 _ => None,
            //             }
            //         })
            // }
        }
    }
}

pub mod matrix_simd {
    use crate::structures::matrix_simd::*;
    use crate::structures::vector_simd::SimdVector;
    use rayon::prelude::*;
    impl SimdMatrix {
        pub async fn from_async(input: Vec<Vec<f32>>) -> Result<SimdMatrix, MatrixCreationError> {
            let row = async { input.get(0).unwrap().len() }.await;
            let lengths = async { input.par_iter().map(Vec::<f32>::len) };
            if lengths.await.all(|i| i == row) {
                let matrix = async {
                    SimdMatrix {
                        matrix: async {
                            input
                                .par_iter()
                                .map(|i| SimdVector::from_vector(i.to_owned()))
                                .collect()
                        }
                        .await,
                        row_size: row,
                    }
                };
                Ok(matrix.await)
            } else {
                Err(MatrixCreationError::InconsistentRowLengthErr)
            }
        }

        pub async fn from_simd_async(
            input: Vec<SimdVector>,
        ) -> Result<SimdMatrix, MatrixCreationError> {
            let row = async { input.get(0).unwrap().len() }.await;
            let lengths = async { input.par_iter().map(SimdVector::len) };
            if lengths.await.all(|i| i == row) {
                let matrix = async {
                    SimdMatrix {
                        matrix: input,
                        row_size: row,
                    }
                };
                Ok(matrix.await)
            } else {
                Err(MatrixCreationError::InconsistentRowLengthErr)
            }
        }
    }
}
