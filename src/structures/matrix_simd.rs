use std::{
    ops::{Add, Div, Mul, Sub},
    simd::{f32x1, f32x16, f32x2, f32x32, f32x4, f32x64, f32x8, SimdFloat},
};

pub struct SimdVector {
    size_64: Vec<f32x64>,
    size_32: Option<f32x32>,
    size_16: Option<f32x16>,
    size_8: Option<f32x8>,
    size_4: Option<f32x4>,
    size_2: Option<f32x2>,
    size_1: Option<f32x1>,
}

impl SimdVector {
    pub fn len(&self) -> usize {
        self.size_1.map_or(0, |i| 1)
            + self.size_2.map_or(0, |i| i.lanes())
            + self.size_4.map_or(0, |i| i.lanes())
            + self.size_8.map_or(0, |i| i.lanes())
            + self.size_16.map_or(0, |i| i.lanes())
            + self.size_32.map_or(0, |i| i.lanes())
            + self.size_64.len() * 64
    }

    pub fn sum(&self) -> f32 {
        self.size_2.map_or(0., |i| i.reduce_sum())
            + self.size_4.map_or(0., |i| i.reduce_sum())
            + self.size_8.map_or(0., |i| i.reduce_sum())
            + self.size_16.map_or(0., |i| i.reduce_sum())
            + self.size_32.map_or(0., |i| i.reduce_sum())
            + self
                .size_64
                .iter()
                .fold(f32x64::splat(0.), |i, j| i + j)
                .reduce_sum()
    }

    fn bitmask(&self) -> usize {
        self.size_32.map_or(0, |i| 0b100000)
            | self.size_16.map_or(0, |i| 0b010000)
            | self.size_8.map_or(0, |i| 0b001000)
            | self.size_4.map_or(0, |i| 0b000100)
            | self.size_2.map_or(0, |i| 0b000010)
            | self.size_1.map_or(0, |i| 0b000001)
    }

    pub fn get(&self, index: usize) -> Option<f32> {
        let bitmask = self.bitmask();
        self.size_64
            .iter()
            .map(|i| i.as_array())
            .flatten()
            .collect::<Vec<&f32>>()
            .get(index)
            .copied()
            .copied()
            .or_else(|| {
                let current = index - (64 * self.size_64.len());
                match (current ^ bitmask).next_power_of_two() >> 1 {
                    32 => self.size_32.unwrap().as_array().get(current).copied(),
                    16 => self
                        .size_16
                        .unwrap()
                        .as_array()
                        .get(current - (bitmask & 32))
                        .copied(),
                    8 => self
                        .size_8
                        .unwrap()
                        .as_array()
                        .get(current - (bitmask & 0b110000))
                        .copied(),
                    4 => self
                        .size_4
                        .unwrap()
                        .as_array()
                        .get(current - (bitmask & 0b111000))
                        .copied(),
                    2 => self
                        .size_2
                        .unwrap()
                        .as_array()
                        .get(current - (bitmask & 0b111100))
                        .copied(),
                    1 => self
                        .size_1
                        .unwrap()
                        .as_array()
                        .get(current - (bitmask & 0b111110))
                        .copied(),
                    _ => None,
                }
            })
    }

    pub fn from_vector(vec: Vec<f32>) -> SimdVector {
        let mut remainder = vec.chunks_exact(64).remainder();
        SimdVector {
            size_64: vec
                .chunks_exact(64)
                .map(|i| f32x64::from_slice(i))
                .collect(),
            size_32: remainder
                .take(..=32)
                .map_or(None, |i| Some(f32x32::from_slice(i))),
            size_16: remainder
                .take(..=16)
                .map_or(None, |i| Some(f32x16::from_slice(i))),
            size_8: remainder
                .take(..=8)
                .map_or(None, |i| Some(f32x8::from_slice(i))),
            size_4: remainder
                .take(..=4)
                .map_or(None, |i| Some(f32x4::from_slice(i))),
            size_2: remainder
                .take(..=2)
                .map_or(None, |i| Some(f32x2::from_slice(i))),
            size_1: remainder
                .take_first()
                .map_or(None, |i| Some(f32x1::from_slice(&[*i]))),
        }
    }
}

impl Add for SimdVector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        SimdVector {
            size_64: self
                .size_64
                .iter()
                .take(rhs.size_64.len())
                .enumerate()
                .map(|(i, j)| j + rhs.size_64.get(i).unwrap())
                .collect(),
            size_32: self.size_32.map_or(rhs.size_32, |i| {
                rhs.size_32.map_or(self.size_32, |j| Some(i + j))
            }),
            size_16: self.size_16.map_or(rhs.size_16, |i| {
                rhs.size_16.map_or(self.size_16, |j| Some(i + j))
            }),
            size_8: self.size_8.map_or(rhs.size_8, |i| {
                rhs.size_8.map_or(self.size_8, |j| Some(i + j))
            }),
            size_4: self.size_4.map_or(rhs.size_4, |i| {
                rhs.size_4.map_or(self.size_4, |j| Some(i + j))
            }),
            size_2: self.size_2.map_or(rhs.size_2, |i| {
                rhs.size_2.map_or(self.size_2, |j| Some(i + j))
            }),
            size_1: self.size_1.map_or(rhs.size_1, |i| {
                rhs.size_1.map_or(self.size_1, |j| Some(i + j))
            }),
        }
    }
}

impl Mul for SimdVector {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        SimdVector {
            size_64: self
                .size_64
                .iter()
                .take(rhs.size_64.len())
                .enumerate()
                .map(|(i, j)| j * rhs.size_64.get(i).unwrap())
                .collect(),
            size_32: self.size_32.map_or(rhs.size_32, |i| {
                rhs.size_32.map_or(self.size_32, |j| Some(i * j))
            }),
            size_16: self.size_16.map_or(rhs.size_16, |i| {
                rhs.size_16.map_or(self.size_16, |j| Some(i * j))
            }),
            size_8: self.size_8.map_or(rhs.size_8, |i| {
                rhs.size_8.map_or(self.size_8, |j| Some(i * j))
            }),
            size_4: self.size_4.map_or(rhs.size_4, |i| {
                rhs.size_4.map_or(self.size_4, |j| Some(i * j))
            }),
            size_2: self.size_2.map_or(rhs.size_2, |i| {
                rhs.size_2.map_or(self.size_2, |j| Some(i * j))
            }),
            size_1: self.size_1.map_or(rhs.size_1, |i| {
                rhs.size_1.map_or(self.size_1, |j| Some(i * j))
            }),
        }
    }
}

impl Div for SimdVector {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        SimdVector {
            size_64: self
                .size_64
                .iter()
                .take(rhs.size_64.len())
                .enumerate()
                .map(|(i, j)| j / rhs.size_64.get(i).unwrap())
                .collect(),
            size_32: self.size_32.map_or(rhs.size_32, |i| {
                rhs.size_32.map_or(self.size_32, |j| Some(j / i))
            }),
            size_16: self.size_16.map_or(rhs.size_16, |i| {
                rhs.size_16.map_or(self.size_16, |j| Some(j / i))
            }),
            size_8: self.size_8.map_or(rhs.size_8, |i| {
                rhs.size_8.map_or(self.size_8, |j| Some(j / i))
            }),
            size_4: self.size_4.map_or(rhs.size_4, |i| {
                rhs.size_4.map_or(self.size_4, |j| Some(j / i))
            }),
            size_2: self.size_2.map_or(rhs.size_2, |i| {
                rhs.size_2.map_or(self.size_2, |j| Some(j / i))
            }),
            size_1: self.size_1.map_or(rhs.size_1, |i| {
                rhs.size_1.map_or(self.size_1, |j| Some(j / i))
            }),
        }
    }
}

impl Sub for SimdVector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        SimdVector {
            size_64: self
                .size_64
                .iter()
                .take(rhs.size_64.len())
                .enumerate()
                .map(|(i, j)| j - rhs.size_64.get(i).unwrap())
                .collect(),
            size_32: self.size_32.map_or(rhs.size_32, |i| {
                rhs.size_32.map_or(self.size_32, |j| Some(j - i))
            }),
            size_16: self.size_16.map_or(rhs.size_16, |i| {
                rhs.size_16.map_or(self.size_16, |j| Some(j - i))
            }),
            size_8: self.size_8.map_or(rhs.size_8, |i| {
                rhs.size_8.map_or(self.size_8, |j| Some(j - i))
            }),
            size_4: self.size_4.map_or(rhs.size_4, |i| {
                rhs.size_4.map_or(self.size_4, |j| Some(j - i))
            }),
            size_2: self.size_2.map_or(rhs.size_2, |i| {
                rhs.size_2.map_or(self.size_2, |j| Some(j - i))
            }),
            size_1: self.size_1.map_or(rhs.size_1, |i| {
                rhs.size_1.map_or(self.size_1, |j| Some(j - i))
            }),
        }
    }
}
