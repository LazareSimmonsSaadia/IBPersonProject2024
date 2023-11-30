use std::{
    ops::{Add, Div, Mul, Sub},
    simd::{f32x1, f32x16, f32x2, f32x32, f32x4, f32x64, f32x8, SimdFloat},
};

#[derive(Debug, Clone)]
pub struct SimdVector {
    size_64: Vec<f32x64>,
    size_32: Option<f32x32>,
    size_16: Option<f32x16>,
    size_8: Option<f32x8>,
    size_4: Option<f32x4>,
    size_2: Option<f32x2>,
    size_1: Option<f32x1>,
}

pub struct VectorIter<'a> {
    parent: &'a SimdVector,
    count: usize,
}

impl SimdVector {
    pub fn len(&self) -> usize {
        self.size_1.map_or(0, |_i| 1)
            + self.size_2.map_or(0, |i| i.lanes())
            + self.size_4.map_or(0, |i| i.lanes())
            + self.size_8.map_or(0, |i| i.lanes())
            + self.size_16.map_or(0, |i| i.lanes())
            + self.size_32.map_or(0, |i| i.lanes())
            + self.size_64.len() * 64
    }

    pub fn iter(&self) -> VectorIter {
        VectorIter {
            parent: (self),
            count: (0),
        }
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
        self.size_32.map_or(0, |_i| 0b100000)
            | self.size_16.map_or(0, |_i| 0b010000)
            | self.size_8.map_or(0, |_i| 0b001000)
            | self.size_4.map_or(0, |_i| 0b000100)
            | self.size_2.map_or(0, |_i| 0b000010)
            | self.size_1.map_or(0, |_i| 0b000001)
    }

    fn get_broken(&self, index: usize) -> Option<f32> {
        if self.len() == 0 {
            return None;
        }
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
                let array = (current ^ bitmask).next_power_of_two() >> 1;
                match array {
                    32 => self.size_32.unwrap().as_array().get(current).copied(),
                    16 => self
                        .size_16
                        .unwrap()
                        .as_array()
                        .get(current - (current << current.leading_ones()))
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

    pub fn get(&self, index: usize) -> Option<f32> {
        self.size_64
            .iter()
            .map(|i| i.as_array())
            .flatten()
            .collect::<Vec<&f32>>()
            .get(index)
            .copied()
            .copied()
            .or_else(|| {
                let mut current = index - (64 * self.size_64.len());
                if self.size_32.is_some() {
                    if current < 32 {
                        return self.size_32.unwrap().as_array().get(current).copied();
                    } else {
                        current -= 32;
                    }
                }
                if self.size_16.is_some() {
                    if current < 16 {
                        return self.size_16.unwrap().as_array().get(current).copied();
                    } else {
                        current -= 16;
                    }
                }
                if self.size_8.is_some() {
                    if current < 8 {
                        return self.size_8.unwrap().as_array().get(current).copied();
                    } else {
                        current -= 8;
                    }
                }
                if self.size_4.is_some() {
                    if current < 4 {
                        return self.size_4.unwrap().as_array().get(current).copied();
                    } else {
                        current -= 4;
                    }
                }
                if self.size_2.is_some() {
                    if current < 2 {
                        return self.size_2.unwrap().as_array().get(current).copied();
                    } else {
                        current -= 2;
                    }
                }
                if self.size_1.is_some() {
                    if current < 1 {
                        return self.size_1.unwrap().as_array().get(current).copied();
                    } else {
                        current -= 1;
                    }
                }
                None
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
                .take(..32)
                .map_or(None, |i| Some(f32x32::from_slice(i))),
            size_16: remainder
                .take(..16)
                .map_or(None, |i| Some(f32x16::from_slice(i))),
            size_8: remainder
                .take(..8)
                .map_or(None, |i| Some(f32x8::from_slice(i))),
            size_4: remainder
                .take(..4)
                .map_or(None, |i| Some(f32x4::from_slice(i))),
            size_2: remainder
                .take(..2)
                .map_or(None, |i| Some(f32x2::from_slice(i))),
            size_1: remainder
                .take_first()
                .map_or(None, |i| Some(f32x1::from_slice(&[*i]))),
        }
    }

    pub fn to_vector(&self) -> Vec<f32> {
        let mut out = self
            .size_64
            .iter()
            .map(|i| i.as_array())
            .flatten()
            .copied()
            .collect::<Vec<f32>>();
        out.extend_from_slice(
            self.size_32
                .map_or(vec![], |i| i.as_array().to_vec())
                .as_slice(),
        );
        out.extend_from_slice(
            self.size_16
                .map_or(vec![], |i| i.as_array().to_vec())
                .as_slice(),
        );
        out.extend_from_slice(
            self.size_8
                .map_or(vec![], |i| i.as_array().to_vec())
                .as_slice(),
        );
        out.extend_from_slice(
            self.size_4
                .map_or(vec![], |i| i.as_array().to_vec())
                .as_slice(),
        );
        out.extend_from_slice(
            self.size_2
                .map_or(vec![], |i| i.as_array().to_vec())
                .as_slice(),
        );
        out.extend_from_slice(
            self.size_1
                .map_or(vec![], |i| i.as_array().to_vec())
                .as_slice(),
        );
        out
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
                rhs.size_32.map_or(self.size_32, |j| Some(j * i))
            }),
            size_16: self.size_16.map_or(rhs.size_16, |i| {
                rhs.size_16.map_or(self.size_16, |j| Some(j * i))
            }),
            size_8: self.size_8.map_or(rhs.size_8, |i| {
                rhs.size_8.map_or(self.size_8, |j| Some(j * i))
            }),
            size_4: self.size_4.map_or(rhs.size_4, |i| {
                rhs.size_4.map_or(self.size_4, |j| Some(j * i))
            }),
            size_2: self.size_2.map_or(rhs.size_2, |i| {
                rhs.size_2.map_or(self.size_2, |j| Some(j * i))
            }),
            size_1: self.size_1.map_or(rhs.size_1, |i| {
                rhs.size_1.map_or(self.size_1, |j| Some(j * i))
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
                rhs.size_32.map_or(self.size_32, |j| Some(i / j))
            }),
            size_16: self.size_16.map_or(rhs.size_16, |i| {
                rhs.size_16.map_or(self.size_16, |j| Some(i / j))
            }),
            size_8: self.size_8.map_or(rhs.size_8, |i| {
                rhs.size_8.map_or(self.size_8, |j| Some(i / j))
            }),
            size_4: self.size_4.map_or(rhs.size_4, |i| {
                rhs.size_4.map_or(self.size_4, |j| Some(i / j))
            }),
            size_2: self.size_2.map_or(rhs.size_2, |i| {
                rhs.size_2.map_or(self.size_2, |j| Some(i / j))
            }),
            size_1: self.size_1.map_or(rhs.size_1, |i| {
                rhs.size_1.map_or(self.size_1, |j| Some(i / j))
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
                rhs.size_32.map_or(self.size_32, |j| Some(i - j))
            }),
            size_16: self.size_16.map_or(rhs.size_16, |i| {
                rhs.size_16.map_or(self.size_16, |j| Some(i - j))
            }),
            size_8: self.size_8.map_or(rhs.size_8, |i| {
                rhs.size_8.map_or(self.size_8, |j| Some(i - j))
            }),
            size_4: self.size_4.map_or(rhs.size_4, |i| {
                rhs.size_4.map_or(self.size_4, |j| Some(i - j))
            }),
            size_2: self.size_2.map_or(rhs.size_2, |i| {
                rhs.size_2.map_or(self.size_2, |j| Some(i - j))
            }),
            size_1: self.size_1.map_or(rhs.size_1, |i| {
                rhs.size_1.map_or(self.size_1, |j| Some(i - j))
            }),
        }
    }
}

impl<'a> Iterator for VectorIter<'a> {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        self.parent.get(self.count)
    }
}

impl Mul<f32> for SimdVector {
    type Output = SimdVector;

    fn mul(self, rhs: f32) -> Self::Output {
        SimdVector {
            size_64: self
                .size_64
                .as_slice()
                .iter()
                .map(|i| i * f32x64::splat(rhs))
                .collect(),
            size_32: self.size_32.map(|i| i * f32x32::splat(rhs)),
            size_16: self.size_16.map(|i| i * f32x16::splat(rhs)),
            size_8: self.size_8.map(|i| i * f32x8::splat(rhs)),
            size_4: self.size_4.map(|i| i * f32x4::splat(rhs)),
            size_2: self.size_2.map(|i| i * f32x2::splat(rhs)),
            size_1: self.size_1.map(|i| i * f32x1::splat(rhs)),
        }
    }
}

impl Mul<SimdVector> for f32 {
    type Output = SimdVector;

    fn mul(self, rh: SimdVector) -> Self::Output {
        let rhs = self;
        let slf = rh;
        SimdVector {
            size_64: slf
                .size_64
                .as_slice()
                .iter()
                .map(|i| i * f32x64::splat(rhs))
                .collect(),
            size_32: slf.size_32.map(|i| i * f32x32::splat(rhs)),
            size_16: slf.size_16.map(|i| i * f32x16::splat(rhs)),
            size_8: slf.size_8.map(|i| i * f32x8::splat(rhs)),
            size_4: slf.size_4.map(|i| i * f32x4::splat(rhs)),
            size_2: slf.size_2.map(|i| i * f32x2::splat(rhs)),
            size_1: slf.size_1.map(|i| i * f32x1::splat(rhs)),
        }
    }
}
