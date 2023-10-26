use std::{
    marker::Tuple,
    simd::{f32x1, f32x16, f32x2, f32x32, f32x4, f32x64, f32x8},
};

pub trait SimdExcess {
    type Excess: Tuple;
}

impl SimdExcess for f32x1 {
    type Excess = ();
}

impl SimdExcess for f32x2 {
    type Excess = (f32x1);
}

impl SimdExcess for f32x4 {
    type Excess = (f32x2, f32x1);
}

impl SimdExcess for f32x8 {
    type Excess = (f32x4, f32x2, f32x1);
}

impl SimdExcess for f32x16 {
    type Excess = (f32x8, f32x4, f32x1);
}

impl SimdExcess for f32x32 {
    type Excess = (f32x16, f32x8, f32x4, f32x1);
}

impl SimdExcess for f32x64 {
    type Excess = (f32x32, f32x16, f32x8, f32x4, f32x2, f32x1);
}
