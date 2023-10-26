use crate::structures::matrix::Matrix;
use gcd::binary_usize;
use opencl3::{context::Context, svm::SvmVec, types::cl_float};
use std::simd::{prelude::Simd, LaneCount, SimdFloat, SupportedLaneCount};

pub enum Argument<'a, SimdType: SimdFloat> {
    SIMD((Vec<SimdType>, usize), (Vec<SimdType>, usize)),
    OpenCL(Vec<SvmVec<'a, cl_float>>, Vec<SvmVec<'a, cl_float>>),
    Naive(Matrix, Matrix),
}

enum MatrixTypes {
    SIMD = 1,
    OpenCL = 3,
    Naive = 2,
}

impl<'a, SimdType: SimdFloat> Argument<'a, SimdType> {
    pub fn from(
        input1: &'a Matrix,
        input2: &'a Matrix,
        opencl_context: Option<&'a Context>,
    ) -> Argument<'a, SimdType> {
        let backend = (if input1.row_size & (input1.row_size - 1) == 0 {
            MatrixTypes::SIMD
        } else if input1.vector_flat.len() >= 200 {
            MatrixTypes::OpenCL
        } else {
            MatrixTypes::Naive
        } as i32)
            .max(
                (if input1.row_size & (input1.row_size - 1) == 0 {
                    MatrixTypes::SIMD
                } else if input1.vector_flat.len() >= 200 {
                    MatrixTypes::OpenCL
                } else {
                    MatrixTypes::Naive
                } as i32),
            );
        match backend {
            // 1 => Argument:: 
        }
    }

    pub fn new_simd(input: &'a [Vec<f32>], input2: &'a [Vec<f32>]) -> Argument<'a, N> {
        Argument::SIMD(
            input
                .iter()
                .map(|i| Simd::from_slice(i.as_slice()))
                .collect(),
            input2
                .iter()
                .map(|i| Simd::from_slice(i.as_slice()))
                .collect(),
        )
    }

    pub fn new_opencl(
        input: &'a [Vec<f32>],
        input2: &'a [Vec<f32>],
        context: &'a Context,
    ) -> Argument<'a, N> {
        Argument::OpenCL(
            input.iter().map(|i| svmalloc(i, context)).collect(),
            input2.iter().map(|i| svmalloc(i, context)).collect(),
        )
    }

    pub fn new_naive(input: &'a [Vec<f32>], input2: &'a [Vec<f32>]) -> Argument<'a, N> {
        Argument::Naive(input.to_vec(), input2.to_vec())
    }
}

fn svmalloc<'a>(vec: &'a Vec<f32>, context: &'a Context) -> SvmVec<'a, cl_float> {
    let mut svm = SvmVec::<cl_float>::new(context);
    svm.clone_from_slice(vec.as_slice());
    svm
}
