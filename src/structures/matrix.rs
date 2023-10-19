use opencl3::{context::Context, svm::SvmVec, types::cl_float};
use std::simd::{prelude::Simd, LaneCount, SupportedLaneCount};

pub enum MatrixNxN<'a, const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    SIMD(Vec<Simd<f32, N>>),
    OpenCL(Vec<SvmVec<'a, cl_float>>),
    Naive(Vec<Vec<f32>>),
}

enum MatrixTypes {
    SIMD = 1,
    OpenCL = 3,
    Naive = 2,
}

impl<'a, const N: usize> MatrixNxN<'a, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn from(
        input1: &'a [Vec<f32>],
        input2: &'a [Vec<f32>],
        opencl_context: Option<&'a Context>,
    ) -> (MatrixNxN<'a, N>, MatrixNxN<'a, N>) {
        let backend = input1
            .iter()
            .map(|i| {
                if i.len() & (i.len() - 1) == 0 && i.len() <= 64 {
                    MatrixTypes::SIMD
                } else if let None = opencl_context && i.len() >= 100 {
                    MatrixTypes::OpenCL
                } else {
                    MatrixTypes::Naive
                }
            })
            .fold(1, |i, j| (i as i32).max(j as i32))
            .max(
                input2
                    .iter()
                    .map(|i| {
                        if (i.len() & (i.len() - 1) == 0 && i.len() <= 64) {
                            MatrixTypes::SIMD
                        } else if let None = opencl_context && i.len() >= 100 {
                            MatrixTypes::OpenCL
                        } else {
                            MatrixTypes::Naive
                        }
                    })
                    .fold(1, |i, j| (i as i32).max(j as i32)),
            );
        match backend {
            1 => (MatrixNxN::new_simd(input1), MatrixNxN::new_simd(input2)),
            2 => (MatrixNxN::new_naive(input1), MatrixNxN::new_naive(input2)),
            3 => match opencl_context {
                Some(context) => (
                    MatrixNxN::new_opencl(input1, context),
                    MatrixNxN::new_opencl(input2, context),
                ),
                None => (MatrixNxN::new_naive(input1), MatrixNxN::new_naive(input2)),
            },
            _ => panic!("MatrixNxN::from(): non-type integer passed to backend match\n")

        }
    }

    pub fn new_simd(input: &'a [Vec<f32>]) -> MatrixNxN<'a, N> {
        MatrixNxN::SIMD(
            input
                .iter()
                .map(|i| Simd::from_slice(i.as_slice()))
                .collect(),
        )
    }

    pub fn new_opencl(input: &'a [Vec<f32>], context: &'a Context) -> MatrixNxN<'a, N> {
        MatrixNxN::OpenCL(input.iter().map(|i| svmalloc(i, context)).collect())
    }

    pub fn new_naive(input: &'a [Vec<f32>]) -> MatrixNxN<'a, N> {
        MatrixNxN::Naive(input.to_vec())
    }
}

fn svmalloc<'a>(vec: &'a Vec<f32>, context: &'a Context) -> SvmVec<'a, cl_float> {
    let mut svm = SvmVec::<cl_float>::new(context);
    svm.clone_from_slice(vec.as_slice());
    svm
}
