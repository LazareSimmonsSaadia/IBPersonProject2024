use opencl3::{
    context::Context,
    memory::{create_buffer, Buffer},
    svm::SvmVec,
    types::cl_float,
};
use std::{
    ffi::c_void,
    simd::{prelude::Simd, LaneCount, SimdElement, SupportedLaneCount},
};

pub enum Matrix<'a, const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    SIMD(Simd<f32, N>),
    OpenCL(SvmVec<'a, cl_float>),
}

impl<'a, const N: usize> Matrix<'a, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn from(input: Vec<f32>, opencl_context: Option<&Context>) -> Matrix<N> {
        let len = input.len();
        if (len & (len - 1) == 0 && len <= 64) {
            Matrix::SIMD(Simd::from_slice(input.as_slice()))
        } else {
            match opencl_context {
                Some(context) => Matrix::OpenCL(svmalloc(input, context)),
                None => {
                    panic!("OpenCL context is required to create an OpenCl buffer")
                }
            }
        }
    }
}

fn svmalloc(vec: Vec<f32>, context: &Context) -> SvmVec<cl_float> {
    let mut svm = SvmVec::<cl_float>::new(context);
    svm.clone_from_slice(vec.as_slice());
    svm
}
