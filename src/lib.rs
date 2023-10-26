#![feature(tuple_trait)]
#![feature(let_chains)]
#![feature(iter_collect_into)]
#![feature(portable_simd)]
pub mod algebra;
mod opencl;
pub mod structures;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
