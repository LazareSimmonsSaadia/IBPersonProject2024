#![feature(slice_take)]
#![feature(tuple_trait)]
#![feature(let_chains)]
#![feature(iter_collect_into)]
#![feature(portable_simd)]
pub mod algebra;
mod opencl;
pub mod structures;

#[cfg(test)]
mod tests {

    use crate::structures::{matrix_simd::SimdMatrix, vector_simd::SimdVector};

    #[test]
    fn check_create_simd_vector() {
        let init = vec![5.23, 3.2, 0.44, 8.9, 9.0, 5.5, 2.0];
        let mut init_large: Vec<f32> = vec![];
        init_large.resize(335, 1.1);
        let vector = SimdVector::from_vector(init.clone());
        let vector_large = SimdVector::from_vector(init_large.clone());
        println!("{:?}", vector);
        assert_eq!(init, vector.to_vector());
        assert_eq!(init_large, vector_large.to_vector());
    }

    #[test]
    fn check_create_simd_matrix() {
        let init_working = vec![
            vec![0.44, 5.55, 77.4, 9.3],
            vec![2.3, 77.3, 0., 4.5],
            vec![88.9, 9.0, 32.1, 0.9],
        ];
        let init_fail = vec![
            vec![0.44, 5.55, 77.4, 9.3, 7.8],
            vec![2.3, 77.3, 0., 4.5],
            vec![88.9, 9.0, 32.1, 0.9],
        ];
        let matrix_working = SimdMatrix::from(init_working.clone());
        let matrix_fail = SimdMatrix::from(init_fail.clone());

        assert!(matrix_fail.is_err());
        assert_eq!(matrix_working.unwrap().to_vector(), init_working);
    }

    #[test]
    fn check_simd_len_fns() {
        let init = vec![5.23, 3.2, 0.44, 8.9, 9.0, 5.5, 2.0];
        let vector = SimdVector::from_vector(init.clone());
        assert_eq!(init.len(), vector.len());
        let init_working = vec![
            vec![0.44, 5.55, 77.4, 9.3],
            vec![2.3, 77.3, 0., 4.5],
            vec![88.9, 9.0, 32.1, 0.9],
        ];
        let matrix_working = SimdMatrix::from(init_working.clone());
        let mut init_large: Vec<f32> = vec![];
        init_large.resize(675, 1.1);
        let vec_large = SimdVector::from_vector(init_large.clone());

        assert_eq!(init_working.len(), matrix_working.unwrap().height());
        assert_eq!(init_large.len(), vec_large.len());
    }

    #[test]
    fn check_matrix_ops() {
        let init_working = vec![
            vec![0.44, 5.55, 77.4, 9.3],
            vec![2.3, 77.3, 0., 4.5],
            vec![88.9, 9.0, 32.1, 0.9],
            vec![2.2, 90.0, 32.2, 0.2],
        ];
        let mut matrix = SimdMatrix::from(init_working.clone()).unwrap();
        assert!(matrix.is_square());
        assert!(matrix.guarantee());
    }

    #[test]
    fn check_simd_get_fns() {
        let init = vec![5.23, 3.2, 0.44, 8.9, 9.0, 5.5, 2.0, 5.0];
        let vector = SimdVector::from_vector(init.clone());
        assert_eq!(init.get(0).as_deref(), vector.get(0).as_ref());
        println!("0 passed");
        assert_eq!(init.get(1).as_deref(), vector.get(1).as_ref());
        println!("1 passed");
        assert_eq!(init.get(2).as_deref(), vector.get(2).as_ref());
        println!("2 passed");
        assert_eq!(init.get(3).as_deref(), vector.get(3).as_ref());
        println!("3 passed");
        assert_eq!(init.get(4).as_deref(), vector.get(4).as_ref());
        println!("4 passed");
        assert_eq!(init.get(5).as_deref(), vector.get(5).as_ref());
        println!("5 passed");
        assert_eq!(init.get(6).as_deref(), vector.get(6).as_ref());
        println!("6 passed");
        assert_eq!(init.get(7).as_deref(), vector.get(7).as_ref());
        println!("7 passed");

        let mut init_large: Vec<f32> = vec![];
        init_large.resize(556, 1.1);
        init_large.fill_with(|| rand::random());
        let vector_large = SimdVector::from_vector(init_large.clone());

        assert_eq!(init_large.get(0).as_deref(), vector_large.get(0).as_ref());
        assert_eq!(init_large.get(45).as_deref(), vector_large.get(45).as_ref());
        assert_eq!(
            init_large.get(337).as_deref(),
            vector_large.get(337).as_ref()
        );
        assert_eq!(
            init_large.get(500).as_deref(),
            vector_large.get(500).as_ref()
        );
        assert_eq!(
            init_large.get(555).as_deref(),
            vector_large.get(555).as_ref()
        );
    }

    #[test]
    fn check_simd_get_thourogh() {
        for i in 0..=62 {
            let mut init: Vec<f32> = vec![];
            init.resize_with(i, || rand::random());
            let vector = SimdVector::from_vector(init.clone());
            for j in 0..=i {
                println!("operating on {:?}, element {}", vector, j);
                assert_eq!(init.get(j).as_deref(), vector.get(j).as_ref());
                println!("vector size {}, element {} passed", i, j);
            }
        }
    }

    #[test]
    fn check_simd_ops() {
        let sum_lhs = SimdVector::from_vector(vec![2.33, 5.66, 9.8]);
        let sum_rhs = SimdVector::from_vector(vec![4.556, 2.3445, 9.67]);
        let prod_lhs = SimdVector::from_vector(vec![2.33, 5.66, 9.8]);
        let prod_rhs = SimdVector::from_vector(vec![4.56, 2.3445, 9.67]);
        let quot_lhs = SimdVector::from_vector(vec![2.33, 5.66, 9.8]);
        let quot_rhs = SimdVector::from_vector(vec![4.556, 2.3445, 9.67]);
        let diff_lhs = SimdVector::from_vector(vec![2.33, 5.66, 9.8]);
        let diff_rhs = SimdVector::from_vector(vec![4.556, 2.3445, 9.67]);

        let sum = sum_rhs + sum_lhs;
        let prod = prod_rhs * prod_lhs;
        let quot = quot_lhs / quot_rhs;
        let diff = diff_lhs - diff_rhs;

        assert_eq!(
            sum.to_vector(),
            vec![2.33 + 4.556, 5.66 + 2.3445, 9.8 + 9.67]
        );
        assert_eq!(
            prod.to_vector(),
            vec![2.33 * 4.56, 5.66 * 2.3445, 9.8 * 9.67]
        );
        assert_eq!(
            quot.to_vector(),
            vec![2.33 / 4.556, 5.66 / 2.3445, 9.8 / 9.67]
        );
        assert_eq!(
            diff.to_vector(),
            vec![2.33 - 4.556, 5.66 - 2.3445, 9.8 - 9.67]
        );
    }

    #[test]
    fn check_matrix_column() {
        let init = vec![
            vec![0.3, 4.3, 5.6],
            vec![0.5, 4.6, 8.9],
            vec![1.2, 22.3, 8.9],
        ];
        let col = SimdVector::from_vector(vec![4.3, 4.6, 22.3]);
        let matrix = SimdMatrix::from(init);
        assert_eq!(
            col.to_vector(),
            matrix.unwrap().column(1).unwrap().to_vector()
        );
    }
}
