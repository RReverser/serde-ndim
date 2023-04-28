use crate::de::MakeNDim;
use crate::ser::NDim;
use nalgebra::{DMatrix, Dim, Dyn, IsContiguous, Matrix, RawStorage, Scalar, VecStorage};
use std::vec::Vec;

impl<T> MakeNDim for DMatrix<T> {
    type Shape = [usize; 2];
    type Item = T;

    fn from_shape_and_data(shape: [usize; 2], data: Vec<Self::Item>) -> Self {
        // note the inverted order: nalgebra constructors take rows then cols,
        // but we're parsing in standard column-major order
        let [cols, rows] = shape.map(Dyn);
        Self::from_vec_storage(VecStorage::new(rows, cols, data))
    }
}

impl<'a, T: Scalar + Copy, R: Dim, C: Dim, S: RawStorage<T, R, C> + IsContiguous> NDim
    for &'a Matrix<T, R, C, S>
{
    type Shape = [usize; 2];
    type IterColumnMajor = std::iter::Copied<<Self as IntoIterator>::IntoIter>;

    fn shape(self) -> Self::Shape {
        let (rows, cols) = self.shape();
        [cols, rows]
    }

    fn iter_column_major(self) -> Self::IterColumnMajor {
        self.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::test_roundtrip;
    use nalgebra::DMatrix;
    use serde_json::json;

    macro_rules! roundtrip {
        ($json:tt) => {
            test_roundtrip::<DMatrix<i32>>(json!($json))
        };
    }

    #[test]
    fn test_matrix() {
        let matrix = roundtrip!([[1, 2, 3, 4], [5, 6, 7, 8]]).unwrap();
        // not using `.shape()` to explicitly check that it's column-major
        assert_eq!((matrix.ncols(), matrix.nrows()), (2, 4));
        insta::assert_display_snapshot!(matrix);
    }

    #[test]
    fn test_smaller_dimension_count() {
        insta::assert_display_snapshot!(roundtrip!([1, 2, 3, 4]).unwrap_err());
    }

    #[test]
    fn test_larger_dimension_count() {
        insta::assert_display_snapshot!(roundtrip!([
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [[9, 10, 11, 12], [13, 14, 15, 16]],
            [[17, 18, 19, 20], [21, 22, 23, 24]]
        ])
        .unwrap_err());
    }

    #[test]
    fn test_inner_mismatch() {
        insta::assert_display_snapshot!(roundtrip!([[1, 2, 3, 4], [5, 6, 8]]).unwrap_err());
    }

    #[test]
    fn test_inner_mismatch_during_first_descent() {
        insta::assert_display_snapshot!(roundtrip!([[1, [2], 3, 4], [5, 6, 7, 8]]).unwrap_err());
    }

    #[test]
    fn test_invalid_type() {
        insta::assert_display_snapshot!(roundtrip!([[false]]).unwrap_err());
    }
}
