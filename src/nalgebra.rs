use crate::de::MakeNDim;
use crate::ser::NDim;
use nalgebra::{DMatrix, Dim, Dyn, IsContiguous, Matrix, RawStorage, VecStorage};

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

impl<T, R: Dim, C: Dim, S: RawStorage<T, R, C> + IsContiguous> NDim for Matrix<T, R, C, S> {
    type Shape = [usize; 2];
    type Item = T;

    fn shape(&self) -> Self::Shape {
        let (rows, cols) = self.shape();
        [cols, rows]
    }

    fn data(&self) -> &[Self::Item] {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use format_serde_error::SerdeError;
    use nalgebra::DMatrix;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(transparent)]
    struct WrapMatrix(#[serde(with = "crate")] DMatrix<i32>);

    macro_rules! roundtrip {
        ($json:tt) => {{
            let json = json!($json);
            let json_string = serde_json::to_string_pretty(&json).unwrap();
            // using `from_str` for better errors with locations
            match serde_json::from_str::<WrapMatrix>(&json_string) {
                Ok(wrap) => {
                    let new_json = serde_json::to_value(&wrap).unwrap();
                    assert_eq!(
                        json,
                        new_json,
                        "Roundtrip mismatch\nOriginal input: {json:#}\nAfter roundtrip: {new_json:#}"
                    );
                    Ok(wrap.0)
                }
                Err(err) => Err(SerdeError::new(json_string, err)),
            }
        }};
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
