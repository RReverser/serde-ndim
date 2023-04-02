use crate::de::MakeNDim;
use ndarray::{Array, ArrayD, Dim, Dimension, IntoDimension};

impl<T, const N: usize> MakeNDim for Array<T, Dim<[usize; N]>>
where
    // ndarray doesn't use const-generics for Dimension yet so we need
    // some extra bounds to filter out unsupported [usize; N] combinations.
    Dim<[usize; N]>: Dimension,
    [usize; N]: IntoDimension<Dim = Dim<[usize; N]>>,
{
    type Shape = [usize; N];
    type Item = T;

    fn from_shape_and_data(shape: Self::Shape, data: Vec<Self::Item>) -> Self {
        Self::from_shape_vec(shape, data)
            .expect("internal mismatch: parsed shape doesn't match the parsed data")
    }
}

impl<T> MakeNDim for ArrayD<T> {
    type Shape = Box<[usize]>;
    type Item = T;

    fn from_shape_and_data(shape: Self::Shape, data: Vec<Self::Item>) -> Self {
        Self::from_shape_vec(
            // without into_vec() it will go via &[usize] cloning path
            shape.into_vec(),
            data,
        )
        .expect("internal mismatch: parsed shape doesn't match the parsed data")
    }
}

#[cfg(test)]
mod tests {
    use crate::de::MakeNDim;
    use format_serde_error::SerdeError;
    use ndarray::{Array3, ArrayD};
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    #[serde(transparent)]
    #[serde(bound(deserialize = "A: MakeNDim, A::Item: Deserialize<'de> + std::fmt::Debug"))]
    struct WrapArray<A>(#[serde(with = "crate")] A);

    macro_rules! deserialize_json {
        ($T:ty, $json:expr) => {{
            let json = stringify!($json);
            match serde_json::from_str::<WrapArray<$T>>(json) {
                Ok(WrapArray(array)) => Ok(array),
                Err(err) => Err(SerdeError::new(json.to_owned(), err)),
            }
        }};
    }

    #[test]
    fn test_static_array() {
        let array = deserialize_json!(
            Array3<i32>,
            [
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [[9, 10, 11, 12], [13, 14, 15, 16]],
                [[17, 18, 19, 20], [21, 22, 23, 24]]
            ]
        )
        .unwrap();
        assert_eq!(array.shape(), &[3, 2, 4]);
        assert!(array.is_standard_layout());
        insta::assert_display_snapshot!(array);
    }

    #[test]
    fn test_dyn_array() {
        let array = deserialize_json!(
            ArrayD<i32>,
            [
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [[9, 10, 11, 12], [13, 14, 15, 16]],
                [[17, 18, 19, 20], [21, 22, 23, 24]]
            ]
        )
        .unwrap();
        assert_eq!(array.shape(), &[3, 2, 4]);
        assert!(array.is_standard_layout());
        insta::assert_display_snapshot!(array);
    }

    #[test]
    fn test_smaller_dimension_count() {
        insta::assert_display_snapshot!(
            deserialize_json!(Array3<i32>, [[1, 2, 3], [4, 5, 6]]).unwrap_err()
        );
    }

    #[test]
    fn test_larger_dimension_count() {
        insta::assert_display_snapshot!(
            deserialize_json!(Array3<i32>, [[[[1, 2, 3], [4, 5, 6]]]]).unwrap_err()
        );
    }

    #[test]
    fn test_inner_mismatch() {
        insta::assert_display_snapshot!(deserialize_json!(
            ArrayD<i32>,
            [[[1, 2, 3, 4], [5, 6, 7, 8]], [9, 10]]
        )
        .unwrap_err());
    }

    #[test]
    fn test_inner_mismatch_during_first_descent() {
        insta::assert_display_snapshot!(
            deserialize_json!(ArrayD<i32>, [[[1, 2, 3, [4]]]]).unwrap_err()
        );
    }

    #[test]
    fn test_invalid_type() {
        insta::assert_display_snapshot!(deserialize_json!(ArrayD<i32>, [[[false]]]).unwrap_err());
    }
}
