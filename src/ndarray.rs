use crate::MakeNDim;
use ndarray::{Array, ArrayD, Dim, IntoDimension};

impl<T> MakeNDim for ArrayD<T> {
    type Shape = Box<[usize]>;
    type Item = T;

    fn from_shape_and_data(shape: Box<[usize]>, data: Vec<Self::Item>) -> Self {
        Self::from_shape_vec(shape.into_dimension(), data)
            .expect("internal mismatch: parsed shape doesn't match the parsed data")
    }
}

// Unfortunately, ndarray doesn't use const-generics yet, but a plain old macro,
// so we have to use one as well.
macro_rules! impl_ndim {
    ($($N:literal)*) => ($(
        impl<T> MakeNDim for Array<T, Dim<[usize; $N]>> {
            type Shape = [usize; $N];
            type Item = T;

            fn from_shape_and_data(shape: [usize; $N], data: Vec<Self::Item>) -> Self {
                Self::from_shape_vec(shape.into_dimension(), data)
                    .expect("internal mismatch: parsed shape doesn't match the parsed data")
            }
        }
    )*);
}

impl_ndim!(1 2 3 4 5 6);

#[cfg(test)]
mod tests {
    use format_serde_error::SerdeError;
    use ndarray::Array3;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    #[serde(transparent)]
    struct WrapArray3(#[serde(with = "crate")] Array3<i32>);

    macro_rules! deserialize_json {
        ($json:expr) => {{
            let json = stringify!($json);
            match serde_json::from_str::<WrapArray3>(json) {
                Ok(WrapArray3(array)) => Ok(array),
                Err(err) => Err(SerdeError::new(json.to_owned(), err)),
            }
        }};
    }

    #[test]
    fn test_sample() {
        insta::assert_display_snapshot!(deserialize_json!([
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [[9, 10, 11, 12], [13, 14, 15, 16]],
            [[17, 18, 19, 20], [21, 22, 23, 24]]
        ])
        .unwrap());
    }

    #[test]
    fn test_invalid_dimension_count() {
        insta::assert_display_snapshot!(deserialize_json!([[1, 2, 3], [4, 5, 6]]).unwrap_err());
    }

    #[test]
    fn test_inner_mismatch() {
        insta::assert_display_snapshot!(
            deserialize_json!([[[1, 2, 3, 4], [5, 6, 7, 8]], [9, 10]]).unwrap_err()
        );
    }

    #[test]
    fn test_inner_mismatch_during_first_descent() {
        insta::assert_display_snapshot!(deserialize_json!([[[1, 2, 3, [4]]]]).unwrap_err());
    }

    #[test]
    fn test_invalid_type() {
        insta::assert_display_snapshot!(deserialize_json!([[[false]]]).unwrap_err());
    }
}
