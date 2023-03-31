use crate::MakeNDim;
use ndarray::{Array, ArrayD, Dim, IntoDimension};

impl<T> MakeNDim for ArrayD<T> {
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
        impl<T> MakeNDim<[usize; $N]> for Array<T, Dim<[usize; $N]>> {
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
    use ndarray::Array3;
    use serde::Deserialize;

    macro_rules! deserialize_json {
        ($json:literal) => {
            serde_json::from_str(concat!(/* reset column */ "\n", $json)).unwrap_or_else(|err| {
                // Offset error position to the original location of JSON literal so it's clickable in terminal.
                panic!(
                    "{file}:{line}:{column}: {err}",
                    file = file!(),
                    line = line!() + err.line() as u32 + /* offset we added */ 2,
                    column = err.column()
                )
            })
        };
    }

    #[test]
    fn test_sample() {
        #[derive(Deserialize)]
        #[serde(transparent)]
        struct WrapArray3(#[serde(with = "crate")] Array3<i32>);

        let WrapArray3(array) = deserialize_json!(
            r#"[
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8]
                ],
                [
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]
                ],
                [
                    [17, 18, 19, 20],
                    [21, 22, 23, 24]
                ]
            ]"#
        );

        insta::assert_display_snapshot!(array);
    }
}
