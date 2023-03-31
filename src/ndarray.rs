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
