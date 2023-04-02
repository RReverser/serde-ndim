use crate::de::MakeNDim;
use crate::ser::NDim;
use ndarray::{Array, ArrayBase, ArrayD, Data, Dim, Dimension, IntoDimension};

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

impl<'a, S: Data, D: Dimension> NDim<'a> for ArrayBase<S, D> {
    type Shape = &'a [usize];
    type Item = S::Elem;

    fn shape(&'a self) -> Self::Shape {
        ArrayBase::shape(self)
    }

    fn data(&self) -> Option<&[Self::Item]> {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::test_roundtrip;
    use ndarray::{Array3, ArrayD};

    macro_rules! roundtrip {
        ($T:ty, $json:tt) => {
            test_roundtrip::<$T>(serde_json::json!($json))
        };
    }

    #[test]
    fn test_static_array() {
        let array = roundtrip!(
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
        let array = roundtrip!(
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
            roundtrip!(Array3<i32>, [[1, 2, 3], [4, 5, 6]]).unwrap_err()
        );
    }

    #[test]
    fn test_larger_dimension_count() {
        insta::assert_display_snapshot!(
            roundtrip!(Array3<i32>, [[[[1, 2, 3], [4, 5, 6]]]]).unwrap_err()
        );
    }

    #[test]
    fn test_inner_mismatch() {
        insta::assert_display_snapshot!(roundtrip!(
            ArrayD<i32>,
            [[[1, 2, 3, 4], [5, 6, 7, 8]], [9, 10]]
        )
        .unwrap_err());
    }

    #[test]
    fn test_inner_mismatch_during_first_descent() {
        insta::assert_display_snapshot!(roundtrip!(ArrayD<i32>, [[[1, 2, 3, [4]]]]).unwrap_err());
    }

    #[test]
    fn test_invalid_type() {
        insta::assert_display_snapshot!(roundtrip!(ArrayD<i32>, [[[false]]]).unwrap_err());
    }
}
