use ndarray::{ArrayD, ShapeBuilder};

impl<T> super::de::MakeNDim for ArrayD<T> {
    type Item = T;

    fn from_shape_and_data(shape: Box<[usize]>, data: Vec<Self::Item>) -> Self {
        ArrayD::from_shape_vec(shape.into_shape(), data)
            .expect("internal mismatch: parsed shape doesn't match the parsed data")
    }
}
