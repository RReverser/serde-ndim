use crate::MakeNDim;
use nalgebra::{DMatrix, Dyn, VecStorage};

impl<T> MakeNDim for DMatrix<T> {
    type Shape = [usize; 2];
    type Item = T;

    fn from_shape_and_data(shape: [usize; 2], data: Vec<Self::Item>) -> Self {
        let [rows, columns] = shape.map(Dyn);
        Self::from_vec_storage(VecStorage::new(rows, columns, data))
    }
}
