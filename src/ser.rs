use serde::{Serialize, Serializer};
use std::borrow::Borrow;
use std::cell::RefCell;

/// Trait for types that can be serialized as N-dimensional arrays.
///
/// This needs to be implemented on references to arrays, not on arrays themselves,
pub trait NDim {
    /// Shape of the multi-dimensional array (either borrowed or owned).
    type Shape: Borrow<[usize]>;
    /// Iterator over array elements in the column-major order.
    type IterColumnMajor: Iterator;

    /// Get the shape of the multi-dimensional array.
    fn shape(self) -> Self::Shape;
    /// Iterate over array elements in the column-major order.
    fn iter_column_major(self) -> Self::IterColumnMajor;
}

struct SerializeWithShape<'ndim, 'iter, I> {
    count: usize,
    shape_rest: &'ndim [usize],
    iter: &'iter RefCell<I>,
}

impl<'ndim, 'iter, I: Iterator> Serialize for SerializeWithShape<'ndim, 'iter, I>
where
    I::Item: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.shape_rest.split_first() {
            None => serializer.collect_seq((&mut *self.iter.borrow_mut()).take(self.count)),
            Some((&next_count, next_shape_rest)) => {
                serializer.collect_seq((0..self.count).map(|_| SerializeWithShape {
                    count: next_count,
                    shape_rest: next_shape_rest,
                    iter: self.iter,
                }))
            }
        }
    }
}

/// Serialize a multi-dimensional array as a recursively nested sequence of numbers.
///
/// The array must be contiguous and in column-major layout.
pub fn serialize<'a, A, S: Serializer>(array: &'a A, serializer: S) -> Result<S::Ok, S::Error>
where
    &'a A: NDim,
    <<&'a A as NDim>::IterColumnMajor as Iterator>::Item: Serialize,
{
    let shape = array.shape();
    let (&count, shape_rest) = shape
        .borrow()
        .split_first()
        .ok_or_else(|| serde::ser::Error::custom("array must be at least 1-dimensional"))?;
    let iter = RefCell::new(array.iter_column_major());

    SerializeWithShape {
        count,
        shape_rest,
        iter: &iter,
    }
    .serialize(serializer)
}
