use serde::{Serialize, Serializer};
use std::borrow::Borrow;

pub trait NDim {
    type Shape: Borrow<[usize]>;
    type Item;

    fn shape(&self) -> Self::Shape;
    fn data(&self) -> &[Self::Item];
}

struct SerializeWithShape<'ndim, 'data, T> {
    count: usize,
    shape_rest: &'ndim [usize],
    data: &'data [T],
}

impl<'ndim, 'data, T: Serialize> Serialize for SerializeWithShape<'ndim, 'data, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.shape_rest.split_first() {
            None => self.data.serialize(serializer),
            Some((&next_count, next_shape_rest)) => {
                serializer.collect_seq(self.data.chunks_exact(self.data.len() / self.count).map(
                    |chunk| SerializeWithShape {
                        count: next_count,
                        shape_rest: next_shape_rest,
                        data: chunk,
                    },
                ))
            }
        }
    }
}

pub fn serialize<A: NDim, S: Serializer>(array: &A, serializer: S) -> Result<S::Ok, S::Error>
where
    A::Item: Serialize,
{
    match array.shape().borrow().split_first() {
        None => Err(serde::ser::Error::custom(
            "array must be at least 1-dimensional",
        )),
        Some((&count, shape_rest)) => SerializeWithShape {
            count,
            shape_rest,
            data: array.data(),
        }
        .serialize(serializer),
    }
}
