use serde::{Serialize, Serializer};
use std::borrow::Borrow;

pub trait NDim<'a> {
    type Shape: 'a + Borrow<[usize]>;
    type Item;

    fn shape(&'a self) -> Self::Shape;
    fn data(&self) -> Option<&[Self::Item]>;
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

pub fn serialize<'a, A: NDim<'a>, S: Serializer>(
    array: &'a A,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    A::Item: Serialize,
{
    let shape = array.shape();
    let (&count, shape_rest) = shape
        .borrow()
        .split_first()
        .ok_or_else(|| serde::ser::Error::custom("array must be at least 1-dimensional"))?;
    let data = array.data().ok_or_else(|| {
        serde::ser::Error::custom("array must be contiguous and in column-major layout")
    })?;
    SerializeWithShape {
        count,
        shape_rest,
        data,
    }
    .serialize(serializer)
}
