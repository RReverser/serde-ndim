use serde::de::{DeserializeSeed, Error, IgnoredAny, IntoDeserializer, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::borrow::BorrowMut;
use std::boxed::Box;
use std::vec;
use std::vec::Vec;

macro_rules! forward_visitors {
    ($(fn $method:ident ($arg:ty);)*) => ($(
        fn $method<E: Error>(self, arg: $arg) -> Result<Self::Value, E> {
            self.deserialize_num(arg)
        }
    )*);
}

/// Multi-dimensional shape storage for deserialization.
pub trait Shape: BorrowMut<[usize]> {
    /// Minimum number of dimensions.
    const MIN_DIMS: usize;
    /// Maximum number of dimensions.
    const MAX_DIMS: usize;

    /// Create a new shape with all dimensions set to `0`.
    fn new_zeroed(dims: usize) -> Self;

    /// Get the length of the given dimension (or `None` if the dimension is out of bounds).
    fn dim_len(&self, dims: usize) -> Option<usize> {
        self.borrow().get(dims).copied()
    }

    /// Set the length of the given dimension.
    fn set_dim_len(&mut self, dims: usize, value: usize) {
        // Check that we're replacing a `0` placeholder.
        debug_assert_eq!(self.dim_len(dims), Some(0));
        self.borrow_mut()[dims] = value;
    }
}

impl Shape for Box<[usize]> {
    const MIN_DIMS: usize = 0;
    const MAX_DIMS: usize = usize::MAX;

    fn new_zeroed(dims: usize) -> Self {
        vec![0; dims].into_boxed_slice()
    }
}

impl<const DIMS: usize> Shape for [usize; DIMS] {
    const MIN_DIMS: usize = DIMS;
    const MAX_DIMS: usize = DIMS;

    fn new_zeroed(dims: usize) -> Self {
        debug_assert_eq!(dims, DIMS);
        [0; DIMS]
    }
}

#[cfg(feature = "arrayvec")]
impl<const MAX_DIMS: usize> Shape for arrayvec::ArrayVec<usize, MAX_DIMS> {
    const MIN_DIMS: usize = 0;
    const MAX_DIMS: usize = MAX_DIMS;

    fn new_zeroed(dims: usize) -> Self {
        debug_assert!(dims <= MAX_DIMS);
        let mut shape = Self::new();
        shape.extend(core::iter::repeat(0).take(dims));
        shape
    }
}

#[derive(Debug)]
struct Context<T, S> {
    data: Vec<T>,
    shape: Option<S>,
    current_dim: usize,
}

impl<'de, T: Deserialize<'de>, S: Shape> Context<T, S> {
    fn got_number<E: Error>(&mut self) -> Result<(), E> {
        match &self.shape {
            Some(shape) => {
                if self.current_dim < shape.borrow().len() {
                    // We've seen a sequence at this dims before, but got a number now.
                    return Err(E::invalid_type(
                        serde::de::Unexpected::Other("a single number"),
                        &"a sequence",
                    ));
                }
            }
            None => {
                // We've seen a sequence at this dims before, but got a number now.
                // Once we've seen a numeric value for the first time, this means we reached the innermost dimension.
                // From now on, start collecting shape info.
                // To start, allocate the dimension lenghs with placeholders.
                if self.current_dim < S::MIN_DIMS {
                    return Err(Error::custom(format_args!(
                        "didn't reach the expected minimum dims {}, got {}",
                        S::MIN_DIMS,
                        self.current_dim,
                    )));
                }
                self.shape = Some(S::new_zeroed(self.current_dim));
            }
        }
        Ok(())
    }

    fn deserialize_num_from<D: Deserializer<'de>>(
        &mut self,
        deserializer: D,
    ) -> Result<(), D::Error> {
        self.got_number()?;
        let value = T::deserialize(deserializer)?;
        self.data.push(value);
        Ok(())
    }

    fn deserialize_num<E: Error>(&mut self, arg: impl IntoDeserializer<'de, E>) -> Result<(), E> {
        self.deserialize_num_from(arg.into_deserializer())
    }
}

impl<'de, T: Deserialize<'de>, S: Shape> Visitor<'de> for &mut Context<T, S> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a sequence or a single number")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        // The code paths for when we know the shape and when we're still doing the first
        // descent are quite different, so we split them into two separate branches.
        if let Some(shape) = &self.shape {
            // This is not the first pass anymore, so we've seen all the dimensions.
            // Check that the current dimension has seen a sequence before and return
            // its expected length.
            let expected_len = shape.dim_len(self.current_dim).ok_or_else(|| {
                Error::invalid_type(serde::de::Unexpected::Seq, &"a single number")
            })?;
            self.current_dim += 1;
            // Consume the expected number of elements.
            for _ in 0..expected_len {
                seq.next_element_seed(&mut *self)?
                    .ok_or_else(|| Error::custom("expected more elements"))?;
            }
            // We've seen all the expected elements in this sequence.
            // Ensure there are no more elements.
            if seq.next_element::<IgnoredAny>()?.is_some() {
                return Err(Error::custom("expected end of sequence"));
            }
            self.current_dim -= 1;
        } else {
            // We're still in the first pass, so we don't know the shape yet.
            debug_assert!(self.shape.is_none());
            self.current_dim += 1;
            if self.current_dim > S::MAX_DIMS {
                return Err(Error::custom(format_args!(
                    "maximum dims of {} exceeded",
                    S::MAX_DIMS
                )));
            }
            // Consume & count all the elements.
            let mut len = 0;
            while seq.next_element_seed(&mut *self)?.is_some() {
                len += 1;
            }
            self.current_dim -= 1;
            // Replace the placeholder `0` with the actual length.
            let shape = self
                .shape
                .as_mut()
                .expect("internal error: shape should be allocated by now");
            shape.set_dim_len(self.current_dim, len);
        }
        Ok(())
    }

    forward_visitors! {
        fn visit_i8(i8);
        fn visit_i16(i16);
        fn visit_i32(i32);
        fn visit_i64(i64);
        fn visit_u8(u8);
        fn visit_u16(u16);
        fn visit_u32(u32);
        fn visit_u64(u64);
        fn visit_f32(f32);
        fn visit_f64(f64);
        fn visit_i128(i128);
        fn visit_u128(u128);
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        // TODO(?): some deserialize implementations don't treat newtypes as transparent.
        // If someone complains, add logic for deserializing as actual wrapped newtype.
        self.deserialize_num_from(deserializer)
    }
}

impl<'de, T: Deserialize<'de>, S: Shape> DeserializeSeed<'de> for &mut Context<T, S> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }
}

/// A trait for types that can be constructed from a shape and a flat data.
pub trait MakeNDim {
    /// The shape of the multi-dimensional array.
    type Shape: Shape;
    /// Array element type.
    type Item;

    /// Construct a multi-dimensional array from a shape and a flat data.
    fn from_shape_and_data(shape: Self::Shape, data: Vec<Self::Item>) -> Self;
}

/// Deserialize a multi-dimensional column-major array from a recursively nested sequence of numbers.
///
/// See [crate-level documentation](../#Deserialization) for more details.
pub fn deserialize<'de, A, D>(deserializer: D) -> Result<A, D::Error>
where
    A: MakeNDim,
    A::Item: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let mut context = Context {
        data: Vec::new(),
        shape: None,
        current_dim: 0,
    };
    deserializer.deserialize_any(&mut context)?;
    Ok(A::from_shape_and_data(context.shape.unwrap(), context.data))
}
