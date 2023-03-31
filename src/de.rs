use serde::de::{DeserializeSeed, Error, IgnoredAny, IntoDeserializer, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::borrow::BorrowMut;
use std::fmt::Debug;

macro_rules! forward_visitors {
    ($(fn $method:ident ($arg:ty);)*) => ($(
        fn $method<E: Error>(self, arg: $arg) -> Result<Self::Value, E> {
            self.deserialize_num(arg)
        }
    )*);
}

pub trait Shape: BorrowMut<[usize]> + Debug {
    const MAX_DEPTH: Option<usize>;

    fn new_zeroed(depth: usize) -> Self;

    fn shape_at(&self, depth: usize) -> Option<usize> {
        self.borrow().get(depth).copied()
    }

    fn set_shape_at(&mut self, depth: usize, value: usize) {
        // Check that we're replacing a `0` placeholder.
        debug_assert_eq!(self.shape_at(depth), Some(0));
        self.borrow_mut()[depth] = value;
    }
}

impl Shape for Box<[usize]> {
    const MAX_DEPTH: Option<usize> = None;

    fn new_zeroed(depth: usize) -> Self {
        vec![0; depth].into_boxed_slice()
    }
}

impl<const DIMS: usize> Shape for [usize; DIMS] {
    const MAX_DEPTH: Option<usize> = Some(DIMS);

    fn new_zeroed(depth: usize) -> Self {
        assert_eq!(depth, DIMS);
        [0; DIMS]
    }
}

#[derive(Debug)]
struct Context<T, S> {
    data: Vec<T>,
    shape: Option<S>,
    current_depth: usize,
}

impl<'de, T: Debug + Deserialize<'de>, S: Shape> Context<T, S> {
    fn got_number<E: Error>(&mut self) -> Result<(), E> {
        match &self.shape {
            Some(shape) => {
                if self.current_depth < shape.borrow().len() {
                    // We've seen a sequence at this depth before, but got a number now.
                    return Err(E::invalid_type(
                        serde::de::Unexpected::Other("a single number"),
                        &"a sequence",
                    ));
                }
            }
            None => {
                // We've seen a sequence at this depth before, but got a number now.
                // Once we've seen a numeric value for the first time, this means we reached the innermost dimension.
                // From now on, start collecting shape info.
                // To start, allocate the dimension lenghs with placeholders.
                self.shape = Some(S::new_zeroed(self.current_depth));
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

impl<'de, T: Deserialize<'de> + Debug, S: Shape> Visitor<'de> for &mut Context<T, S> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a sequence or a single element")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        // The code paths for when we know the shape and when we're still doing the first
        // descent are quite different, so we split them into two separate branches.
        if let Some(shape) = &self.shape {
            // This is not the first pass anymore, so we've seen all the dimensions.
            // Check that the current dimension has seen a sequence before and return
            // its expected length.
            let expected_len = shape.shape_at(self.current_depth).ok_or_else(|| {
                Error::invalid_type(serde::de::Unexpected::Seq, &"a single number")
            })?;
            self.current_depth += 1;
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
            self.current_depth -= 1;
        } else {
            // We're still in the first pass, so we don't know the shape yet.
            debug_assert!(self.shape.is_none());
            self.current_depth += 1;
            if let Some(max_depth) = S::MAX_DEPTH {
                if self.current_depth > max_depth {
                    return Err(Error::custom(format!(
                        "maximum depth of {} exceeded",
                        max_depth
                    )));
                }
            }
            // Consume & count all the elements.
            let mut len = 0;
            while seq.next_element_seed(&mut *self)?.is_some() {
                len += 1;
            }
            self.current_depth -= 1;
            // Replace the placeholder `0` with the actual length.
            let shape = self
                .shape
                .as_mut()
                .expect("internal error: shape should be allocated by now");
            shape.set_shape_at(self.current_depth, len);
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

impl<'de, T: Deserialize<'de> + Debug, S: Shape> DeserializeSeed<'de> for &mut Context<T, S> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }
}

pub trait MakeNDim {
    type Shape: Shape;
    type Item;

    fn from_shape_and_data(shape: Self::Shape, data: Vec<Self::Item>) -> Self;
}

pub fn deserialize<'de, A, D>(deserializer: D) -> Result<A, D::Error>
where
    A: MakeNDim,
    A::Item: Deserialize<'de> + Debug,
    D: Deserializer<'de>,
{
    let mut context = Context {
        data: Vec::new(),
        shape: None,
        current_depth: 0,
    };
    deserializer.deserialize_any(&mut context)?;
    Ok(A::from_shape_and_data(context.shape.unwrap(), context.data))
}
