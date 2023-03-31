use serde::de::{DeserializeSeed, Error, IgnoredAny, IntoDeserializer, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

macro_rules! forward_visitors {
    ($(fn $method:ident ($arg:ty);)*) => ($(
        fn $method<E: Error>(self, arg: $arg) -> Result<Self::Value, E> {
            self.deserialize_num(arg)
        }
    )*);
}

struct Context<T> {
    data: Vec<T>,
    shape: Option<Box<[usize]>>,
    current_depth: usize,
}

impl<'de, T: Deserialize<'de>> Context<T> {
    fn got_number<E: Error>(&mut self) -> Result<(), E> {
        match &self.shape {
            Some(shape) => {
                if self.current_depth < shape.len() {
                    // We've seen a sequence at this depth before, but got a number now.
                    return Err(E::invalid_type(
                        serde::de::Unexpected::Other("a single number"),
                        &"a sequence",
                    ));
                }
            }
            None => {
                // Once we've seen a numeric value for the first time, this means we reached the innermost dimension.
                // From now on, start collecting shape info.
                // To start, allocate the dimension lenghs with placeholders.
                self.shape = Some(vec![0; self.current_depth + 1].into_boxed_slice());
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

impl<'de, T: Deserialize<'de>> Visitor<'de> for &mut Context<T> {
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
            let expected_len = shape.get(self.current_depth).copied().ok_or_else(|| {
                Error::invalid_type(serde::de::Unexpected::Seq, &"a single number")
            })?;
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
            debug_assert_eq!(self.shape, None);
            self.current_depth += 1;
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
            shape[self.current_depth] = len;
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

impl<'de, T: Deserialize<'de>> DeserializeSeed<'de> for &mut Context<T> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }
}

pub trait MakeNDim {
    type Item;

    fn from_shape_and_data(shape: Box<[usize]>, data: Vec<Self::Item>) -> Self;
}

pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: MakeNDim,
    T::Item: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let mut context = Context {
        data: Vec::new(),
        shape: None,
        current_depth: 0,
    };
    deserializer.deserialize_any(&mut context)?;
    Ok(T::from_shape_and_data(context.shape.unwrap(), context.data))
}
