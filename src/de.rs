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
    shape: Vec<usize>,
    current_depth: usize,
    first_pass: bool,
}

impl<'de, T: Deserialize<'de>> Context<T> {
    fn got_number<E: Error>(&mut self) -> Result<(), E> {
        if self.shape.get(self.current_depth).is_some() {
            // We've seen a sequence at this depth before, but got a number now.
            return Err(E::invalid_type(
                serde::de::Unexpected::Other("a single number"),
                &"a sequence",
            ));
        }
        // Once we've seen a numeric value for the first time, this means we reached the innermost dimension.
        // Unmark the first pass flag so that from now on we start validating shapes.
        self.first_pass = false;
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
        // The code paths for when we know the shape and when we don't are quite different,
        // so we split them into two separate branches.
        if self.first_pass {
            // We're still in the first pass, so we don't know the shape yet.
            debug_assert_eq!(self.current_depth, self.shape.len());
            // Just push a `0` placeholder so that we could recurse deeper.
            // It will be replaced with an actual length once we finish this
            // sequence for the first time.
            self.shape.push(0);
            self.current_depth += 1;
            // Consume & count all the elements.
            let mut len = 0;
            while seq.next_element_seed(&mut *self)?.is_some() {
                len += 1;
            }
            self.current_depth -= 1;
            // Replace the placeholder with the actual length.
            debug_assert_eq!(self.shape[self.current_depth], 0);
            self.shape[self.current_depth] = len;
        } else {
            // This is not the first pass anymore, so we've seen all the dimensions.
            // Check that the current dimension has seen a sequence before and return
            // its expected length.
            let expected_len = self.shape.get(self.current_depth).copied().ok_or_else(|| {
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

    fn from_shape_and_data(shape: Vec<usize>, data: Vec<Self::Item>) -> Self;
}

pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: MakeNDim,
    T::Item: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let mut context = Context {
        data: Vec::new(),
        shape: Vec::new(),
        current_depth: 0,
        first_pass: true,
    };
    deserializer.deserialize_any(&mut context)?;
    Ok(T::from_shape_and_data(context.shape, context.data))
}
