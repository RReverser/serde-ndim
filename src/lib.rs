#![cfg_attr(feature = "docs_rs", feature(doc_auto_cfg))]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc as std;

/// Deserialization module.
pub mod de;
/// Serialization module.
pub mod ser;

#[cfg(feature = "nalgebra")]
mod nalgebra;
#[cfg(feature = "ndarray")]
mod ndarray;

pub use de::deserialize;
pub use ser::serialize;

#[cfg(test)]
mod tests {
    use serde::de::DeserializeOwned;
    use serde::{Deserialize, Serialize};
    use std::fmt::Debug;

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(transparent)]
    #[serde(bound(
        serialize = "for<'a> &'a A: crate::ser::NDim, for<'a> <<&'a A as crate::ser::NDim>::IterColumnMajor as Iterator>::Item: Serialize",
        deserialize = "A: crate::de::MakeNDim, A::Item: Deserialize<'de>"
    ))]
    pub(crate) struct TestWrapper<A>(#[serde(with = "crate")] pub(crate) A);

    pub(crate) fn test_roundtrip<A: crate::de::MakeNDim>(
        json: serde_json::Value,
    ) -> Result<A, format_serde_error::SerdeError>
    where
        for<'a> &'a A: crate::ser::NDim,
        for<'a> <<&'a A as crate::ser::NDim>::IterColumnMajor as Iterator>::Item: Serialize,
        <A as crate::de::MakeNDim>::Item: DeserializeOwned,
    {
        let json_string = serde_json::to_string_pretty(&json).unwrap();
        // using `from_str` for better errors with locations
        match serde_json::from_str::<TestWrapper<A>>(&json_string) {
            Ok(wrapper) => {
                let new_json = serde_json::to_value(&wrapper).unwrap();
                assert_eq!(
                    json, new_json,
                    "Roundtrip mismatch\nOriginal input: {json:#}\nAfter roundtrip: {new_json:#}"
                );
                Ok(wrapper.0)
            }
            Err(err) => Err(format_serde_error::SerdeError::new(json_string, err)),
        }
    }
}
