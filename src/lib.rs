mod de;
mod ser;

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
        serialize = "A: for<'a> crate::ser::NDim<'a>, for<'a> <A as crate::ser::NDim<'a>>::Item: Serialize",
        deserialize = "A: crate::de::MakeNDim, A::Item: Deserialize<'de> + Debug"
    ))]
    struct TestWrapper<A>(#[serde(with = "crate")] A);

    pub(crate) fn test_roundtrip<A: for<'a> crate::ser::NDim<'a> + crate::de::MakeNDim>(
        json: serde_json::Value,
    ) -> Result<A, format_serde_error::SerdeError>
    where
        <A as crate::de::MakeNDim>::Item: DeserializeOwned + Debug,
        for<'a> <A as crate::ser::NDim<'a>>::Item: Serialize,
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
