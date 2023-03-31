mod de;
mod ser;

#[cfg(feature = "nalgebra")]
mod nalgebra;
#[cfg(feature = "ndarray")]
mod ndarray;

pub use de::{deserialize, MakeNDim};
