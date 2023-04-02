# serde-ndim

[![Crates.io](https://img.shields.io/crates/v/serde-ndim.svg)](https://crates.io/crates/serde-ndim)
[![Documentation](https://docs.rs/serde-ndim/badge.svg)](https://docs.rs/serde-ndim)

## Overview

This crate provides a way to serialize and deserialize arrays of arbitrary dimensionality from self-described formats such as JSON where no out-of-band data is provided about the shape of the resulting array.

This is useful for some data sources (e.g. in astronomical applications), but not the format supported by the built-in Serde integration of popular crates like `ndarray` or `nalgebra`.

Consider input like the following:

```json
[
    [
        [1, 2, 3, 4],
        [4, 5, 6, 7]
    ],
    [
        [7, 8, 9, 10],
        [10, 11, 12, 13]
    ],
    [
        [13, 14, 15, 16],
        [16, 17, 18, 19]
    ]
]
```

This should deserialize into a 3-dimensional array of shape `[3, 2, 4]`. This crate provides `serialize` and `deserialize` functions that can be used via `#[serde(with = "serde_ndim")]` that do just that.

## Deserialization

The tricky bit is that deserialization is built to learn and ensure internal consistency while reading the data:

1. During the first descent, it waits until it reaches a leaf number (`1`) to determine number of dimensions from recursion depth (`3` in example above).
2. It unwinds from the number one step up and reads the sequence `[1, 2, 3, 4]`, learning its length (`4`). Now it remembers the expected shape as `[unknown, unknown, 4]` - it hasn't seen the lengths of the upper dimensions, but at least it knows there are `3` dimensions and the last one has length `4`.
3. It unwinds a step up, recurses into the next sequence, and reads `[4, 5, 6, 7]`. This time it knows it's not the first descent to this dimension, so instead of learning it, it validates the new length against the stored one (`4 == 4`, all good).
4. It reached the end of this sequence of sequences, so now it knows and stores the expected shape as `[unknown, 2, 4]`.
5. By repeating the process, it eventually learns and validates the shape of the whole array as `[3, 2, 4]`.
6. All this time it was collecting raw numbers into a flat `Vec<_>` traditionally as an optimised storage of multidimensional arrays. Now it just needs to call a function that constructs a multidimensional array from the shape and flat data.

**Note**: The resulting array will be in the standard column-major layout.

Constructors for [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html) and [`nalgebra::DMatrix`](https://docs.rs/nalgebra/latest/nalgebra/base/type.DMatrix.html) are provided out of the box under the `ndarray` and `nalgebra` features respectively, so you can use them like this:

```rust
struct MyStruct {
    #[serde(with = "serde_ndim")]
    ndarray: ndarray::ArrayD<f32>,
    /* ... */
}
```

You can also reuse deserialization for custom types by implementing the [`serde_ndarray::de::MakeNDim`](https://docs.rs/serde-ndim/latest/serde_ndim/de/trait.MakeNDim.html) trait.

## Serialization

Serialization is also provided. Its implementaton is much simpler, so I won't go into details here, feel free to check out the code if you want.

It's also provided for `ndarray::Array` and `nalgebra::DMatrix`, but if you want to serialize custom types, you can do so by implementing the [`serde_ndarray::ser::NDim`](https://docs.rs/serde-ndim/latest/serde_ndim/ser/trait.NDim.html) trait.
