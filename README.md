# Crate `dlt` for the [Rust language](https://www.rust-lang.org/)

<!-- Note: README.md is generated automatically by `cargo readme` -->

[![Crates.io](https://img.shields.io/crates/v/dlt.svg)](https://crates.io/crates/dlt)
[![Documentation](https://docs.rs/dlt/badge.svg)](https://docs.rs/dlt/)
[![Crate License](https://img.shields.io/crates/l/dlt.svg)](https://crates.io/crates/dlt)
[![Dependency status](https://deps.rs/repo/github/strawlab/dlt/status.svg)](https://deps.rs/repo/github/strawlab/dlt)

DLT (direct linear transform) algorithm for camera calibration

This is typically used for calibrating cameras and requires a minimum of 6
corresponding pairs of 2D and 3D locations.

## Testing

### Unit tests

To run the unit tests:

```
cargo test
```

### Test for `no_std`

Since the `thumbv7em-none-eabihf` target does not have `std` available, we
can build for it to check that our crate does not inadvertently pull in std.
The unit tests require std, so cannot be run on a `no_std` platform. The
following will fail if a std dependency is present:

```
# install target with: "rustup target add thumbv7em-none-eabihf"
cargo build --no-default-features --target thumbv7em-none-eabihf
```

**Currently, this crate does not build without std, but this is a bug that
will be fixed.**

## Example

```rust
use nalgebra::{Dynamic, MatrixMN, U2, U3, U4, U8};

// homogeneous 3D coords
let x3dh_data: Vec<f64> = vec![
    -1., -2., -3., 1.0,
    0., 0., 0., 1.0,
    1., 2., 3., 1.0,
    1.1, 2.2, 3.3, 1.0,
    4., 5., 6., 1.0,
    4.4, 5.5, 6.6, 1.0,
    7., 8., 9., 1.0,
    7.7, 8.8, 9.9, 1.0,
    ];

let n_points = x3dh_data.len() / 4;

let x3dh = MatrixMN::<_, Dynamic, U4>::from_row_slice(&x3dh_data);

// example camera calibration matrix
#[rustfmt::skip]
let pmat_data: Vec<f64> = vec![
    100.0,  0.0, 0.1, 320.0,
    0.0, 100.0, 0.2, 240.0,
    0.0,  0.0, 0.0,   1.0,
    ];
let pmat = MatrixMN::<_, U3, U4>::from_row_slice(&pmat_data);

// compute 2d coordinates of camera projection
let x2dh = pmat * x3dh.transpose();

// convert 2D homogeneous coords into normal 2D coords
let mut data = Vec::with_capacity(2 * n_points);
for i in 0..n_points {
    let r = x2dh[(0, i)];
    let s = x2dh[(1, i)];
    let t = x2dh[(2, i)];
    data.push(r / t);
    data.push(s / t);
}
let x2d_expected = MatrixMN::<_, Dynamic, U2>::from_row_slice(&data);

// convert homogeneous 3D coords into normal 3D coords
let x3d = x3dh.fixed_columns::<U3>(0).into_owned();
// perform DLT
let dlt_results = dlt::dlt(&x3d, &x2d_expected, 1e-10).unwrap();

// compute 2d coordinates of camera projection with DLT-found matrix
let x2dh2 = dlt_results * x3dh.transpose();

// convert 2D homogeneous coords into normal 2D coords
let mut data = Vec::with_capacity(2 * n_points);
for i in 0..n_points {
    let r = x2dh2[(0, i)];
    let s = x2dh2[(1, i)];
    let t = x2dh2[(2, i)];
    data.push(r / t);
    data.push(s / t);
}
let x2d_actual = MatrixMN::<_, Dynamic, U2>::from_row_slice(&data);

assert_eq!(x2d_expected.nrows(), x2d_actual.nrows());
assert_eq!(x2d_expected.ncols(), x2d_actual.ncols());
for i in 0..x2d_expected.nrows() {
    for j in 0..x2d_expected.ncols() {
        approx::assert_relative_eq!(
            x2d_expected[(i, j)],
            x2d_actual[(i, j)],
            epsilon = 1e-10
        );
    }
}
```

## See also

You may also be interested in:

- [`cam-geom`](https://crates.io/crates/cam-geom) - Rust crate with 3D
  camera models which can use the calibration data from DLT.

## Regenerate `README.md`

The `README.md` file can be regenerated with:

```text
cargo readme > README.md
```

## Code of conduct

Anyone who interacts with this software in any space, including but not limited
to this GitHub repository, must follow our [code of
conduct](code_of_conduct.md).

## License

Licensed under either of these:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   https://opensource.org/licenses/MIT)
