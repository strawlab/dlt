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
use dlt::{dlt_corresponding, CorrespondingPoint};
use cam_geom::{Camera, Points};

let points: Vec<CorrespondingPoint<f64>> = vec![
    CorrespondingPoint {
        object_point: [-1., -2., -3.],
        image_point: [219.700, 39.400],
    },
    CorrespondingPoint {
        object_point: [0., 0., 0.],
        image_point: [320.000, 240.000],
    },
    CorrespondingPoint {
        object_point: [1., 2., 3.],
        image_point: [420.300, 440.600],
    },
    CorrespondingPoint {
        object_point: [1.1, 2.2, 3.3],
        image_point: [430.330, 460.660],
    },
    CorrespondingPoint {
        object_point: [4., 5., 6.],
        image_point: [720.600, 741.200],
    },
    CorrespondingPoint {
        object_point: [4.4, 5.5, 6.6],
        image_point: [760.660, 791.320],
    },
    CorrespondingPoint {
        object_point: [7., 8., 9.],
        image_point: [1020.900, 1041.800],
    },
    CorrespondingPoint {
        object_point: [7.7, 8.8, 9.9],
        image_point: [1090.990, 1121.980],
    },
];

let pmat = dlt_corresponding(&points, 1e-10).unwrap();
let cam = Camera::from_perspective_matrix(&pmat).unwrap();
for orig in points.iter() {
    let world = Points::new(nalgebra::RowVector3::from_row_slice(&orig.object_point));
    let px = cam.world_to_pixel(&world);
    approx::assert_relative_eq!(px.data.as_slice(), &orig.image_point[..], epsilon = 1e-4);
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
