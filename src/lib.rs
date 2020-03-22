//! DLT (direct linear transform) algorithm for camera calibration
//!
//! This is typically used for calibrating cameras and requires a minimum of 6
//! corresponding pairs of 2D and 3D locations.
//!
//! # Testing
//!
//! ## Unit tests
//!
//! To run the unit tests:
//!
//! ```text
//! cargo test
//! ```
//!
//! ## Test for `no_std`
//!
//! Since the `thumbv7em-none-eabihf` target does not have `std` available, we
//! can build for it to check that our crate does not inadvertently pull in std.
//! The unit tests require std, so cannot be run on a `no_std` platform. The
//! following will fail if a std dependency is present:
//!
//! ```text
//! # install target with: "rustup target add thumbv7em-none-eabihf"
//! cargo build --no-default-features --target thumbv7em-none-eabihf
//! ```
//!
//! # Example
//!
//! ```
//! use dlt::{dlt_corresponding, CorrespondingPoint};
//!
//! let points: Vec<CorrespondingPoint<f64>> = vec![
//!     CorrespondingPoint {
//!         object_point: [-1., -2., -3.],
//!         image_point: [219.700, 39.400],
//!     },
//!     CorrespondingPoint {
//!         object_point: [0., 0., 0.],
//!         image_point: [320.000, 240.000],
//!     },
//!     CorrespondingPoint {
//!         object_point: [1., 2., 3.],
//!         image_point: [420.300, 440.600],
//!     },
//!     CorrespondingPoint {
//!         object_point: [1.1, 2.2, 3.3],
//!         image_point: [430.330, 460.660],
//!     },
//!     CorrespondingPoint {
//!         object_point: [4., 5., 6.],
//!         image_point: [720.600, 741.200],
//!     },
//!     CorrespondingPoint {
//!         object_point: [4.4, 5.5, 6.6],
//!         image_point: [760.660, 791.320],
//!     },
//!     CorrespondingPoint {
//!         object_point: [7., 8., 9.],
//!         image_point: [1020.900, 1041.800],
//!     },
//!     CorrespondingPoint {
//!         object_point: [7.7, 8.8, 9.9],
//!         image_point: [1090.990, 1121.980],
//!     },
//! ];
//!
//! let pmat = dlt_corresponding(&points, 1e-10).unwrap();
//! // could now call `cam_geom::Camera::from_perspective_matrix(&pmat)`
//! ```
//!
//! # See also
//!
//! You may also be interested in:
//!
//! - [`cam-geom`](https://crates.io/crates/cam-geom) - Rust crate with 3D
//!   camera models which can use the calibration data from DLT.
//! - [`dlt-examples`](https://github.com/strawlab/dlt/blob/master/dlt-examples)
//!   - Unpublished crate in the dlt repository which demonstrates usage with
//!   cam-geom library.

#![deny(rust_2018_idioms, unsafe_code, missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

use nalgebra::allocator::Allocator;
use nalgebra::{
    DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimMul, DimProd, DimSub, MatrixMN,
    RealField, RowVectorN, U1, U11, U2, U3, U4,
};

#[allow(non_snake_case)]
fn build_Bc<R, N>(
    world: &MatrixMN<R, N, U3>,
    cam: &MatrixMN<R, N, U2>,
) -> (
    MatrixMN<R, DimProd<N, U2>, U11>,
    MatrixMN<R, DimProd<N, U2>, U1>,
)
where
    R: RealField,
    N: DimMul<U2>,
    DimProd<N, U2>: DimMin<U11>,
    DefaultAllocator: Allocator<R, N, U3>
        + Allocator<R, N, U2>
        + Allocator<R, DimProd<N, U2>, U11>
        + Allocator<R, DimProd<N, U2>, U1>,
{
    let n_pts = world.nrows();

    let n_pts2 = DimProd::<N, U2>::from_usize(n_pts * 2);

    let mut B = MatrixMN::zeros_generic(n_pts2, U11::from_usize(11));
    let mut c = MatrixMN::zeros_generic(n_pts2, U1::from_usize(1));

    let zero = nalgebra::zero();
    let one = nalgebra::one();

    for i in 0..n_pts {
        let X = world[(i, 0)];
        let Y = world[(i, 1)];
        let Z = world[(i, 2)];
        let x = cam[(i, 0)];
        let y = cam[(i, 1)];

        let tmp = RowVectorN::<R, U11>::from_row_slice_generic(
            U1::from_usize(1),
            U11::from_usize(11),
            &[X, Y, Z, one, zero, zero, zero, zero, -x * X, -x * Y, -x * Z],
        );
        B.row_mut(i * 2).copy_from(&tmp);

        let tmp = RowVectorN::<R, U11>::from_row_slice_generic(
            U1::from_usize(1),
            U11::from_usize(11),
            &[zero, zero, zero, zero, X, Y, Z, one, -y * X, -y * Y, -y * Z],
        );
        B.row_mut(i * 2 + 1).copy_from(&tmp);

        c[i * 2] = x;
        c[i * 2 + 1] = y;
    }

    (B, c)
}

/// Direct Linear Transformation (DLT) to find a camera calibration matrix.
///
/// Takes `world`, a matrix of 3D world coordinates, and `cam` a matrix of 2D
/// camera coordinates, which is the image of the world coordinates via the
/// desired projection matrix. Generic over `N`, the number of points, which
/// must be at least `nalgebra::U6`, and can also be `nalgebra::Dynamic`. Also
/// generic over `R`, the data type, which must implement `nalgebra::RealField`.
///
/// You may find it more ergonomic to use the
/// [`dlt_corresponding`](fn.dlt_corresponding.html) function as a convenience
/// wrapper around this function.
///
/// Note that this approach is known to be "unstable" (see Hartley and
/// Zissermann). We should add normalization to fix it. Also, I don't like the
/// notation used by [kwon3d.com](http://www.kwon3d.com/theory/dlt/dlt.html) and
/// prefer that from Carl Olsson as seen
/// [here](http://www.maths.lth.se/matematiklth/personal/calle/datorseende13/notes/forelas3.pdf).
/// That said, kwon3d also suggests how to use the DLT to estimate distortion.
///
/// The DLT method will return intrinsic matrices with skew.
///
/// See
/// [http://www.kwon3d.com/theory/dlt/dlt.html](http://www.kwon3d.com/theory/dlt/dlt.html).
pub fn dlt<R, N>(
    world: &MatrixMN<R, N, U3>,
    cam: &MatrixMN<R, N, U2>,
    epsilon: R,
) -> Result<MatrixMN<R, U3, U4>, &'static str>
where
    // These complicated trait bounds come from:
    // - the matrix `B` that we create has shape (N*2, 11). Thus, everything
    //    with `DimProd<N, U2>, U11>`.
    // - the vector `c` that we create has shape (N*2, 1). Thus, everything with
    //    `DimProd<N, U2>, U1>`.
    // - the SVD operation has its own complicated trait bounds. I copied the
    //    trait bounds required from the SVD source and and then substituted
    //    `DimProd<N, U2>` for `R` (number of rows) and `U11` for `C` (number of
    //    columns).
    R: RealField,
    N: DimMul<U2>,
    DimProd<N, U2>: DimMin<U11>,
    DimMinimum<DimProd<N, U2>, U11>: DimSub<U1>,
    DefaultAllocator: Allocator<R, N, U3>
        + Allocator<R, N, U2>
        + Allocator<R, DimProd<N, U2>, U11>
        + Allocator<R, DimProd<N, U2>, U1>
        + Allocator<R, DimMinimum<DimProd<N, U2>, U11>, U11>
        + Allocator<R, DimProd<N, U2>, DimMinimum<DimProd<N, U2>, U11>>
        + Allocator<R, DimMinimum<DimProd<N, U2>, U11>, U1>
        + Allocator<R, DimDiff<DimMinimum<DimProd<N, U2>, U11>, U1>, U1>,
{
    #[allow(non_snake_case)]
    let (B, c): (
        MatrixMN<R, DimProd<N, U2>, U11>,
        MatrixMN<R, DimProd<N, U2>, U1>,
    ) = build_Bc(&world, &cam);

    // calculate solution with epsilon
    let svd = nalgebra::linalg::SVD::<R, DimProd<N, U2>, U11>::try_new(
        B,
        true,
        true,
        R::default_epsilon(),
        0,
    )
    .ok_or("svd failed")?;
    let solution = svd.solve(&c, epsilon)?;

    let mut pmat_t = MatrixMN::<R, U4, U3>::zeros();
    pmat_t.as_mut_slice()[0..11].copy_from_slice(solution.as_slice());
    pmat_t[(3, 2)] = nalgebra::one();

    let pmat = pmat_t.transpose();

    Ok(pmat)
}

/// A point with a view in image (2D) and world (3D).
///
/// Used by the [`dlt_corresponding`](fn.dlt_corresponding.html) function as a
/// convenience compared to calling the [`dlt`](fn.dlt.html) function directly.
#[derive(Debug)]
pub struct CorrespondingPoint<R: RealField> {
    /// the location of the point in 3D world coordinates
    pub object_point: [R; 3],
    /// the location of the point in 2D pixel coordinates
    pub image_point: [R; 2],
}

#[cfg(feature = "std")]
/// Convenience wrapper around the [`dlt`](fn.dlt.html) function.
///
/// This allows using the [`CorrespondingPoint`](struct.CorrespondingPoint.html)
/// if you find that easier.
///
/// Requires the `std` feature.
pub fn dlt_corresponding<R: RealField>(
    points: &[CorrespondingPoint<R>],
    epsilon: R,
) -> Result<MatrixMN<R, U3, U4>, &'static str> {
    let nrows = nalgebra::Dynamic::from_usize(points.len());

    let world_mat =
        nalgebra::MatrixMN::from_fn_generic(nrows, U3, |i, j| points[i].object_point[j]);

    let image_mat = nalgebra::MatrixMN::from_fn_generic(nrows, U2, |i, j| points[i].image_point[j]);

    // perform the DLT
    dlt(&world_mat, &image_mat, epsilon)
}

#[cfg(test)]
mod tests {
    use nalgebra::{Dynamic, MatrixMN, U2, U3, U4, U8};

    #[test]
    fn test_dlt_corresponding() {
        use crate::CorrespondingPoint;

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

        crate::dlt_corresponding(&points, 1e-10).unwrap();
    }

    #[test]
    fn test_dlt_dynamic() {
        // homogeneous 3D coords
        #[rustfmt::skip]
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
        let dlt_results = crate::dlt(&x3d, &x2d_expected, 1e-10).unwrap();

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
    }

    #[test]
    fn test_dlt_static() {
        // homogeneous 3D coords
        #[rustfmt::skip]
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
        assert!(n_points == 8);

        let x3dh = MatrixMN::<_, U8, U4>::from_row_slice(&x3dh_data);

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
        let x2d_expected = MatrixMN::<_, U8, U2>::from_row_slice(&data);

        // convert homogeneous 3D coords into normal 3D coords
        let x3d = x3dh.fixed_columns::<U3>(0).into_owned();
        // perform DLT
        let dlt_results = crate::dlt(&x3d, &x2d_expected, 1e-10).unwrap();

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
        let x2d_actual = MatrixMN::<_, U8, U2>::from_row_slice(&data);

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
    }
}
