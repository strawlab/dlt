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
//! **Currently, this crate does not build without std, but this is a bug that
//! will be fixed.**
//!
//! # Example
//!
//! ```
//! use nalgebra::{Dynamic, MatrixMN, U2, U3, U4, U8};
//!
//! // homogeneous 3D coords
//! let x3dh_data: Vec<f64> = vec![
//!     -1., -2., -3., 1.0,
//!     0., 0., 0., 1.0,
//!     1., 2., 3., 1.0,
//!     1.1, 2.2, 3.3, 1.0,
//!     4., 5., 6., 1.0,
//!     4.4, 5.5, 6.6, 1.0,
//!     7., 8., 9., 1.0,
//!     7.7, 8.8, 9.9, 1.0,
//!     ];
//!
//! let n_points = x3dh_data.len() / 4;
//!
//! let x3dh = MatrixMN::<_, Dynamic, U4>::from_row_slice(&x3dh_data);
//!
//! // example camera calibration matrix
//! #[rustfmt::skip]
//! let pmat_data: Vec<f64> = vec![
//!     100.0,  0.0, 0.1, 320.0,
//!     0.0, 100.0, 0.2, 240.0,
//!     0.0,  0.0, 0.0,   1.0,
//!     ];
//! let pmat = MatrixMN::<_, U3, U4>::from_row_slice(&pmat_data);
//!
//! // compute 2d coordinates of camera projection
//! let x2dh = pmat * x3dh.transpose();
//!
//! // convert 2D homogeneous coords into normal 2D coords
//! let mut data = Vec::with_capacity(2 * n_points);
//! for i in 0..n_points {
//!     let r = x2dh[(0, i)];
//!     let s = x2dh[(1, i)];
//!     let t = x2dh[(2, i)];
//!     data.push(r / t);
//!     data.push(s / t);
//! }
//! let x2d_expected = MatrixMN::<_, Dynamic, U2>::from_row_slice(&data);
//!
//! // convert homogeneous 3D coords into normal 3D coords
//! let x3d = x3dh.fixed_columns::<U3>(0).into_owned();
//! // perform DLT
//! let dlt_results = dlt::dlt(&x3d, &x2d_expected, 1e-10).unwrap();
//!
//! // compute 2d coordinates of camera projection with DLT-found matrix
//! let x2dh2 = dlt_results * x3dh.transpose();
//!
//! // convert 2D homogeneous coords into normal 2D coords
//! let mut data = Vec::with_capacity(2 * n_points);
//! for i in 0..n_points {
//!     let r = x2dh2[(0, i)];
//!     let s = x2dh2[(1, i)];
//!     let t = x2dh2[(2, i)];
//!     data.push(r / t);
//!     data.push(s / t);
//! }
//! let x2d_actual = MatrixMN::<_, Dynamic, U2>::from_row_slice(&data);
//!
//! assert_eq!(x2d_expected.nrows(), x2d_actual.nrows());
//! assert_eq!(x2d_expected.ncols(), x2d_actual.ncols());
//! for i in 0..x2d_expected.nrows() {
//!     for j in 0..x2d_expected.ncols() {
//!         approx::assert_relative_eq!(
//!             x2d_expected[(i, j)],
//!             x2d_actual[(i, j)],
//!             epsilon = 1e-10
//!         );
//!     }
//! }
//! ```
//!
//! # See also
//!
//! You may also be interested in:
//!
//! - [`cam-geom`](https://crates.io/crates/cam-geom) - Rust crate with 3D
//!   camera models which can use the calibration data from DLT.

#![deny(rust_2018_idioms, unsafe_code, missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate core as std;

use nalgebra::allocator::Allocator;
use nalgebra::{
    DVector, DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimSub, Dynamic, MatrixMN,
    RealField, VectorN, U1, U11, U2, U3, U4,
};

#[allow(non_snake_case)]
fn build_Bc<R, N>(
    world: &MatrixMN<R, N, U3>,
    cam: &MatrixMN<R, N, U2>,
) -> (MatrixMN<R, Dynamic, U11>, DVector<R>)
where
    R: RealField,
    N: Dim,
    DefaultAllocator:
        Allocator<R, N, U3> + Allocator<R, N, U2> + Allocator<R, N, U11> + Allocator<R, N>,
{
    let n_pts = world.nrows();
    let mut b_data = Vec::with_capacity(n_pts * 2 * 11);
    let mut c_data = Vec::with_capacity(n_pts * 2);

    let zero = nalgebra::convert(0.0);
    let one = nalgebra::convert(1.0);

    for i in 0..n_pts {
        let X = world[(i, 0)];
        let Y = world[(i, 1)];
        let Z = world[(i, 2)];
        let x = cam[(i, 0)];
        let y = cam[(i, 1)];

        let b1 = [X, Y, Z, one, zero, zero, zero, zero, -x * X, -x * Y, -x * Z];
        let b2 = [zero, zero, zero, zero, X, Y, Z, one, -y * X, -y * Y, -y * Z];
        b_data.extend_from_slice(&b1);
        b_data.extend_from_slice(&b2);

        c_data.push(x);
        c_data.push(y);
    }

    #[allow(non_snake_case)]
    let B = MatrixMN::<R, Dynamic, U11>::from_row_slice(&b_data);
    let c = DVector::<R>::from_column_slice(&c_data);

    (B, c)
}

/// Return the least-squares solution to a linear matrix equation.
fn lstsq<R, M, N>(
    a: MatrixMN<R, M, N>,
    b: &VectorN<R, M>,
    epsilon: R,
) -> Result<VectorN<R, N>, &'static str>
where
    R: RealField,
    M: DimMin<N>,
    N: Dim,
    DimMinimum<M, N>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<R, M, N>
        + Allocator<R, N>
        + Allocator<R, M>
        + Allocator<R, DimDiff<DimMinimum<M, N>, U1>>
        + Allocator<R, DimMinimum<M, N>, N>
        + Allocator<R, M, DimMinimum<M, N>>
        + Allocator<R, DimMinimum<M, N>>,
{
    // calculate solution with epsilon
    let svd = nalgebra::linalg::SVD::new(a, true, true);
    let solution = svd.solve(&b, epsilon)?;

    Ok(solution)
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
    R: RealField,
    N: Dim + DimMin<U11>,
    DimMinimum<N, U11>: DimSub<U1>,
    DefaultAllocator: Allocator<R, N, U11>
        + Allocator<R, U11>
        + Allocator<R, N>
        + Allocator<R, N, U3>
        + Allocator<R, N, U2>,
{
    #[allow(non_snake_case)]
    let (B, c) = build_Bc(&world, &cam);

    let solution: VectorN<R, U11> = lstsq(B, &c, epsilon)?;

    let one = nalgebra::convert(1.0);
    let mut pmat_data = solution.as_slice().to_vec();
    pmat_data.push(one);
    let pmat = MatrixMN::<R, U3, U4>::from_row_slice(&pmat_data);

    Ok(pmat)
}

/// A point with a view in image (2D) and world (3D).
///
/// Used by the [`dlt_corresponding`](fn.dlt_corresponding.html) function as a
/// convenience compared to calling the [`dlt`](fn.dlt.html) function directly.
#[derive(Debug)]
pub struct CorrespondingPoint {
    /// the location of the point in 3D world coordinates
    pub object_point: (f64, f64, f64),
    /// the location of the point in 2D pixel coordinates
    pub image_point: (f64, f64),
}

/// Convenience wrapper around the [`dlt`](fn.dlt.html) function.
///
/// This allows using the [`CorrespondingPoint`](struct.CorrespondingPoint.html)
/// if you find that easier.
pub fn dlt_corresponding(
    points: &[CorrespondingPoint],
    epsilon: f64,
) -> Result<MatrixMN<f64, U3, U4>, &'static str> {
    // build matrices from input data
    let world_data: Vec<f64> = points
        .iter()
        .map(|p| vec![p.object_point.0, p.object_point.1, p.object_point.2])
        .flatten()
        .collect();
    let world_mat = nalgebra::MatrixMN::<f64, nalgebra::Dynamic, U3>::from_row_slice(&world_data);

    let image_data: Vec<f64> = points
        .iter()
        .map(|p| vec![p.image_point.0, p.image_point.1])
        .flatten()
        .collect();
    let image_mat = nalgebra::MatrixMN::<f64, nalgebra::Dynamic, U2>::from_row_slice(&image_data);

    // perform the DLT
    dlt(&world_mat, &image_mat, epsilon)
}

#[cfg(test)]
mod tests {
    use nalgebra::{Dynamic, MatrixMN, U2, U3, U4, U8};

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
