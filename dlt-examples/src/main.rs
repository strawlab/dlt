use cam_geom::{Camera, Points};
use dlt::{dlt_corresponding, CorrespondingPoint};

fn main() {
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
    println!("Camera model:");
    println!("{}", pmat);
    println!("3D -> 2D projections:");
    let cam = Camera::from_perspective_matrix(&pmat).unwrap();
    for orig in points.iter() {
        let world = Points::new(nalgebra::RowVector3::from_row_slice(&orig.object_point));
        let px = cam.world_to_pixel(&world);
        approx::assert_relative_eq!(px.data.as_slice(), &orig.image_point[..], epsilon = 1e-4);
        println!("   {}    ->     {}", world.data, px.data);
    }
}
