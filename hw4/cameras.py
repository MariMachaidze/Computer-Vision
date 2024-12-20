from typing import Tuple

import numpy as np


def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    """Define a transformation matrix in homogeneous coordinates that
    transforms coordinates from world space to camera space, according
    to the coordinate systems in Question 1.


    Args:
        d (float, optional): Total distance of displacement between world and camera
            origins. Will always be greater than or equal to zero. Defaults to 1.0.

    Returns:
        T (np.ndarray): Left-hand transformation matrix, such that c = Tw
            for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
    """
   
    T = np.eye(4)

    theta = -5 * np.pi/4

    R = np.array([
        [np.cos(theta),  0, np.sin(theta)],
        [0,              1,            0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    t = np.array([0, 0, d]).reshape(3, 1)

    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    assert T.shape == (4, 4)
    return T


def apply_transform(T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray]:
    """Apply a transformation matrix to a set of points.

    Hint: You'll want to first convert all of the points to homogeneous coordinates.
    Each point in the (3,N) shape edges is a length 3 vector for x, y, and z, so
    appending a 1 after z to each point will make this homogeneous coordinates.

    You shouldn't need any loops for this function.

    Args:
        T (np.ndarray):
            Left-hand transformation matrix, such that c = Tw
                for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
        points (np.ndarray):
            Shape = (3,N) where 3 means 3D and N is the number of points to transform.

    Returns:
        points_transformed (np.ndarray):
            Transformed points.
            Shape = (3,N) where 3 means 3D and N is the number of points.
    """
    N = points.shape[1]
    assert points.shape == (3, N)

    ones = np.ones((1, N))

    homogeneous_coord = np.vstack((points, ones))
    transformed = np.dot(T,homogeneous_coord)
    points_transformed = transformed[:-1, :]

    assert points_transformed.shape == (3, N)
    return points_transformed


def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray: the intersection of the two lines definied by (a0, a1)
                    and (b0, b1).
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == float

    # Intersection point between lines
    out = np.zeros(2)

    a0_h = np.array([a_0[0], a_0[1], 1])
    a1_h = np.array([a_1[0], a_1[1], 1])
    b0_h = np.array([b_0[0], b_0[1], 1])
    b1_h = np.array([b_1[0], b_1[1], 1])

    line_a = np.cross(a0_h, a1_h)
    line_b = np.cross(b0_h, b1_h)

    intersection_h = np.cross(line_a, line_b)

    if intersection_h[2] != 0:  # Check for parallel lines
        out = np.array([
            intersection_h[0] / intersection_h[2],
            intersection_h[1] / intersection_h[2]
        ])
    else:
        # Lines are parallel
        out = np.zeros(2)


    assert out.shape == (2,)
    assert out.dtype == float

    return out

def optical_center_from_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your intersection_from_lines() implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape (2,).
        v1 (np.ndarray): Vanishing point in image space; shape (2,).
        v2 (np.ndarray): Vanishing point in image space; shape (2,).

    Returns:
        np.ndarray: Optical center; shape (2,).
    """
    assert v0.shape == v1.shape == v2.shape == (2,), "Wrong shape!"

    optical_center = np.zeros(2)
    # Set up perpendicular points for triangle sides
    a_perp_pt = np.zeros(2)
    b_perp_pt = np.zeros(2)
    
    # Calculate slopes of triangle sides
    
    # Handle v1-v0 side
    if v1[0] == v0[0]:  # Vertical line
        a_slope = np.inf
    else:
        a_slope = (v1[1] - v0[1])/(v1[0] - v0[0])
    a_perp_pt = v2  
    
    # Handle v2-v1 side
    if v1[0] == v2[0]:  # Vertical line
        b_slope = np.inf
    else:
        b_slope = (v2[1] - v1[1])/(v2[0] - v1[0])
    b_perp_pt = v0 
    
    # Calculate perpendicular slopes
    if a_slope != 0:
        a_perp = -1/a_slope 
    if b_slope != 0:
        b_perp = -1/b_slope
        
    # Create points for defining the lines
    a0 = a_perp_pt
    b0 = b_perp_pt
    
    # special scenarios for computation
    if a_slope == 0:  # Horizontal line
        a1 = a_perp_pt.copy()
        a1[1] += 1  #
    else:
        a1 = a0 + np.array([1, a_perp])  # Point along perpendicular slope
        
    if b_slope == 0: 
        b1 = b_perp_pt.copy()
        b1[1] += 1  
    else:
        b1 = b0 + np.array([1, b_perp])  
        
    # Find intersection of the perpendicular lines
    optical_center = intersection_from_lines(a0, a1, b0, b1)
    return optical_center


def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """
    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    f = None

    xs = (v0[0]-optical_center[0])*(v1[0]-optical_center[0])
    ys = (v0[1]-optical_center[1])*(v1[1]-optical_center[1])

    f2 = xs + ys
    
    f = np.sqrt(-1*f2)

    return float(f)


def physical_focal_length_from_calibration(
    f: float, sensor_diagonal_mm: float, image_diagonal_pixels: float
) -> float:
    """Compute the physical focal length of our camera, in millimeters.

    Args:
        f (float): Calibrated focal length, using pixel units.
        sensor_diagonal_mm (float): Length across the diagonal of our camera
            sensor, in millimeters.
        image_diagonal_pixels (float): Length across the diagonal of the
            calibration image, in pixels.

    Returns:
        float: Calibrated focal length, in millimeters.
    """
    f_mm = None

    f_mm = f*sensor_diagonal_mm/image_diagonal_pixels

    return f_mm
