#!/usr/bin/env python
import numpy as np
from skspatial.objects import Points


# functions
def unit_vector(v: np.ndarray) -> np.ndarray:
    """
    Calculates the unit vector of a given vector.

    Args:
        v: A numpy array representing the vector.

    Returns:
        A numpy array representing the unit vector.
    """
    return v / np.linalg.norm(v)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculates the angle in radians between two vectors.

    Args:
        u: A numpy array representing the first vector.
        v: A numpy array representing the second vector.

    Returns:
        The angle in radians between the two vectors.
    """
    v_1 = unit_vector(v)
    u_1 = unit_vector(u)
    return np.arccos(np.clip(np.dot(v_1, u_1), -1.0, 1.0))


def hh_min_rot(symbols: list, coords_list: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """
    Performs a rotation of a molecular fragment to minimize the distance between
    two hydrogen atoms (H-H distance). This is typically used to optimize the
    orientation of a substituent on a molecule.

    The function identifies a "skeleton" and a "rotatable" fragment within the
    provided coordinates. It then iteratively rotates the rotatable fragment
    around a specified vector and calculates the H-H distance. The rotation that
    results in the minimum H-H distance without causing atomic clashes is applied.

    Args:
        symbols (list): List of atomic symbols. This argument is not directly used
                        in the rotation but is maintained for compatibility with
                        other functions.
        coords_list (np.ndarray): A numpy array of atomic coordinates. The last two
                                  atoms are assumed to be the hydrogen atoms of interest.
        rotvec (np.ndarray): The rotation vector (axis) around which the fragment
                             will be rotated.

    Returns:
        np.ndarray: A numpy array of the new coordinates with the rotated fragment.
    """
    coords_list = np.array(coords_list)
    hhref = np.linalg.norm(coords_list[-1] - coords_list[-2])  # Must be the HH distance
    bref = coords_list[-11]  # Must be the B position
    print(f"Starting hh distance was {hhref}")
    skeleton = coords_list[:-11]  # Must be the fragment to rotate
    rotatable_ref = coords_list[-11:-1].T
    rotatable_new = None
    nh = coords_list[-1]
    for i in np.linspace(0, 360, 720):
        rotmat = g_rot_matrix(i, rotvec)
        rotated = np.dot(rotmat, rotatable_ref).T
        displace = np.array(bref - rotated[0]).reshape(1, 3)
        rotated = rotated + displace
        hhdist = np.linalg.norm(nh - rotated[-1])
        if hhdist < hhref and not clash(skeleton, rotated):
            hhref = hhdist
            rotatable_new = rotated
    if rotatable_new is None:
        rotatable_new = rotatable_ref.T
    return np.vstack([skeleton, rotatable_new, nh])


def clash(skeleton: np.ndarray, rotated: np.ndarray) -> bool:
    """
    Checks for clashes between a skeleton and a rotated fragment.

    Args:
        skeleton: A numpy array representing the coordinates of the skeleton.
        rotated: A numpy array representing the coordinates of the rotated fragment.

    Returns:
        True if a clash is detected, False otherwise.
    """
    for i in rotated[::-1]:
        for j in skeleton[::-1]:
            d = np.linalg.norm(i - j)
            if d < 1.1:
                return True
    return False


def write_xyz(path: str, name: str, natoms: int, new_symbols: list, coords: np.ndarray):
    """
    Writes atomic coordinates to an XYZ file.

    Args:
        path: The directory path to save the XYZ file.
        name: The name of the XYZ file (without extension).
        natoms: The number of atoms.
        new_symbols: A list of atomic symbols.
        coords: A numpy array of atomic coordinates.
    """
    f = open(path + name + ".xyz", "w")
    f.write(str(natoms))
    f.write("\n")
    f.write(name + "_I2")
    f.write("\n")
    for a, b in enumerate(coords):
        line = new_symbols[a]
        line = line + "    " + str(b[0]) + "    " + str(b[1]) + "    " + str(b[2])
        print(line)
        f.write(line)
        f.write("\n")
    f.close()


def g_rot_matrix(degree: float = 0.0, axis: np.ndarray = np.asarray([1, 1, 1]), verb_lvl: int = 0) -> np.ndarray:
    """
    Calculates the rotation matrix for a given degree and axis.

    Args:
        degree: The angle of rotation in degrees. Defaults to 0.0.
        axis: The axis of rotation as a numpy array. Defaults to [1, 1, 1].
        verb_lvl: Verbosity level. Defaults to 0.

    Returns:
        The 3x3 rotation matrix.
    """
    try:
        theta = degree * (np.pi / 180)
    except:
        degree = float(degree)
        theta = degree * (np.pi / 180)
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
    if verb_lvl > 1:
        print("Rotation matrix generated.")
    return rot


def calculate_angle(z: np.ndarray, crd_N: int, crd_B: int, cm: np.ndarray, coords: np.ndarray) -> float:
    """
    Calculates the angle between two vectors defined by substituents around two central atoms.

    Args:
        z: A numpy array with atomic numbers (or symbols).
        crd_N: The index of the first central atom (FLP base).
        crd_B: The index of the second central atom (FLP acid).
        cm: The connectivity matrix.
        coords: A numpy array of atomic coordinates.

    Returns:
        The calculated angle in degrees, or 0.0 if the angle cannot be determined.
    """
    subs_base = []
    for n, j in enumerate(cm[crd_N, :]):
        if j == 1:
            pos = coords[n]
            pos = np.around(pos, 4)
            subs_base.append(pos)

    if len(subs_base) == 1:
        return 0.0
    elif len(subs_base) == 2:
        subs_base.append(coords[crd_N, :])
        points = Points(subs_base)
        try:
            centroid = points.centroid()
        except ValueError:
            return 0.0
        vec_b = np.array(coords[crd_N, :]) - centroid

    elif len(subs_base) == 3:
        points = Points(subs_base)
        try:
            centroid = points.centroid()
        except ValueError:
            return 0.0
        vec_b = np.array(coords[crd_N, :]) - centroid
    else:
        return 0.0

    # Vector on N center
    vec_b = unit_vector(vec_b)

    # Acid vector definition
    subs_acid = []
    for n, j in enumerate(cm[crd_B, :]):
        if j == 1:
            pos = coords[n]
            pos = np.around(pos, 4)
            subs_acid.append(pos)

    if len(subs_acid) == 3:
        points = Points(subs_acid)
        try:
            centroid = points.centroid()
        except ValueError:
            return 0.0
        vec_a = np.array(coords[crd_B, :]) - centroid
    else:
        return 0.0

    # Vectro on B center
    vec_a = unit_vector(vec_a)

    # Calculate vector
    theta = angle_between(vec_a, vec_b) * 180 / np.pi
    return theta
