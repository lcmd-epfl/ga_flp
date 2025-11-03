#!/usr/bin/env python
import numpy as np
from skspatial.objects import Plane, Points


# functions
def unit_vector(v):
    return v / np.linalg.norm(v)


def angle_between(u, v):
    v_1 = unit_vector(v)
    u_1 = unit_vector(u)
    return np.arccos(np.clip(np.dot(v_1, u_1), -1.0, 1.0))


def hh_min_rot(symbols, coords_list, rotvec):
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


def clash(skeleton, rotated):
    for i in rotated[::-1]:
        for j in skeleton[::-1]:
            d = np.linalg.norm(i - j)
            if d < 1.1:
                return True
    return False


def write_xyz(path, name, natoms, new_symbols, coords):
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


def g_rot_matrix(degree="0.0", axis=np.asarray([1, 1, 1]), verb_lvl=0):
    """
       Return the rotation matrix associated with counterclockwise rotation about
       the given axis by theta radians.
    Parameters
    ----------
    axis
        Axis of the rotation. Array or list.
    degree
        Angle of the rotation in degrees.
    verb_lvl
        Verbosity level integer flag.
    Returns
    -------
    rot
        Rotation matrix to be used.
    """
    try:
        theta = degree * (np.pi / 180)
    except:
        degree = float(degree)
        theta = degree * (np.pi / 180)
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
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


def calculate_angle(z, crd_N, crd_B, cm, coords):
    # Find or pass crd_N and crd_B which are the indices of the FLP base and acid respectively
    # z is a vector with the atomic numbers (symbols could be used instead)
    # cm is the connectivity matrix
    # coords is the coordinates matrix

    # Base vector definition
    subs_base = []
    for n, j in enumerate(cm[crd_N, :]):
        if j == 1:
            pos = coords[n]
            pos = np.around(pos, 4)
            subs_base.append(pos)
    if len(subs_base) == 1:
        # print("\nsp base identified. Setting angle to 0.0")
        return 0.0
    elif len(subs_base) == 2:
        # print("\nsp2 base requires perpendicular plane. Calculating planes.")

        subs_base.append(coords[crd_N, :])
        points = Points(subs_base)
        try:
            centroid = points.centroid()
        except ValueError:
            return 0.0
        vec_b = np.array(coords[crd_N, :]) - centroid

    elif len(subs_base) == 3:
        # print("\nsp3 base. Calculating planes.")
        points = Points(subs_base)
        try:
            centroid = points.centroid()
        except ValueError:
            return 0.0
        vec_b = np.array(coords[crd_N, :]) - centroid
    else:
        # print(
        #    f"Base substituents are not 2 or 3 but rather {len(subs_base)} Angle set to 0!"
        # )
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
        # print("\nsp3 acid. Calculating planes.")
        points = Points(subs_acid)
        try:
            centroid = points.centroid()
        except ValueError:
            return 0.0
        vec_a = np.array(coords[crd_B, :]) - centroid
    else:
        # print("Acid substituents are not 3!")
        return 0.0

    # Vectro on B center
    vec_a = unit_vector(vec_a)

    # Calculate vector
    theta = angle_between(vec_a, vec_b) * 180 / np.pi
    return theta
