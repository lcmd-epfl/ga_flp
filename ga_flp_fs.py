#!/usr/bin/env python
# Functions for ga_flp.py

import math
import os
import re
import subprocess as sp
import timeit
from subprocess import PIPE, run

import numpy as np
from navicatGA.chemistry_smiles import get_structure_ff
try:
    from frustration_c import predict_from_xyz, frustration_model_loaded
except ImportError:
    print("Warning: frustration_c module not found. Frustration prediction will be skipped.")
    predict_from_xyz = None
    frustration_model_loaded = False
except Exception as e:
    print(f"Warning: Error importing frustration_c: {e}. Frustration prediction will be skipped.")
    predict_from_xyz = None
    frustration_model_loaded = False
from rdkit import Chem
from rdkit.Chem import AllChem, Get3DDistanceMatrix, MolFromSmiles, rdDistGeom
from scipy.spatial import distance
from skspatial.objects import Plane, Points

# Helper function


def unit_vector(v):
    return v / np.linalg.norm(v)


# To get the most optimal B-N distance(closest to the target)


def get_BN_index(mol, dist_matrix, Best_dist):
    BN = np.where(dist_matrix == Best_dist)[1]
    crd_N = BN[0]
    crd_B = BN[1]
    i = 0
    for atom in mol.GetAtoms():
        if i == crd_N and (atom.GetAtomicNum() == 7):
            # print("Correct index assignment")
            count = 0
            break
        elif i == crd_N and (atom.GetAtomicNum() == 5):
            crd_N, crd_B = crd_B, crd_N
            count = 1
            break
        i += 1
    return crd_N, crd_B, count


def get_BN_dist(mol, dist_matrix, Threshold):
    """
    Measure the shortest B-N distance but higher than the threshold value
    Parameters:
    1. mol
    2. dist_matrix: distance matrix
    3. Threshold: dative B-N bond distance

    Returns:
    d(B-N) (float)
    """

    N_index = []
    B_index = []
    i = 0
    for atom in mol.GetAtoms():
        i += 1
        if atom.GetAtomicNum() == 5:
            B_index.append(i)
        elif atom.GetAtomicNum() == 7:
            N_index.append(i)
        else:
            pass
    BN_list = np.asarray([dist_matrix[i - 1, j - 1] for j in B_index for i in N_index])
    if np.size(BN_list) > 0:
        for t in np.linspace(Threshold, 0):
            posBNL = BN_list[BN_list > Threshold]
            if np.size(posBNL) > 0:
                return np.min(BN_list)
        return np.min(BN_list)
    else:
        return 0


# Embedding function


def embed_mol(smiles, n_confs=25):
    """
    Parameters:
    1. smiles
    2. n_confs: number of confomers considered

    Returns:
    mol
    """

    # print("Running embedding.")
    mol = MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    mol = cleanup_mol(mol, n_confs)

    return mol


def cleanup_mol(mol, n_confs=20):
    """
    Input: "bad confomer id" mol

    clean up the mol with bad conformer id

    Return: mol object
    """

    # Directly use get_structure_ff
    try:
        mol_2 = get_structure_ff(mol, n_confs)
        return mol_2
    # Method 3: for "Bad system"
    except BaseException:
        ps = rdDistGeom.ETKDG()
        ps.useRandomCoords = True
        ps.ignoreSmoothingFailures = True
        # print(
        #    "Problematic system encountered, trying to use manual method instead of navicatga wrapper."
        # )
        conf_id = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=ps, maxAttempts=500)
    try:
        assert mol.GetNumConformers() >= 1
        assert mol.GetConformer().Is3D()
    except AssertionError:
        # print("Randomcoords method also failed, returning naked mol.")
        return mol
    if Chem.rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
        results_UFF = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=0)
        index = 0
        min_energy_UFF = np.inf
        min_energy_UFF = 0
        for index, result in enumerate(results_UFF):
            if min_energy_UFF > result[1]:
                min_energy_UFF = result[1]
                min_energy_index_UFF = index
        mol_2 = Chem.Mol(mol, index)
    elif Chem.rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        results_MMFF = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0)
        index = 0
        min_energy_MMFF = np.inf
        min_energy_MMFF = 0
        for index, result in enumerate(results_MMFF):
            if min_energy_MMFF > result[1]:
                min_energy_MMFF = result[1]
                min_energy_index_MMFF = index
        mol_2 = Chem.Mol(mol, index)
    else:
        # print("Forcefield typing failed, returning first conformer.")
        mol_2 = Chem.Mol(mol, 0)
    return mol_2


def GS_FLP(smiles, n_confs=10):
    """
    input: SMILES string

    get the input arguements of GS FLP for get_H2_pos()

    output: z = vector of atomic numbers;
           crd_N, crd_B: atomic indices of N and B
           cm = connectivity matrix
           coords = coordinates matrix
    """

    # Threshold for the B-N covalent bond distance.
    Threshold = 1.60

    mol = embed_mol(smiles, n_confs)

    dist_matrix = []
    n = 10
    while len(dist_matrix) == 0:
        if n > 50:
            break
        try:
            dist_matrix = Get3DDistanceMatrix(mol)
        except ValueError:
            # print(f"Trying to generate dist_mat again with n = {n}")
            mol_FLP_H2 = cleanup_mol(mol, n)
            n += 3

    if len(dist_matrix) == 0:
        # print("Ridiculous FLPs. Fitness will be punished.")
        return 0, 0, 0, 0, 0, 0
    # dist_matrix = Get3DDistanceMatrix(mol)

    # F1, d(B-N)
    bn_distance = get_BN_dist(mol, dist_matrix, Threshold)

    if bn_distance == 0:
        # print("Ridiculous FLPs. Fitness will be punished.")
        return 0, 0, 0, 0, 0, 0

    # F2; angle

    # A. Connectivity matrix(or Adjacency matrix)
    cm = Chem.rdmolops.GetAdjacencyMatrix(mol)

    # B. list of atomic number
    z = []
    for atom in mol.GetAtoms():
        z.append(atom.GetAtomicNum())

    crd_N, crd_B, count = get_BN_index(mol, dist_matrix, bn_distance)

    # D. Coordinate matrix
    for c in mol.GetConformers():
        coords = c.GetPositions()

    return z, crd_N, crd_B, cm, coords, mol


def get_H2_pos(z, crd_N, crd_B, cm, coords):
    """
    Determine cartesian coodinates to which hydride and proton are to be added.

    Vector definition on N: The centroid point of substituent groups - Coordinate of N
    Vector definition on B: Coordinate of H(N) - Coordinate of B

    Set bond distance:
    d(B-H) = 1.2
    d(N-H) = 1.0

    Parameters:
    1. z: vector of atomic numbers
    2. crd_N: atomic index of N
    3. crd_B: atomic index of B
    4. cm: connectivity matrix
    5. coords: coordinates matrix

    Returns:
    H_a, H_b: coordinate of H(B) and H(N), respectively
    """

    # Base vector definition
    subs_base = []
    for n, j in enumerate(cm[crd_N, :]):
        if j == 1:
            pos = coords[n]
            pos = np.around(pos, 4)
            subs_base.append(pos)
    if len(subs_base) == 2:
        # print("\nsp2 base requires perpendicular plane. Calculating planes.")

        subs_base.append(coords[crd_N, :])
        points = Points(subs_base)
        try:
            centroid = points.centroid()
        except ValueError:
            return [0, 0, 0], [0, 0, 0]
        vec_b = np.array(coords[crd_N, :]) - centroid

    elif len(subs_base) == 3:
        # print("\nsp3 base. Calculating planes.")
        points = Points(subs_base)
        try:
            centroid = points.centroid()
        except ValueError:
            return [0, 0, 0], [0, 0, 0]

        # flat sp3(N): N-embedded in the aromatic ring
        if distance.euclidean(centroid, coords[crd_N, :]) < 0.1:
            # print(f"Flat sp3(N), centroid too close to N: {distance.euclidean(centroid, coords[crd_N,:])}")
            try:
                plane = Plane.best_fit(points)
                n1 = plane.normal
                n2 = -n1
                # H_b1 = n1*1.2 + coords[crd_N,:]
                H_b1 = n1 * 1.2
                H_b2 = n2 * 1.2
                d1 = distance.euclidean(H_b1, coords[crd_B, :])
                d2 = distance.euclidean(H_b2, coords[crd_B, :])
                if d1 < d2:
                    vec_b = H_b1
                else:
                    vec_b = H_b2
            except ValueError:
                return [0, 0, 0], [0, 0, 0]
        # trigonal pyramidal sp3(N)
        else:
            vec_b = np.array(coords[crd_N, :]) - centroid

    else:
        # print(f"Base substituents are not 2 or 3 but rather {len(subs_base)}!")
        return [0, 0, 0], [0, 0, 0]
    vec_b = unit_vector(vec_b)
    # Supposed position of H(N)
    H_b = vec_b * 1.0 + coords[crd_N, :]

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
            plane = Plane.best_fit(points)
            n1 = plane.normal
            n2 = -n1
            H_a1 = n1 * 1.2 + coords[crd_B, :]
            H_a2 = n2 * 1.2 + coords[crd_B, :]
            d1 = distance.euclidean(H_a1, H_b)
            d2 = distance.euclidean(H_a2, H_b)
            if d1 < d2:
                H_a = H_a1
            else:
                H_a = H_a2
        except ValueError:
            return [0, 0, 0], [0, 0, 0]
    else:
        # print("Acid substituents are not 3!")
        return [0, 0, 0], [0, 0, 0]

    return H_a, H_b


def append_H2(H_a, H_b, mol, crd_N, crd_B):
    """
    input: coodinates of the substrate,
           mol object of the FLP

    append substrate information into the FLP mol object, and proceed to
    employ get_structure_ff to optimize for the relaxed geom

    output: optimized relaxed geom in the mol
    """

    Chem.rdmolfiles.MolToMolFile(mol, "original.mol")

    H_a = [round(n, 4) for n in H_a]
    H_b = [round(n, 4) for n in H_b]

    f = "   {0:>%d}   {1:>%d}   {2:>%d} H   0  0  0  0  0  0  0  0  0  0  0  0" % (
        7,
        7,
        7,
    )
    Ha = f.format(H_a[0], H_a[1], H_a[2])
    Hb = f.format(H_b[0], H_b[1], H_b[2])

    with open("original.mol", "r") as f:
        content = f.readlines()

    try:
        n_tot = int(content[3].split(" ")[1])
        b_bond = int(content[3].split(" ")[2])
    except BaseException:
        if content[3].split(" ")[0] != "":
            n_tot = int(content[3].split(" ")[0][0:3])
            b_bond = int(content[3].split(" ")[0][3:6])
        else:
            n_tot = int(content[3].split(" ")[1][0:2])
            b_bond = int(content[3].split(" ")[1][2:5])

    f = "{0:>%d}{1:>%d}  1  0" % (3, 3)
    HN = f.format(int(crd_N) + 1, n_tot + 1)
    HB = f.format(int(crd_B) + 1, n_tot + 2)

    with open("FLP_H2.mol", "w") as f:
        f.writelines(content[: int(n_tot) + 4])

    with open("FLP_H2.mol", "a") as f:
        f.write(Ha + "\n")
        f.write(Hb)
        f.write("\n")
        f.writelines(content[4 + int(n_tot) : 4 + int(n_tot) + int(b_bond)])
        f.write(HN + "\n")
        f.write(HB + "\n")
        f.write(f"M  CHG  1  {int(crd_B)+1}  {-1}\n")
        f.write(f"M  CHG  1  {int(crd_N)+1}  {+1}\n")
        f.writelines(content[4 + int(n_tot) + int(b_bond) : len(content)])

    with open("FLP_H2.mol", "r") as f:
        content = f.readlines()

    f = "{0:>%d}{1:>%d}  0  0  0  0  0  0  0  0999 V2000\n" % (3, 3)
    sui = f.format(n_tot + 2, b_bond + 2)
    content[3] = sui

    suiB = list(content[4 + crd_B])
    suiB[38] = str(5)
    suiB = "".join(suiB)
    content[4 + crd_B] = suiB

    suiN = list(content[4 + crd_N])
    suiN[38] = str(3)
    suiN = "".join(suiN)
    content[4 + crd_N] = suiN

    with open("FLP_H2.mol", "w") as f:
        f.writelines(content)

    mol_FLP_H2 = Chem.rdmolfiles.MolFromMolFile(
        "FLP_H2.mol", removeHs=False, sanitize=False
    )

    mol_FLP_H2 = cleanup_mol(mol)

    return mol_FLP_H2


def del_substrate(mol_origin="original.mol", mol_FLP_H2="FLP_H2.mol"):
    """
    input: (fixed) mol files of GS FLP and FLP-H2 structure

    extract the relaxed FLP from FLP-H2

    output: mol object of the relaxed FLP
    """

    with open(mol_origin, "r") as f:
        content = f.readlines()

    top = content[:4]
    if len(top[3].split(" ")[0]) == 6:
        n_tot = int(top[3].split(" ")[0][0:3])
    elif len(top[3].split(" ")[1]) == 5:
        n_tot = int(top[3].split(" ")[1][0:2])
    else:
        n_tot = int(top[3].split(" ")[1])
    tail = content[4 + int(n_tot) : len(content)]

    with open(mol_FLP_H2, "r") as f:
        content = f.readlines()
    content_xyz = content[4 : int(n_tot) + 4]

    content_FLP = top + content_xyz + tail
    with open("FLP_relaxed.mol", "w") as f:
        f.writelines(content_FLP)

    mol_FLP_s = Chem.rdmolfiles.MolFromMolFile(
        "FLP_relaxed.mol", removeHs=False, sanitize=False
    )

    return mol_FLP_s


def sa_score(smiles):
    execution = [
        "python",
        "/home/sdas/scscore/scscore/standalone_model_numpy.py",
        smiles,
    ]
    sc = run(execution, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    temp = sc
    with open(f"scscore.out", "a") as f:
        f.writelines(sc.stdout)
    temp1 = sc.stdout
    scscore = 0.0  # Initialize scscore to a default value
    for part in temp1.splitlines():
        if re.search(r"scs", part):
            scscore = re.findall(r"(?<![a-zA-Z:])[-+]?\d*\.?\d+", part)[0]
    start = timeit.default_timer()

    # All the program statements
    stop = timeit.default_timer()
    execution_time = stop - start
    # print("scscore Executed in " + str(execution_time))  # It returns time in seconds
    return float(scscore)


def get_frustration(B_index, N_index):
    if not frustration_model_loaded or predict_from_xyz is None:
        print("Warning: Frustration prediction is unavailable. Returning 0.")
        return 0  # Return a default value when frustration prediction is skipped

    start = timeit.default_timer()
    res = predict_from_xyz("Catalyst_relax_xbtopt.xyz", B_index, N_index)
    stop = timeit.default_timer()
    execution_time = stop - start
    # print(
    #    f"Frustration {res}, classifier Executed in {str(execution_time)}"
    # )  # It returns time in seconds
    return res

    # execution = [
    #    "python",
    #    "/home/sdas/GA_FLP/38-scs/frustration_c.py",
    #    "Catalyst_relax_xbtopt.xyz",
    #    f"{B_index}",
    #    f"{N_index}",
    # ]
    # fr = run(execution, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    # with open(f"frustration.out", "a") as f:
    #    f.writelines(fr.stdout)
    # temp1 = fr.stdout
    # for part in temp1.splitlines():
    #    if re.search(r"Frustration", part):
    #        res = re.findall(r"(?<![a-zA-Z:])[-+]?\d*\.?\d+", part)[0]
    # start = timeit.default_timer()
    ## All the program statements
    # stop = timeit.default_timer()
    # execution_time = stop - start
    # print(
    #    f"Frustration {res}, classifier Executed in {str(execution_time)}"
    # )  # It returns time in seconds
    # return float(res)


def xtb_opt(xyz, charge=0, unpaired_e=0, level=2, cycle=500, irun=0):
    """
    Perform quick and reliable geometry optimization with xtb module
    Parameters:
    1. xyz: xyz file
    2. charge: (int)
    3. unpaired_e: number of unpaired electrons
    4. level: 0-2; Halmitonian level [gfn-1,2, gfnff], default=2
    5. cycle: number of maximum cycles

    Returns:
    none
    (filename_xbtopt.xyz)
    """

    if level == 0:
        execution = ["xtb", "--gfnff", xyz, "--opt", "vtight", "--cycles", str(cycle)]
    elif level == 1:
        execution = [
            "xtb",
            "--gfn",
            "1",
            xyz,
            "--opt",
            "vtight",
            "--cycles",
            str(cycle),
        ]
    elif level == 2:
        execution = ["xtb", "--gfn", "2", xyz, "--opt", "--cycles", str(cycle)]
    elif level == 3:
        execution = ["xtb", "--gfn", "2", xyz, "--opt", "loose", "--cycles", str(cycle)]
    elif level == 4:
        execution = [
            "xtb",
            "--gfn",
            "2",
            xyz,
            "--opt",
            "sloppy",
            "--cycles",
            str(cycle),
        ]
    if charge != 0:
        execution.extend(["--charge", str(charge)])
    if unpaired_e != 0:
        execution.extend(["--uhf", str(unpaired_e)])
    # execution.extend([">", "/dev/null"])
    out = run(execution, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    name = xyz[:-4]
    temp1 = out.stdout
    energy = None
    for part in temp1.splitlines():
        # print(part)
        if re.search(r"TOTAL ENERGY", part):
            # print(part)
            # if 'TOTAL ENERGY' in part:
            energy = re.findall(r"(?<![a-zA-Z:])[-+]?\d*\.?\d+", part)[0]
            # energy = float(energy)
        # temp2 = part.split('|')[1]
        # energy = temp2.split(" ")[-1]
        # print('The total energy is', energy)
    bad = False
    if energy is None:
        bad = True
        if irun == 0:
            bad, energy = xtb_opt(xyz, charge, unpaired_e, level=3, cycle=500, irun=1)
        if irun == 1:
            bad, energy = xtb_opt(xyz, charge, unpaired_e, level=4, cycle=500, irun=2)
    if not bad and irun == 0:
        with open(f"{name}_xtb.out", "w") as f:
            f.writelines(out.stdout)
        with open(f"{name}_energy.out", "w") as f:
            f.writelines(energy)
            energy = float(energy)
        try:
            os.rename("xtbopt.xyz", f"{name}_xbtopt.xyz")
        except BaseException:
            os.rename("xtblast.xyz", f"{name}_xbtopt.xyz")
    return bad, energy


def gen_relaxed_FLP_H2_xtb(smiles, level=2):
    """
    generate mol object of relaxed IFLP in INT2, FLP_H2 optimized at gfn-ffs
    Parameters:
    1. smiles: SMILES of IFLP
    2. level: default = 0
    Returns:
    mol
    """

    Threshold = 1.4

    mol = embed_mol(smiles)

    dist_matrix = []
    n = 10
    while len(dist_matrix) == 0:
        if n > 50:
            break
        try:
            dist_matrix = Get3DDistanceMatrix(mol)
        except ValueError:
            # print(f"Trying to generate dist_mat again with n = {n}")
            mol_FLP_H2 = cleanup_mol(mol, n)
            n += 3

    if len(dist_matrix) == 0:
        # print(
        #    "Ridiculous FLPs cannot produce a valid distance matrix. Fitness will be punished."
        # )
        return mol, 0.0, 0.0, True

    bn_distance = get_BN_dist(mol, dist_matrix, Threshold)
    if bn_distance == 0:
        # print("Ridiculous FLPs. Fitness will be punished.")
        return mol, 0.0, 0.0, True
    cm = Chem.rdmolops.GetAdjacencyMatrix(mol)
    z = []
    for atom in mol.GetAtoms():
        z.append(atom.GetAtomicNum())

    Best_dist = get_BN_dist(mol, dist_matrix, Threshold)
    if bn_distance == 0:
        # print("Ridiculous FLPs. Fitness will be punished.")
        return mol, 0.0, 0.0, True

    crd_N, crd_B, count = get_BN_index(mol, dist_matrix, Best_dist)

    for c in mol.GetConformers():
        coords = c.GetPositions()

    bad = False
    H_a, H_b = get_H2_pos(z, crd_N, crd_B, cm, coords)
    H_a = np.array(H_a)
    H_b = np.array(H_b)
    if not H_a.any() and not H_b.any():
        # print(
        #    f"get_H2_pos failed and returned {H_a} and {H_b}. This will be punished in fitness."
        # )
        bad = True
    if not bad:
        AllChem.MolToXYZFile(mol, "Catalyst.xyz")

        H_A = f"H  {H_a[0]}   {H_a[1]}   {H_a[2]}\n"
        H_B = f"H  {H_b[0]}   {H_b[1]}   {H_b[2]}"

        with open("Catalyst.xyz", "r") as f:
            contents = f.readlines()

        contents.extend([H_A, H_B])
        contents[0] = str(len(contents[2:])) + "\n"

        with open("Catalyst_FLP_H2.xyz", "w") as f:
            f.writelines(contents)

        bad, energy = xtb_opt("Catalyst_FLP_H2.xyz", level=2)
        if bad is True:
            pass
            # print("I2 relaxation XTB run had issues. This will be punished in fitness.")

        with open("Catalyst_FLP_H2_xbtopt.xyz", "r") as f:
            contents = f.readlines()
            n = len(contents) - 3
            # print(contents[-1])
        with open("Catalyst_FLP_H2_energy.out", "r") as f:
            energy_H2 = f.readlines()[0]

        with open("Catalyst_relax.xyz", "w") as f:
            f.write(str(len(z)) + "\n\n")
            f.writelines(contents[2:-2])

        sp.call(["xtb", "Catalyst_relax.xyz"], stdout=sp.DEVNULL, stderr=sp.STDOUT)
        bad, energy = xtb_opt("Catalyst_relax.xyz", level=2)
        if bad is True:
            # print(
            #    "Catalyst relaxation XTB run had issues. This will be punished in fitness."
            # )
            pass
        with open("Catalyst_relax_energy.out", "r") as f:
            energy_n = f.readlines()[0]
        # energy_n = energy
        # e_rrs_i2 = (float(energy_H2) - float(energy_n))*627.509
        # print("RRS I2", e_rrs_i2)
        if count == 0:  #####fixing depending on the indexing of B and N: SD
            with open("Catalyst_FLP_BH.xyz", "w") as f:
                f.write(str(n) + "\n\n")
                f.writelines(contents[2:-1])
            with open("Catalyst_FLP_NH.xyz", "w") as f:
                coords = np.append(contents[2:-2], contents[-1])
                f.write(str(n) + "\n\n")
                f.writelines(coords)
                # print(count)
                # break
        elif count == 1:
            with open("Catalyst_FLP_NH.xyz", "w") as f:
                f.write(str(n) + "\n\n")
                f.writelines(contents[2:-1])
            with open("Catalyst_FLP_BH.xyz", "w") as f:
                coords = np.append(contents[2:-2], contents[-1])
                f.write(str(n) + "\n\n")
                f.writelines(coords)
                # print(count)
                # break

        mol_FLP_s = Chem.rdmolfiles.MolFromMolFile(
            "xtbtopo.mol", removeHs=False, sanitize=False
        )
    if bad:
        mol_FLP_s = None
        energy_H2 = 0.0
        energy_n = 0.0
    return mol_FLP_s, float(energy_H2), float(energy_n), bad


def calc_paha(energy_n):
    """
    generate mol object of relaxed IFLP from the xtb optimized geometry of I2
    1. smiles: SMILES of IFLP
    2. level: default = 0
    Returns:
    mol
    """
    bad, energy = xtb_opt("Catalyst_FLP_BH.xyz", charge=-1, level=2)
    if bad is True:
        # print("BH XTB run had issues")
        pass
    with open("Catalyst_FLP_BH_energy.out", "r") as f:
        energy_H = f.readlines()[0]
    feha = (float(energy_H) - energy_n - (-0.610746694539)) * 627.509

    bad, energy = xtb_opt("Catalyst_FLP_NH.xyz", charge=1, level=2)
    if bad is True:
        # print("NH XTB run had issues")
        pass
    with open("Catalyst_FLP_NH_energy.out", "r") as f:
        energy_P = f.readlines()[0]
    fepa = (float(energy_P) - energy_n) * 627.509

    return feha, fepa


def chem_fit(point, coef):
    chem_dis = abs((coef[0] * point[0]) - point[1] + coef[1]) / math.sqrt(
        (coef[0] * coef[0]) + 1
    )
    return chem_dis


def smiles_2_mol(smiles, level=2):
    """
    generate an xtb-optimized mol object given a smiles string
    """
    mol = MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    mol = get_structure_ff(mol, 20)
    AllChem.MolToXYZFile(mol, "Catalyst.xyz")
    bad, energy = xtb_opt("Catalyst.xyz", level=level)
    if bad is True:
        # print("XTB run had issues")
        pass

    sp.call(["xtb", "Catalyst_xbtopt.xyz"], stdout=sp.DEVNULL, stderr=sp.STDOUT)
    mol_xtb = Chem.rdmolfiles.MolFromMolFile(
        "xtbtopo.mol", removeHs=False, sanitize=False
    )

    return mol_xtb
