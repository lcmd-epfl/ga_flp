"""
This module provides a collection of functions for computational chemistry tasks
related to Frustrated Lewis Pairs (FLPs). It includes functions for:

- Geometry optimization and electronic structure calculations using xtb.
- Molecular embedding and conformer generation.
- Analysis of molecular properties like B-N distance, angles, and affinities.
- File I/O for molecular structures (SMILES, XYZ, MOL).

These functions are designed to be used as building blocks in genetic algorithms
or other high-throughput screening workflows for FLP catalyst design.
"""


import math
import os
import sys
from pathlib import Path

# Add scscore to sys.path
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parents[1]
SCSCORE_DIR = ROOT_DIR / "scscore"
if str(SCSCORE_DIR) not in sys.path:
    sys.path.append(str(SCSCORE_DIR))

try:
    from scscore import standalone_model_numpy
except ImportError:
    print("Warning: Could not import scscore. sa_score will return 0.0.")
    standalone_model_numpy = None

import re
import subprocess as sp
import timeit
from subprocess import PIPE, run

import numpy as np
from .frustration_c import FrustrationPredictor
from navicatGA.chemistry_smiles import get_structure_ff
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Get3DDistanceMatrix,
    MolFromSmiles,
    rdDistGeom,
)
from scipy.spatial import distance
from skspatial.objects import Plane, Points


frustration_predictor = None


def initialize_frustration_predictor(model_path: str):
    """
    Initializes the global frustration predictor with a trained model.

    Args:
        model_path: The path to the saved classifier model.
    """
    global frustration_predictor
    frustration_predictor = FrustrationPredictor(model_path)


# Helper function


def unit_vector(v: np.ndarray) -> np.ndarray:
    """
    Calculates the unit vector of a given vector.

    Args:
        v: A numpy array representing the vector.

    Returns:
        A numpy array representing the unit vector.
    """
    return v / np.linalg.norm(v)


# To get the most optimal B-N distance(closest to the target)


def get_BN_index(
    mol: Chem.Mol, dist_matrix: np.ndarray, Best_dist: float
) -> tuple[int, int]:
    """
    Retrieves the indices of Boron (B) and Nitrogen (N) atoms from the distance
    matrix corresponding to the shortest B-N distance.

    Args:
        mol: The RDKit molecule object.
        dist_matrix: The 3D distance matrix of the molecule.
        Best_dist: The optimal B-N distance.

    Returns:
        A tuple containing the indices of the Nitrogen and Boron atoms.

    Raises:
        ValueError: If the provided Best_dist does not correspond to a B-N bond.
    """
    BN_indices = np.where(dist_matrix == Best_dist)[0]
    if len(BN_indices) < 2:
        raise ValueError("Could not find two atoms for the given distance.")

    idx1, idx2 = int(BN_indices[0]), int(BN_indices[1])

    atom1 = mol.GetAtomWithIdx(idx1)
    atom2 = mol.GetAtomWithIdx(idx2)

    if atom1.GetAtomicNum() == 7 and atom2.GetAtomicNum() == 5:
        return idx1, idx2
    elif atom1.GetAtomicNum() == 5 and atom2.GetAtomicNum() == 7:
        return idx2, idx1
    else:
        raise ValueError("The provided distance does not correspond to a B-N bond.")


def get_BN_dist(mol: Chem.Mol, dist_matrix: np.ndarray, Threshold: float) -> float:
    """
    Measures the shortest B-N distance that is greater than a given threshold.

    This function identifies all Boron and Nitrogen atoms in the molecule, calculates
    the distances between all possible B-N pairs, and returns the minimum distance
    that is above the specified threshold.

    Args:
        mol: The RDKit molecule object.
        dist_matrix: The distance matrix of the molecule.
        Threshold: The dative B-N bond distance threshold.

    Returns:
        The shortest B-N distance greater than the threshold. If no such
        distance is found, it returns 0.0.
    """
    n_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]
    b_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 5]

    bn_distances = (
        dist_matrix[n_idx, b_idx]
        for n_idx in n_indices
        for b_idx in b_indices
        if dist_matrix[n_idx, b_idx] > Threshold
    )

    return min(bn_distances, default=0.0)


# Embedding function


def embed_mol(smiles: str, n_confs: int = 25) -> Chem.Mol:
    """
    Generates a 3D conformation for a molecule from its SMILES string.

    Args:
        smiles: The SMILES string of the molecule.
        n_confs: The number of conformers to consider.

    Returns:
        An RDKit molecule object with 3D conformers.
    """
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Could not create molecule from SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    mol = cleanup_mol(mol, n_confs)
    return mol


def cleanup_mol(mol: Chem.Mol, n_confs: int = 20) -> Chem.Mol:
    """
    Cleans up an RDKit molecule object by generating conformers and optimizing them.

    This function attempts to use `get_structure_ff` first. If that fails or produces
    invalid conformers, it falls back to RDKit's embedding and optimization methods.

    Args:
        mol: The RDKit molecule object to clean up.
        n_confs: The number of conformers to generate.

    Returns:
        An RDKit molecule object with an optimized conformer.
    """
    try:
        # First, try to get a structure using a force field.
        mol_ff = get_structure_ff(mol, n_confs)
        if mol_ff.GetNumConformers() >= 1 and mol_ff.GetConformer().Is3D():
            return mol_ff
    except Exception as e:
        print(f"get_structure_ff failed with error: {e}")

    # If the force field method fails, use RDKit's embedding.
    params = rdDistGeom.ETKDG()
    params.useRandomCoords = True
    params.ignoreSmoothingFailures = True
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params, maxAttempts=500)

    if not cids:
        raise RuntimeError("Could not generate any conformers for the molecule.")

    # Optimize the conformers and find the one with the lowest energy.
    min_energy = np.inf
    min_energy_index = -1

    if Chem.rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
        results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=200)
        for i, result in enumerate(results):
            if result[1] < min_energy:
                min_energy = result[1]
                min_energy_index = i
    elif Chem.rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
        for i, result in enumerate(results):
            if result[1] < min_energy:
                min_energy = result[1]
                min_energy_index = i

    if min_energy_index != -1:
        # Return the conformer with the lowest energy.
        conf = mol.GetConformer(min_energy_index)
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conf, assignId=True)
        return new_mol
    else:
        # If optimization fails, return the first generated conformer.
        conf = mol.GetConformer(cids[0])
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conf, assignId=True)
        return new_mol


def GS_FLP(
    smiles: str, n_confs: int = 10
) -> tuple[list[int], int, int, np.ndarray, np.ndarray, Chem.Mol] | None:
    """
    Generates initial geometry for a Frustrated Lewis Pair (FLP) from a SMILES string.

    This function embeds the molecule, calculates the distance matrix, identifies
    the B-N bond, and extracts atomic numbers, connectivity matrix, and coordinates.

    Args:
        smiles: The SMILES string of the FLP.
        n_confs: The number of conformers to consider for embedding.

    Returns:
        A tuple containing:
            - z: List of atomic numbers.
            - crd_N: Atomic index of Nitrogen.
            - crd_B: Atomic index of Boron.
            - cm: Connectivity matrix.
            - coords: Coordinates matrix.
            - mol: The RDKit molecule object.
        Returns None if an error occurs or conditions are not met.
    """
    THRESHOLD_BN_DISTANCE = 1.60

    try:
        mol = embed_mol(smiles, n_confs)
    except (ValueError, RuntimeError) as e:
        print(f"Error embedding molecule: {e}")
        return None

    try:
        dist_matrix = Get3DDistanceMatrix(mol)
    except ValueError:
        # If the initial embedding fails, try to clean up the molecule and retry.
        try:
            mol = cleanup_mol(mol, n_confs + 10)
            dist_matrix = Get3DDistanceMatrix(mol)
        except (ValueError, RuntimeError) as e:
            print(f"Error getting distance matrix after cleanup: {e}")
            return None

    bn_distance = get_BN_dist(mol, dist_matrix, THRESHOLD_BN_DISTANCE)

    if bn_distance == 0:
        print("No B-N distance found above the threshold.")
        return None

    cm = Chem.rdmolops.GetAdjacencyMatrix(mol)
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    try:
        crd_N, crd_B = get_BN_index(mol, dist_matrix, bn_distance)
    except ValueError as e:
        print(f"Error getting B-N indices: {e}")
        return None

    coords = mol.GetConformer().GetPositions()

    return z, crd_N, crd_B, cm, coords, mol


def get_vector_from_substituents(
    center_idx: int,
    substituent_indices: list[int],
    current_coords: np.ndarray,
    target_idx: int | None = None,
) -> np.ndarray:
    """
    Calculates a vector from the centroid of substituent groups to the center atom,
    or a plane normal for planar geometries.

    Args:
        center_idx: Index of the central atom.
        substituent_indices: Indices of the substituent atoms.
        current_coords: Current coordinates of all atoms.
        target_idx: Optional index of a target atom to help choose the correct
                    normal for planar cases.

    Returns:
        A numpy array representing the calculated vector.
    """
    B_H_DISTANCE = 1.2  # Defined here for use in this helper, though also in get_H2_pos
    CENTROID_THRESHOLD = 0.1

    points_list = [np.around(current_coords[idx], 4) for idx in substituent_indices]
    if not points_list:
        return np.array([0.0, 0.0, 0.0])

    points = Points(points_list)
    try:
        centroid = points.centroid()
    except (ValueError, ZeroDivisionError):
        return np.array([0.0, 0.0, 0.0])

    # Check if the central atom is in the plane of the substituents.
    if (
        len(substituent_indices) == 3
        and distance.euclidean(centroid, current_coords[center_idx]) < CENTROID_THRESHOLD
    ):
        try:
            plane = Plane.best_fit(points)
            n1 = plane.normal
            n2 = -n1

            # If a target atom is provided, choose the normal that points towards it.
            if target_idx is not None:
                h_candidate1 = n1 * B_H_DISTANCE + current_coords[center_idx]
                h_candidate2 = n2 * B_H_DISTANCE + current_coords[center_idx]
                if distance.euclidean(
                    h_candidate1, current_coords[target_idx]
                ) < distance.euclidean(h_candidate2, current_coords[target_idx]):
                    return n1
                else:
                    return n2
            return n1
        except (ValueError, ZeroDivisionError):
            # If the plane fitting fails, fall back to the centroid method.
            return np.array(current_coords[center_idx]) - centroid
    else:
        # If not planar, return the vector from the centroid to the central atom.
        return np.array(current_coords[center_idx]) - centroid


def get_H2_pos(
    z: list[int],
    crd_N: int,
    crd_B: int,
    cm: np.ndarray,
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Determines the Cartesian coordinates to which hydride and proton are to be added.

    Args:
        z: List of atomic numbers.
        crd_N: Atomic index of Nitrogen.
        crd_B: Atomic index of Boron.
        cm: Connectivity matrix.
        coords: Coordinates matrix.

    Returns:
        A tuple containing two numpy arrays: H_a (coordinate of H(B)) and
        H_b (coordinate of H(N)). Returns None if the positions cannot be determined.
    """
    B_H_DISTANCE = 1.2
    N_H_DISTANCE = 1.0

    # Determine the position of H on the Nitrogen atom.
    subs_base_indices = [n for n, j in enumerate(cm[crd_N, :]) if j == 1]
    if len(subs_base_indices) == 2:
        subs_base_indices.append(crd_N)
        vec_b_unnormalized = get_vector_from_substituents(
            crd_N, subs_base_indices, coords
        )
    elif len(subs_base_indices) == 3:
        vec_b_unnormalized = get_vector_from_substituents(
            crd_N, subs_base_indices, coords, target_idx=crd_B
        )
    else:
        return None

    if not vec_b_unnormalized.any():
        return None

    vec_b = unit_vector(vec_b_unnormalized)
    h_b = vec_b * N_H_DISTANCE + coords[crd_N, :]

    # Determine the position of H on the Boron atom.
    subs_acid_indices = [n for n, j in enumerate(cm[crd_B, :]) if j == 1]
    if len(subs_acid_indices) == 3:
        vec_a_unnormalized = get_vector_from_substituents(
            crd_B, subs_acid_indices, coords, target_idx=crd_N
        )
        if not vec_a_unnormalized.any():
            return None

        n1 = unit_vector(vec_a_unnormalized)
        n2 = -n1

        h_a1 = n1 * B_H_DISTANCE + coords[crd_B, :]
        h_a2 = n2 * B_H_DISTANCE + coords[crd_B, :]

        # Choose the H position that is closer to the other H.
        if distance.euclidean(h_a1, h_b) < distance.euclidean(h_a2, h_b):
            h_a = h_a1
        else:
            h_a = h_a2
    else:
        return None

    return h_a, h_b


def append_H2(
    H_a: np.ndarray, H_b: np.ndarray, mol: Chem.Mol, crd_N: int, crd_B: int
) -> Chem.Mol:
    """
    Appends substrate information (H_a and H_b coordinates) into the FLP mol object,
    and then optimizes the geometry using `get_structure_ff`.

    Args:
        H_a: Coordinates of H(B).
        H_b: Coordinates of H(N).
        mol: The RDKit molecule object of the FLP.
        crd_N: Atomic index of Nitrogen.
        crd_B: Atomic index of Boron.

    Returns:
        An RDKit molecule object with optimized relaxed geometry.
    """
    mol_file = "original.mol"
    Chem.rdmolfiles.MolToMolFile(mol, mol_file)

    h_a_line = f"   {H_a[0]:>7.4f}   {H_a[1]:>7.4f}   {H_a[2]:>7.4f} H   0  0  0  0  0  0  0  0  0  0  0  0\n"
    h_b_line = f"   {H_b[0]:>7.4f}   {H_b[1]:>7.4f}   {H_b[2]:>7.4f} H   0  0  0  0  0  0  0  0  0  0  0  0\n"

    with open(mol_file, "r") as f:
        content = f.readlines()

    # Extract atom and bond counts from the MOL file header.
    try:
        parts = content[3].strip().split()
        n_tot = int(parts[0])
        b_bond = int(parts[1])
    except (ValueError, IndexError):
        raise ValueError("Could not parse atom and bond counts from MOL file.")

    # Prepare new lines for the MOL file.
    hn_line = f"{int(crd_N) + 1:>3}{n_tot + 1:>3}  1  0\n"
    hb_line = f"{int(crd_B) + 1:>3}{n_tot + 2:>3}  1  0\n"

    # Assemble the new MOL file content.
    new_content = content[: n_tot + 4]
    new_content.append(h_a_line)
    new_content.append(h_b_line)
    new_content.extend(content[4 + n_tot : 4 + n_tot + b_bond])
    new_content.append(hn_line)
    new_content.append(hb_line)
    new_content.append(f"M  CHG  1  {int(crd_B) + 1}  -1\n")
    new_content.append(f"M  CHG  1  {int(crd_N) + 1}  +1\n")
    new_content.extend(content[4 + n_tot + b_bond :])

    # Update the atom and bond counts in the header.
    new_atom_count = n_tot + 2
    new_bond_count = b_bond + 2
    new_content[3] = f"{new_atom_count:>3}{new_bond_count:>3}  0  0  0  0  0  0  0  0999 V2000\n"

    # Update the charge of the B and N atoms.
    sui_b_list = list(new_content[4 + crd_B])
    sui_b_list[38] = str(5)
    new_content[4 + crd_B] = "".join(sui_b_list)

    sui_n_list = list(new_content[4 + crd_N])
    sui_n_list[38] = str(3)
    new_content[4 + crd_N] = "".join(sui_n_list)

    # Write the new MOL file and create a molecule object from it.
    flp_h2_mol_file = "FLP_H2.mol"
    with open(flp_h2_mol_file, "w") as f:
        f.writelines(new_content)

    mol_flp_h2 = Chem.rdmolfiles.MolFromMolFile(
        flp_h2_mol_file, removeHs=False, sanitize=False
    )

    # Clean up the generated molecule.
    return cleanup_mol(mol_flp_h2)


def del_substrate(
    mol_origin: str = "original.mol", mol_FLP_H2: str = "FLP_H2.mol"
) -> Chem.Mol:
    """
    Extracts the relaxed FLP molecule from the FLP-H2 structure.

    This function reads the original molecule and the relaxed FLP-H2 molecule,
    and then combines the coordinates of the relaxed FLP with the topology of the
    original molecule to create a new molecule object.

    Args:
        mol_origin: Filename of the original molecule (before H2 addition).
        mol_FLP_H2: Filename of the FLP-H2 molecule.

    Returns:
        An RDKit molecule object of the relaxed FLP.
    """
    with open(mol_origin, "r") as f:
        content_origin = f.readlines()

    # Extract the header and tail of the original MOL file.
    try:
        n_tot = int(content_origin[3].strip().split()[0])
    except (ValueError, IndexError):
        raise ValueError("Could not parse atom count from original MOL file.")

    header = content_origin[:4]
    tail = content_origin[4 + n_tot :]

    with open(mol_FLP_H2, "r") as f:
        content_flp_h2 = f.readlines()

    # Extract the coordinates of the relaxed FLP from the FLP-H2 MOL file.
    content_xyz = content_flp_h2[4 : 4 + n_tot]

    # Combine the header, relaxed coordinates, and tail to create the new MOL file.
    content_flp = header + content_xyz + tail

    flp_relaxed_mol_file = "FLP_relaxed.mol"
    with open(flp_relaxed_mol_file, "w") as f:
        f.writelines(content_flp)

    # Create a molecule object from the new MOL file.
    mol_flp_s = Chem.rdmolfiles.MolFromMolFile(
        flp_relaxed_mol_file, removeHs=False, sanitize=False
    )

    return mol_flp_s


sc_scorer_model = None


def sa_score(smiles: str) -> float:
    """
    Calculates the synthetic accessibility (SA) score for a given SMILES string.

    Args:
        smiles: The SMILES string of the molecule.

    Returns:
        The synthetic accessibility score as a float.
    """
    global sc_scorer_model
    if standalone_model_numpy is None:
        return 0.0

    if sc_scorer_model is None:
        try:
            sc_scorer_model = standalone_model_numpy.SCScorer()
            model_path = SCSCORE_DIR / "models" / "full_reaxys_model_1024bool" / "model.ckpt-10654.as_numpy.json.gz"
            sc_scorer_model.restore(str(model_path))
        except Exception as e:
            print(f"Error loading SCScore model: {e}")
            return 0.0

    try:
        _, score = sc_scorer_model.get_score_from_smi(smiles)
        return float(score)
    except Exception as e:
        print(f"Error calculating SA score: {e}")
        return 0.0


def get_frustration(B_index: int, N_index: int) -> float:
    """
    Predicts the frustration score for a given B-N pair using a pre-trained model.

    Args:
        B_index: The index of the Boron atom.
        N_index: The index of the Nitrogen atom.

    Returns:
        The frustration score as a float, or 0.0 if the model is not loaded or
        prediction is skipped.
    """
    if frustration_predictor is None or not frustration_predictor.model_loaded:
        print("Warning: Frustration prediction is unavailable. Returning 0.0.")
        return 0.0

    xyz_file = "Catalyst_relax_xbtopt.xyz"
    if not os.path.exists(xyz_file):
        raise FileNotFoundError(f"XYZ file not found for frustration prediction: {xyz_file}")

    # start_time = timeit.default_timer()
    try:
        result = frustration_predictor.predict_from_xyz(xyz_file, B_index, N_index)
    except Exception as e:
        raise RuntimeError(f"Frustration prediction failed with error: {e}") from e
    # finally:
    #     execution_time = timeit.default_timer() - start_time
    #     print(f"Frustration prediction executed in {execution_time:.4f} seconds.")

    return result


def xtb_opt(
    xyz: str,
    charge: int = 0,
    unpaired_e: int = 0,
    level: int = 2,
    cycle: int = 500,
    irun: int = 0,
) -> tuple[bool, float | None]:
    """
    Performs geometry optimization using the xtb program.

    Args:
        xyz: Path to the XYZ file for optimization.
        charge: Molecular charge (default: 0).
        unpaired_e: Number of unpaired electrons (default: 0).
        level: Hamiltonian level (0-4) (default: 2).
        cycle: Maximum number of optimization cycles (default: 500).
        irun: Internal run counter for recursive calls (default: 0).

    Returns:
        A tuple containing:
            - bad: A boolean indicating if the optimization failed (True) or succeeded (False).
            - energy: The optimized energy as a float, or None if optimization failed.
    """
    if not os.path.exists(xyz):
        raise FileNotFoundError(f"XYZ file not found for xtb optimization: {xyz}")

    level_map = {
        0: ["--gfnff"],
        1: ["--gfn", "1"],
        2: ["--gfn", "2"],
        3: ["--gfn", "2", "--opt", "loose"],
        4: ["--gfn", "2", "--opt", "sloppy"],
    }

    if level not in level_map:
        raise ValueError(f"Invalid optimization level: {level}")

    execution = ["xtb"] + level_map[level] + [xyz, "--opt", "--cycles", str(cycle)]

    if charge != 0:
        execution.extend(["--charge", str(charge)])
    if unpaired_e != 0:
        execution.extend(["--uhf", str(unpaired_e)])

    try:
        result = run(
            execution, stdout=PIPE, stderr=PIPE, universal_newlines=True, check=True, timeout=30
        )
    except sp.CalledProcessError as e:
        print(f"xtb command failed with error: {e.stderr}")
        if irun < 2:
            # If optimization fails, try a lower level of theory.
            return xtb_opt(xyz, charge, unpaired_e, level + 1, cycle, irun + 1)
        else:
            raise RuntimeError(
                f"xtb optimization failed after multiple attempts:\n{e.stderr}"
            ) from e
    except sp.TimeoutExpired as e:
        print(f"xtb command timed out: {e.stdout} {e.stderr}")
        if irun < 2:
            # If optimization fails, try a lower level of theory.
            return xtb_opt(xyz, charge, unpaired_e, level + 1, cycle, irun + 1)
        else:
            raise RuntimeError(
                f"xtb optimization timed out after multiple attempts:\n{e.stdout} {e.stderr}"
            ) from e

    # The output of xtb is expected to contain a line with the total energy.
    match = re.search(r"TOTAL ENERGY\s+([-+]?\d*\.?\d+)", result.stdout)
    if match:
        energy = float(match.group(1))
        name = xyz[:-4]
        with open(f"{name}_xtb.out", "w") as f:
            f.write(result.stdout)
        with open(f"{name}_energy.out", "w") as f:
            f.write(str(energy))

        # Rename the output file to avoid overwriting.
        try:
            os.rename("xtbopt.xyz", f"{name}_xbtopt.xyz")
        except FileNotFoundError:
            os.rename("xtblast.xyz", f"{name}_xbtopt.xyz")

        return False, energy
    else:
        print(f"Could not parse energy from xtb output. Stdout: {result.stdout}, Stderr: {result.stderr}")
        if irun < 2:
            # If optimization fails, try a lower level of theory.
            return xtb_opt(xyz, charge, unpaired_e, level + 1, cycle, irun + 1)
        else:
            raise ValueError("Could not parse energy from xtb output.")


def gen_relaxed_FLP_H2_xtb(
    smiles: str, level: int = 2
) -> tuple[Chem.Mol | None, float, float, bool]:
    """
    Generates an RDKit mol object of a relaxed FLP-H2 complex optimized using xtb.

    Args:
        smiles: SMILES string of the FLP.
        level: Optimization level for xtb (default: 2).

    Returns:
        A tuple containing:
            - mol_FLP_s: RDKit molecule object of the relaxed FLP, or None if an error occurs.
            - energy_H2: Energy of the FLP-H2 complex.
            - energy_n: Energy of the neutral FLP.
            - bad: A boolean indicating if an error occurred during the process.
    """
    gs_flp_result = GS_FLP(smiles)
    if not gs_flp_result:
        return None, 0.0, 0.0, True

    z, crd_N, crd_B, cm, coords, mol = gs_flp_result

    h2_pos_result = get_H2_pos(z, crd_N, crd_B, cm, coords)
    if not h2_pos_result:
        return None, 0.0, 0.0, True

    h_a, h_b = h2_pos_result

    # Create the XYZ file for the FLP-H2 complex.
    AllChem.MolToXYZFile(mol, "Catalyst.xyz")
    with open("Catalyst.xyz", "r+") as f:
        lines = f.readlines()
        lines[0] = f"{int(lines[0]) + 2}\n"
        lines.append(f"H  {h_a[0]}   {h_a[1]}   {h_a[2]}\n")
        lines.append(f"H  {h_b[0]}   {h_b[1]}   {h_b[2]}\n")
        f.seek(0)
        f.writelines(lines)

    # Optimize the FLP-H2 complex.
    try:
        bad, energy_H2 = xtb_opt("Catalyst.xyz", level=level)
        if bad:
            return None, 0.0, 0.0, True
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error optimizing FLP-H2 complex: {e}")
        return None, 0.0, 0.0, True

    # Create the XYZ file for the relaxed FLP.
    with open("Catalyst_xbtopt.xyz", "r") as f:
        lines = f.readlines()
    with open("Catalyst_relax.xyz", "w") as f:
        f.write(f"{int(lines[0]) - 2}\n\n")
        f.writelines(lines[2:-2])

    # Optimize the relaxed FLP.
    try:
        bad, energy_n = xtb_opt("Catalyst_relax.xyz", level=level)
        if bad:
            return None, 0.0, 0.0, True
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error optimizing relaxed FLP: {e}")
        return None, 0.0, 0.0, True

    # Create the molecule object for the relaxed FLP.
    sp.call(["xtb", "Catalyst_relax_xbtopt.xyz"], stdout=sp.DEVNULL, stderr=sp.STDOUT)
    mol_flp_s = Chem.rdmolfiles.MolFromMolFile(
        "xtbtopo.mol", removeHs=False, sanitize=False
    )

    return mol_flp_s, energy_H2, energy_n, False


def calc_paha(energy_n: float) -> tuple[float, float]:
    """
    Calculates the proton affinity (PA) and hydride affinity (HA) values.

    Args:
        energy_n: The energy of the neutral FLP.

    Returns:
        A tuple containing the HA and PA values.
    """
    # Calculate Hydride Affinity (HA)
    try:
        with open("Catalyst_relax_xbtopt.xyz", "r") as f:
            lines = f.readlines()
        with open("Catalyst_FLP_BH.xyz", "w") as f:
            f.write(f"{int(lines[0]) - 1}\n\n")
            f.writelines(lines[2:-1])

        bad, energy_h = xtb_opt("Catalyst_FLP_BH.xyz", charge=-1, level=2)
        if bad:
            raise RuntimeError("xtb optimization failed for BH complex.")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"Error calculating HA: {e}") from e

    ha = (energy_h - energy_n - (-0.610746694539)) * 627.509

    # Calculate Proton Affinity (PA)
    try:
        with open("Catalyst_relax_xbtopt.xyz", "r") as f:
            lines = f.readlines()
        with open("Catalyst_FLP_NH.xyz", "w") as f:
            f.write(f"{int(lines[0]) - 1}\n\n")
            f.writelines(lines[2:-2] + [lines[-1]])

        bad, energy_p = xtb_opt("Catalyst_FLP_NH.xyz", charge=1, level=2)
        if bad:
            raise RuntimeError("xtb optimization failed for NH complex.")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"Error calculating PA: {e}") from e

    pa = (energy_p - energy_n) * 627.509

    return ha, pa


def chem_fit(point: list[float], line_coeffs: list[float]) -> float:
    """
    Calculates the perpendicular distance from a point to a line.

    The line is defined by the equation y = mx + c, where m is the slope and c is
    the y-intercept.

    Args:
        point: A list of two floats [x, y] representing the point.
        line_coeffs: A list of two floats [m, c] representing the coefficients
                     of the line (y = mx + c).

    Returns:
        The perpendicular distance from the point to the line.
    """
    m, c = line_coeffs
    x, y = point

    # The distance is calculated using the formula for the distance from a point
    # to a line: |ax + by + c| / sqrt(a^2 + b^2)
    # In our case, the line is y = mx + c, which can be written as mx - y + c = 0.
    # So, a = m, b = -1, and c = c.
    return abs(m * x - y + c) / math.sqrt(m**2 + 1)


def smiles_2_mol(smiles: str, level: int = 2) -> Chem.Mol:
    """
    Generates an xtb-optimized RDKit molecule object from a SMILES string.

    Args:
        smiles: The SMILES string of the molecule.
        level: Optimization level for xtb (default: 2).

    Returns:
        An RDKit molecule object with optimized geometry.
    """
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Could not create molecule from SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    mol = get_structure_ff(mol, 20)

    xyz_file = "Catalyst.xyz"
    AllChem.MolToXYZFile(mol, xyz_file)

    try:
        bad, _ = xtb_opt(xyz_file, level=level)
        if bad:
            raise RuntimeError("xtb optimization failed.")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"Error during xtb optimization: {e}") from e

    sp.call(["xtb", "Catalyst_xbtopt.xyz"], stdout=sp.DEVNULL, stderr=sp.STDOUT)

    mol_xtb = Chem.rdmolfiles.MolFromMolFile(
        "xtbtopo.mol", removeHs=False, sanitize=False
    )
    if not mol_xtb:
        raise ValueError("Could not create molecule from xtbtopo.mol")

    return mol_xtb