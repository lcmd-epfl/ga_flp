"""
This module provides functions for the genetic algorithm launcher, including
SMILES string generation from chromosomes and fitness evaluation of molecules.
"""

from navicatGA.score_modifiers import score_modifier
from rdkit.Chem import Get3DDistanceMatrix, MolFromSmiles
from rdkit import Chem

from .angle_measure import calculate_angle
from .ga_flp_fs import (
    calc_paha,
    chem_fit,
    gen_relaxed_FLP_H2_xtb,
    get_BN_dist,
    get_frustration,
    sa_score,
    get_BN_index
)


def external_FLP(smiles_list: list[str], BB1: str, LBr1: str, LBr2: str) -> list[str]:
    """
    Appends external Frustrated Lewis Pair (FLP) components to a list of SMILES fragments
    based on specific structural rules. This function handles cases where the Lewis base
    fragments (LBr1, LBr2) might contain a wildcard '*' indicating a specific bonding
    pattern, or appends standard nitrogen-based Lewis base components.

    Args:
        smiles_list: A list of SMILES fragments to which the external FLP components
                     will be appended. This list is modified in-place.
        BB1: The SMILES string for the main building block.
        LBr1: The SMILES string for the first Lewis base fragment.
        LBr2: The SMILES string for the second Lewis base fragment.

    Returns:
        The updated list of SMILES fragments, including the appended external FLP components.
    """
    smiles_list.append(BB1)
    LBr = [LBr1, LBr2]
    if any("*" in LB for LB in LBr):
        lb = next(filter(lambda LB: "*" in LB, LBr))
        if "c1" in lb:
            smiles_list.append("n1")
            smiles_list.append(f"({lb[1:]})")
        else:
            smiles_list.append("N1")
            smiles_list.append(f"({lb[1:]})")
    else:
        smiles_list.append("N")
        smiles_list.extend([f"({LBr1})", f"({LBr2})"])
    return smiles_list


def chromosome_to_smiles(chromosome: list[str]) -> str:
    """
    Generates a full SMILES string representing a molecule from a list of
    SMILES fragments (chromosome). The chromosome is expected to contain
    specific building blocks for Lewis acids (LAr), Lewis bases (LBr),
    and a central building block (BB) with potential substituents (BBr).

    The function handles different bonding patterns, including those with
    wildcard '*' characters indicating specific attachment points, and
    constructs the final SMILES string by concatenating these fragments
    according to predefined rules.

    Args:
        chromosome: A list of SMILES fragments, where each element corresponds
                    to a specific part of the molecule:
                    - chromosome[0]: LAr1 (Lewis Acid fragment 1)
                    - chromosome[1]: LAr2 (Lewis Acid fragment 2)
                    - chromosome[2]: LBr1 (Lewis Base fragment 1)
                    - chromosome[3]: LBr2 (Lewis Base fragment 2)
                    - chromosome[4]: BB1 (Central Building Block)
                    - chromosome[5]: BBrA (Substituent for BB, position A)
                    - chromosome[6]: BBrB (Substituent for BB, position B)
                    - chromosome[7]: BBrC (Substituent for BB, position C)

    Returns:
        A single SMILES string assembled from the chromosome fragments.
    """
    LAr1 = chromosome[0]
    LAr2 = chromosome[1]
    LBr1 = chromosome[2]
    LBr2 = chromosome[3]
    BB1 = chromosome[4]
    BBrA = chromosome[5]
    BBrB = chromosome[6]
    BBrC = chromosome[7]

    smiles_list: list[str] = []
    LAr = [LAr1, LAr2]
    # addressing Catalyst-A-LA
    if any("*" in LA for LA in LAr):
        la = next(filter(lambda LA: "*" in LA, LAr))
        smiles_list.append("B1")
        smiles_list.append(f"({la[1:]})")
    else:
        smiles_list.append("B")
        smiles_list.extend([f"({LAr1})", f"({LAr2})"])

    # adding substituent groups to BB
    if "()" in BB1:
        pieces = BB1.split("()")
        BB1_list = []
        for i, piece in enumerate(pieces[:-1]):
            BB1_list.append(piece)
            if i == 0:
                BB1_list.append(f"({BBrA})")
            elif i == len(pieces[:-1]) - 1:
                BB1_list.append(f"({BBrB})")
            else:
                BB1_list.append(f"({BBrC})")
        BB1_list.append(pieces[-1])
        BB1_str = "".join(BB1_list)

        # LBr2 must be null here, as well as LBr1 if it is *.
        # 2B and 2A
        if "*" in BB1_str:
            smiles_list.append(BB1_str[1:])
            smiles_list.append("N1")
            if not ("*" in LBr1) and (BB1_str[-1] != "="):
                smiles_list.append(f"({LBr1})")
        # 2D
        elif "@" in BB1_str and "$" in BB1_str:
            smiles_list.append(BB1_str[2:])
            smiles_list.append("n1")
        # 2C
        elif "@" in BB1_str:
            smiles_list.append(BB1_str[1:])
            smiles_list.append("N12")

        # External FLP
        else:
            smiles_list = external_FLP(smiles_list, BB1_str, LBr1, LBr2)

    else:
        smiles_list = external_FLP(smiles_list, BB1, LBr1, LBr2)

    attempt_smiles = "".join(smiles_list)

    return attempt_smiles



# The optimal target values and threshold for the B-N covalent bond distance.
target_value_F1 = 3.0
target_value_F2 = 100
target_value_F3 = 0.0
Threshold = 1.4


def fitness_function_12_v2(
    smiles: str, Threshold: float = 1.4, level: int = 2
) -> tuple[float, float, float, float, float, float, int]:
    """
    Measures key properties in INT2, generated with RDKit and optimized with the xtb module.

    Args:
        smiles: SMILES string of the IFLP (Frustrated Lewis Pair).
        Threshold: Dative bond distance threshold.
        level: Optimization level (default = 2).

    Returns:
        A tuple containing:
            - dchem: Chemical distance.
            - bn_distance: Boron-nitrogen distance.
            - angle: Angle between relevant vectors.
            - fepa: FEPA value.
            - feha: FEHA value.
            - sa: Synthetic accessibility score.
            - fr_class: Frustration class (integer).
    """

    sa = sa_score(smiles)
    mol = MolFromSmiles(smiles)
    mol_FLP_s, e_i2, e_n, bad = gen_relaxed_FLP_H2_xtb(smiles, level=2)
    if bad:
        return 1e5, 0.0, 0.0, 0.0, 0.0, 1e5, 0
    ha, pa = calc_paha(e_n)
    feha = 0.67 * ha - 25.69
    fepa = 1.04 * pa - 163.01
    line = [-1.0, -317.18]
    point = [fepa, feha]
    dchem = chem_fit(point, line)
    try:
        dist_matrix = Get3DDistanceMatrix(mol_FLP_s)
    except ValueError:
        return 1e5, 0.0, 0.0, 0.0, 0.0, 1e5, 0

    bn_distance = get_BN_dist(mol_FLP_s, dist_matrix, Threshold)

    cm = Chem.rdmolops.GetAdjacencyMatrix(mol_FLP_s)

    z = [atom.GetAtomicNum() for atom in mol_FLP_s.GetAtoms()]

    crd_N, crd_B = get_BN_index(mol_FLP_s, dist_matrix, bn_distance)
    i = 0
    for atom in mol_FLP_s.GetAtoms():
        if i == crd_N and (atom.GetAtomicNum() == 7):
            break
        elif i == crd_N and (atom.GetAtomicNum() == 5):
            crd_N, crd_B = crd_B, crd_N
            break
        i += 1
    fr_class = get_frustration(crd_N + 1, crd_B + 1)
    for c in mol_FLP_s.GetConformers():
        coords = c.GetPositions()

    angle = calculate_angle(z, crd_N, crd_B, cm, coords)

    if isinstance(fr_class, float):
        return dchem, bn_distance, angle, fepa, feha, sa, fr_class
    else:
        return dchem, bn_distance, angle, fepa, feha, sa, fr_class[0]


def overall_fitness_function(smiles: str) -> tuple[float, float, float]:
    """
    Calculates the overall fitness of a SMILES string based on chemical distance,
    geometric scores, and synthetic accessibility.

    Args:
        smiles: The SMILES string of the molecule.

    Returns:
        A tuple containing the chemical score, geometric score, and synthetic
        accessibility score.
    """
    dchem, bn_distance, angle, fepa, feha, scs, fr = fitness_function_12_v2(smiles)

    score1 = score_modifier(bn_distance, target_value_F1, 1, 0.6) * fr
    score2 = score_modifier(angle, target_value_F2, 1, 0.75) * fr

    score_geom = (score1 + score2) / 2

    eps = 1e-7
    score3 = (1 / (dchem + eps)) * fr

    with open("raw_data.txt", "a") as f:
        print(
            f"{smiles},{dchem},{score3},{bn_distance},{angle},{fepa},{feha},{scs},{fr}",
            file=f,
        )

    return score3, score_geom, scs
