#!/usr/bin/env python
# Operation Bluerose, 88th
# Functions for the laucher

import numpy as np
from navicatGA.score_modifiers import score_modifier
from rdkit import Chem
from rdkit.Chem import AllChem, Get3DDistanceMatrix, MolFromSmiles

from angle_measure import angle_between, calculate_angle
from ga_flp_fs import (
    GS_FLP,
    append_H2,
    calc_paha,
    chem_fit,
    cleanup_mol,
    del_substrate,
    gen_relaxed_FLP_H2_xtb,
    get_BN_dist,
    get_frustration,
    get_H2_pos,
    sa_score,
)

# import standalone_model_numpy
# from standalone_model_numpy import SCScorer


# This is the assembler function we will use.
def chromosome_to_smiles():
    def sc2smiles(chromosome):
        """Generate a smiles string from a list of SMILES fragments."""
        LAr1 = chromosome[0]
        LAr2 = chromosome[1]
        LBr1 = chromosome[2]
        LBr2 = chromosome[3]
        BB1 = chromosome[4]
        BBrA = chromosome[5]
        BBrB = chromosome[6]
        BBrC = chromosome[7]

        def external_FLP(smiles_list, BB1, LBr1, LBr2):
            smiles_list.append(BB1)
            LBr = [LBr1, LBr2]
            if any("*" in LB for LB in LBr):
                lb = filter(lambda LB: "*" in LB in LBr, LBr)
                lb = list(lb)[0]
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

        smiles_list = []
        LAr = [LAr1, LAr2]
        # addressing Catalyst-A-LA
        if any("*" in LA for LA in LAr):
            la = filter(lambda LA: "*" in LA in LAr, LAr)
            la = list(la)[0]
            smiles_list.append("B1")
            smiles_list.append(f"({la[1:]})")
        else:
            smiles_list.append("B")
            smiles_list.extend([f"({LAr1})", f"({LAr2})"])

        # adding substituent groups to BB
        if "()" in BB1:
            pieces = BB1.split("()")
            BB1 = []
            for i, piece in enumerate(pieces[:-1]):
                BB1.append(piece)
                if i == 0:
                    BB1.append(f"({BBrA})")
                elif i == len(pieces[:-1]) - 1:
                    BB1.append(f"({BBrB})")
                else:
                    BB1.append(f"({BBrC})")
            BB1.append(pieces[-1])
            BB1_str = "".join(BB1)

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
        # print("Input SMILES: ", attempt_smiles)

        return attempt_smiles
        # Recommended sanitization
        # mol, smiles, ok = sanitize_smiles(attempt_smiles)

    # if ok and mol is not None:
    #     return smiles
    # else:
    #     print(f"Generated SMILES string {attempt_smiles} could not be sanitized.")
    #     raise ValueError

    return sc2smiles


# The optimal target values and threshold for the B-N covalent bond distance.
target_value_F1 = 3.0
target_value_F2 = 100
target_value_F3 = 0.0
Threshold = 1.4

# F12: d(B-N) and the angle; use xtb


def fitness_function_12_v2(smiles, Threshold=1.4, level=2):
    """
    measure the 2 key properties in INT2 which is generated with RDkit and optimized with xtb module
    Parameters:
    1. smiles: SMILES of IFLP
    2. Threshold: dative bond distance
    3. level: default = 0

    Returns:
    bn_distance, angle
    """

    ####SCSCORE
    sa = sa_score(smiles)
    # print(sa)
    #####
    mol = MolFromSmiles(smiles)
    mol_FLP_s, e_i2, e_n, bad = gen_relaxed_FLP_H2_xtb(smiles, level=2)
    if bad:
        # print("Bad returned by gen_relaxed_FLP_H2_xtb. Punishing fitness.")
        return 1e5, 0, 0, 0, 0, 1e5, 0
    ha, pa = calc_paha(e_n)
    feha = 0.67 * ha - 25.69
    fepa = 1.04 * pa - 163.01
    # e_rrs_i2 = (e_i2 - e_n)*627.509
    # print("FEHA=", feha, "FEPA=", fepa)
    line = [-1.0, -317.18]
    point = [fepa, feha]
    dchem = chem_fit(point, line)
    try:
        dist_matrix = Get3DDistanceMatrix(mol_FLP_s)
    except ValueError:
        # print("Bad distance matrix generation. Punishing fitness.")
        return 1e5, 0, 0, 0, 0, 1e5, 0

    # F1, d(B-N)
    bn_distance = get_BN_dist(mol_FLP_s, dist_matrix, Threshold)
    # F2; angle

    # A. Connectivity matrix(or Adjacency matrix)
    cm = Chem.rdmolops.GetAdjacencyMatrix(mol_FLP_s)

    # B. list of atomic number
    z = []
    for atom in mol_FLP_s.GetAtoms():
        z.append(atom.GetAtomicNum())

    # C. atomic indices of N and B, keep in mind the indices may be swapped.
    def get_BN_index(dist_matrix, Best_dist):
        BN = np.where(dist_matrix == Best_dist)[1]
        # print(BN)
        crd_N = BN[0]
        crd_B = BN[1]
        return crd_N, crd_B

    crd_N, crd_B = get_BN_index(dist_matrix, bn_distance)
    # swapping the indices if incorrect
    i = 0
    for atom in mol_FLP_s.GetAtoms():
        if i == crd_N and (atom.GetAtomicNum() == 7):
            # print("Correct index assignment")
            break
        elif i == crd_N and (atom.GetAtomicNum() == 5):
            crd_N, crd_B = crd_B, crd_N
            break
        i += 1
    fr_class = get_frustration(crd_N + 1, crd_B + 1)
    # print("The frustration class is", fr_class)
    # D. Coordinate matrix
    for c in mol_FLP_s.GetConformers():
        coords = c.GetPositions()

    # With A-D, calculate the angle
    angle = calculate_angle(z, crd_N, crd_B, cm, coords)

    # print(
    #    "The B-N distance = {:0.2f} angstrom, Angle={:0.2f} and dchem={:0.2f} ".format(
    #        bn_distance, angle, dchem
    #    )
    # )
    # dchem = fr_class * dchem
    # bn_distance = fr_class * bn_dista

    return dchem, bn_distance, angle, fepa, feha, sa, fr_class[0]


# F12: d(B-N) and the angle


def fitness_function_12(smiles, Threshold=Threshold):
    """
    Input: smiles

    utilizing Embedding_function and get_BN_dist to generate mol object and the distance between LA and LB, respectively

    Return: bond distance between Lewis Acid center and Lewis Base center, and the angle
    """
    # print(f"Input SMILES : {smiles}")

    z, crd_N, crd_B, cm, coords, mol = GS_FLP(smiles)
    if z == 0:
        return 0, 0

    H_a, H_b = get_H2_pos(z, crd_N, crd_B, cm, coords)
    H_a = np.array(H_a)
    H_b = np.array(H_b)
    if not H_a.any() and not H_b.any():
        return 0, 0

    mol_FLP_H2 = append_H2(H_a, H_b, mol, crd_N, crd_B)

    # Catch the most "reasonable" confs
    d = 5
    theta = 180
    m = 1
    while d > 3.5 and theta > 145:
        if m > 50:
            # print("Exceeded 50 attempts")
            break

        try:
            DM = Get3DDistanceMatrix(mol_FLP_H2)
        except ValueError:
            # print(
            #    "dist_mat cannot be generated due to bad conformer, Setting both angle and distance to be zero"
            # )
            return 0, 0

        for c in mol_FLP_H2.GetConformers():
            coords_FLPs = c.GetPositions()

        HB = coords_FLPs[-2] - coords_FLPs[crd_B, :]
        HN = coords_FLPs[-1] - coords_FLPs[crd_N, :]
        theta = angle_between(HB, HN) * 180 / np.pi

        d = DM[len(z), len(z) + 1]

        if d > 3.5 and theta > 145:
            # print("Bad positions, the correction")
            mol_FLP_H2 = cleanup_mol(mol_FLP_H2, n_confs=2)
        else:
            pass
            # print("All is good")
        # print(f"d(H-H) = {d}; angle between HB and HN = {theta}")
        m += 1

    mol_FLP_s = del_substrate(mol_origin="original.mol", mol_FLP_H2="FLP_H2.mol")
    dist_matrix = Get3DDistanceMatrix(mol_FLP_s)

    # F1, d(B-N)
    bn_distance = get_BN_dist(mol_FLP_s, dist_matrix, Threshold)
    # F2; angle

    # A. Connectivity matrix(or Adjacency matrix)
    cm = Chem.rdmolops.GetAdjacencyMatrix(mol_FLP_s)

    # B. list of atomic number
    z = []
    for atom in mol_FLP_s.GetAtoms():
        z.append(atom.GetAtomicNum())

    # C. atomic indices of N and B, keep in mind the indices may be swapped.
    def get_BN_index(dist_matrix, Best_dist):
        BN = np.where(dist_matrix == Best_dist)[1]
        crd_N = BN[0]
        crd_B = BN[1]
        return crd_N, crd_B

    crd_N, crd_B = get_BN_index(dist_matrix, bn_distance)
    # swapping the indices if incorrect
    i = 0
    for atom in mol.GetAtoms():
        if i == crd_N and (atom.GetAtomicNum() == 7):
            # print("Correct index assignment")
            break
        elif i == crd_N and (atom.GetAtomicNum() == 5):
            crd_N, crd_B = crd_B, crd_N
            break
        i += 1

    # D. Coordinate matrix
    for c in mol_FLP_s.GetConformers():
        coords = c.GetPositions()

    # With A-D, calculate the angle
    angle = calculate_angle(z, crd_N, crd_B, cm, coords)

    # print(
    #    "The B-N distance = {:0.2f} angstrom, Angle={:0.2f} and dchem={:0.2f} ".format(
    #        bn_distance, angle, dchem
    #    )
    # )
    return bn_distance, angle


# This is a wrapper to apply a gaussian scalarizer to the fitness functions.
def fitness_wrt_target():
    def overall_fitnes_function(smiles):
        dchem, bn_distance, angle, fepa, feha, scs, fr = fitness_function_12_v2(smiles)
        # score = (1 * score_modifier(bn_distance, target_value_F1, 1, 0.6) +
        # 1 * score_modifier(angle, target_value_F2, 1, 0.75) + 1 * score_modifier(dchem, target_value_F3, 1, 0.6)) / 3
        # score = score_modifier(dchem, target_value_F3, 2, 0.5)
        score1 = score_modifier(bn_distance, target_value_F1, 1, 0.6)
        score1 = score1 * fr
        score2 = score_modifier(angle, target_value_F2, 1, 0.75)
        score2 = score2 * fr
        score_geom = (
            1 * score_modifier(bn_distance, target_value_F1, 1, 0.6)
            + 1 * score_modifier(angle, target_value_F2, 1, 0.75)
        ) / 2
        score_geom = score_geom * fr
        # score = score_modifier(dchem, target_value_F3, 2, 0.5)
        # score3 = score_modifier(dchem, target_value_F3, 1, 30)
        # print("Fitness value: ", score1,score2)#,score3)
        # print("Merit = ",chimera.scalarize())
        # list = [dchem, score1, score2, fepa, feha]
        eps = 1e-7
        score3 = 1 / (dchem + eps)
        score3 = score3 * fr
        with open("raw_data.txt", "a") as f:
            print(
                f"{smiles},{dchem},{score3},{bn_distance},{angle},{fepa},{feha},{scs},{fr}",
                file=f,
            )

        return score3, score_geom, scs

    return overall_fitnes_function
