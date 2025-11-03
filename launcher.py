#!/usr/bin/env python3
import os
import sys

import numpy as np
import pandas as pd
import psutil
from chimera import Chimera
from navicatGA.chemistry_smiles import sanitize_smiles
from navicatGA.score_modifiers import score_modifier
from navicatGA.smiles_solver import SmilesGenAlgSolver
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Get3DDistanceMatrix, MolFromSmiles

from ga_flp import chromosome_to_smiles, fitness_wrt_target

# For multiobjective optimization we use chimera to scalarize
chimera = Chimera(tolerances=[0.25, 0.1, 0.25], goals=["max", "max", "min"])


# Read the excel/csv file with the SMILES strings.
def read_database(filename) -> pd.DataFrame:
    fdb = pd.read_excel(filename)
    fdb.dropna()
    return fdb


fdb = read_database(sys.argv[1])

# Get the alphabets for each gene.
alphabet_gene01 = list(fdb.LAr.dropna().values)
alphabet_gene23 = list(fdb.LBr.dropna().values)
alphabet_gene4 = list(fdb.BB.dropna().values)
alphabet_gene567 = list(fdb.BBr.dropna().values)


# This is the alphabet list. It has the same shape as a chromosome.
alphabet_list = [
    alphabet_gene01,
    alphabet_gene01,
    alphabet_gene23,
    alphabet_gene23,
    alphabet_gene4,
    alphabet_gene567,
    alphabet_gene567,
    alphabet_gene567,
]


if __name__ == "__main__":
    # This is the SMILES solver.
    solver = SmilesGenAlgSolver(
        n_genes=8,
        pop_size=50,
        mutation_rate=0.25,
        n_crossover_points=1,
        # starting_population=[]
        # selection_rate=0.25,
        # max_gen=
        # max_conv=
        fitness_function=fitness_wrt_target(),
        chromosome_to_smiles=chromosome_to_smiles(),
        scalarizer=chimera,
        alphabet_list=alphabet_list,
        starting_random=True,
        logger_level="INFO",
        verbose=True,
        lru_cache=True,
        prune_duplicates=True,
        to_stdout=True,
        show_stats=True,
        plot_results=True,
        # logger_file= "ga_flp.log"
    )
    # random_state=42,

    # Perform the runs untill convergence[mean_fitness = 0.9]
    #    mean_fitness = 0; maximum_fitness = 0; n = 0
    #   while mean_fitness < 0.925:
    #      solver.solve(2)  # Do some iterations.
    #     mean_fitness = solver.mean_fitness_[n]
    #    maximum_fitness = solver.max_fitness_[n]
    #   print(f"Mean fitness of cycle number {n}: {mean_fitness} ")
    #  n+=1
    # print("Total generations: ",n)
    for i in range(10):
        print("CYCLE ", i)
        solver.solve(1)
        print(
            f"{solver.fitness_} is the current fitness. The next solve call should reuse it."
        )
        solver.write_population()
        for j, k, chromosome in zip(
            solver.fitness_, solver.printable_fitness, solver.population_
        ):
            smiles = chromosome_to_smiles()(chromosome)
            with open("fitness_{0}.txt".format(i), "a") as f:
                print(f"{smiles},{j},{k[0]},{k[1]},{k[2]}", file=f)

        os.system("mkdir gen{0} ; mv chromosome*.* gen{0}".format(i))
        np.savetxt("mean_fitness_{0}.txt".format(i), solver.mean_fitness_)
        np.savetxt("max_fitness_{0}.txt".format(i), solver.max_fitness_)
        np.savetxt("p_fitness_{0}.txt".format(i), solver.printable_fitness)

        # print(smiles)
    np.savetxt("mean_fitness.txt", solver.mean_fitness_)
    np.savetxt("max_fitness.txt", solver.max_fitness_)
    np.savetxt("p_fitness.txt", solver.printable_fitness)
    print(f"Total runtime: {solver.runtime_} second")

print("GA run terminated normally!")
print(
    "Memory used: {} MB.".format(
        psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    )
)
