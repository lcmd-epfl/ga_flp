#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from chimera import Chimera
from navicatGA.smiles_solver import SmilesGenAlgSolver
from rdkit import Chem
from rdkit.Chem import AllChem

from ga_core.ga_flp import chromosome_to_smiles, overall_fitness_function

# For multiobjective optimization we use chimera to scalarize
chimera = Chimera(tolerances=[0.25, 0.1, 0.25], goals=["max", "max", "min"])

# --- Constants for GA parameters ---
N_GENES = 8
POP_SIZE = 50
MUTATION_RATE = 0.25
N_CROSSOVER_POINTS = 1
NUM_CYCLES = 10 # Number of GA cycles to run

# Read the excel/csv file with the SMILES strings.
def read_database(filename: str) -> pd.DataFrame:
    """
    Reads a database from an Excel file into a pandas DataFrame.

    Args:
        filename: The path to the Excel file.

    Returns:
        A pandas DataFrame containing the data from the Excel file, with
        any rows containing NaN values dropped.
    """
    database_df = pd.read_excel(filename)
    database_df.dropna(inplace=True)
    return database_df

def setup_ga_parameters(database_filename: str) -> list:
    """
    Sets up the genetic algorithm parameters by reading the database and extracting alphabets for each gene.

    Args:
        database_filename: The path to the Excel file containing the database.

    Returns:
        A list of alphabets, where each alphabet corresponds to a gene in the chromosome.
    """
    database_df = read_database(database_filename)

    # Get the alphabets for each gene.
    alphabet_gene01 = list(database_df.LAr.dropna().values)
    alphabet_gene23 = list(database_df.LBr.dropna().values)
    alphabet_gene4 = list(database_df.BB.dropna().values)
    alphabet_gene567 = list(database_df.BBr.dropna().values)

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
    return alphabet_list


def main():
    """
    Main function to run the genetic algorithm for optimizing SMILES strings.
    It sets up the GA solver, runs the optimization cycles, and saves the results.
    """
    alphabet_list = setup_ga_parameters(sys.argv[1])
    # Instantiate callable functions once
    chromosome_to_smiles_callable = chromosome_to_smiles
    fitness_function_callable = overall_fitness_function

    # This is the SMILES solver.
    solver = SmilesGenAlgSolver(
        n_genes=N_GENES,
        pop_size=POP_SIZE,
        mutation_rate=MUTATION_RATE,
        n_crossover_points=N_CROSSOVER_POINTS,
        # starting_population=[]
        # selection_rate=0.25,
        # max_gen=
        # max_conv=
        fitness_function=fitness_function_callable,
        chromosome_to_smiles=chromosome_to_smiles_callable,
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
    for i in range(NUM_CYCLES):
        print("CYCLE ", i)
        solver.solve(1)
        print(
            f"{solver.fitness_} is the current fitness. The next solve call should reuse it."
        )
        solver.write_population()
        for j, k, chromosome in zip(
            solver.fitness_, solver.printable_fitness, solver.population_
        ):
            smiles = chromosome_to_smiles_callable(chromosome)
            with open(f"fitness_{i}.txt", "a") as f:
                print(f"{smiles},{j},{k[0]},{k[1]},{k[2]}", file=f)

        # Create directory and move chromosome files
        output_dir = Path(f"gen{i}")
        output_dir.mkdir(exist_ok=True)
        for f in Path(".").glob("chromosome*.*"):
            shutil.move(f, output_dir / f.name)

        np.savetxt(f"mean_fitness_{i}.txt", solver.mean_fitness_)
        np.savetxt(f"max_fitness_{i}.txt", solver.max_fitness_)
        np.savetxt(f"p_fitness_{i}.txt", solver.printable_fitness)

        # print(smiles)
    np.savetxt("mean_fitness.txt", solver.mean_fitness_)
    np.savetxt("max_fitness.txt", solver.max_fitness_)
    np.savetxt("p_fitness.txt", solver.printable_fitness)

    # Concatenate fitness files from each cycle
    with open("fitness.txt", "w") as outfile:
        for i in range(NUM_CYCLES):
            filepath = Path(f"fitness_{i}.txt")
            if filepath.exists():
                with open(filepath, "r") as infile:
                    outfile.write(infile.read())
                filepath.unlink() # Remove individual fitness file after concatenation

    print(f"Total runtime: {solver.runtime_} second")

    print("GA run terminated normally!")
    print(
        "Memory used: {} MB.".format(
            psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        )
    )


if __name__ == "__main__":
    main()
