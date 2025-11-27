# Genetic Algorithm for Frustrated Lewis Pairs (GA-FLP)

## Project Description
This project implements a Genetic Algorithm (GA) to optimize Frustrated Lewis Pairs (FLPs) for chemical applications. It leverages computational chemistry tools like RDKit and xtb for molecular property calculations and a pre-trained model for frustration prediction. The GA explores a chemical space defined by SMILES strings to discover novel FLP structures with desired characteristics.

## Setup

### Prerequisites
- Python 3.9 or higher
- `conda` or `pip` for package management

### Installation


    git clone https://github.com/lcmd-epfl/ga_flp
    conda create -n ga_flp python=3.9 # or your preferred Python version
    conda activate ga_flp

    # Clone the SCScore repository
    git clone https://github.com/connorcoley/scscore.git scscore

    pip install -r requirements.txt

    git clone https://github.com/lcmd-epfl/NaviCatGA/
    conda install xtb==22.1
    cd NaviCatGA
    python setup.py install
    cd ..
    rm -rf NaviCatGA

    
## Usage

To run the genetic algorithm, execute the `launcher.py` script. GA parameters, random seed, and the data file path are now configured via a JSON file.

```bash
python launcher.py --config config/config.json
```

### Configuration File (`config/config.json`)

The `config/config.json` file contains all the configurable parameters for the Genetic Algorithm run. An example structure is shown below:

```json
{
    "GA_PARAMETERS": {
        "N_GENES": 8,
        "POP_SIZE": 50,
        "MUTATION_RATE": 0.25,
        "N_CROSSOVER_POINTS": 1,
        "NUM_CYCLES": 10,
        "RANDOM_SEED": 42,
        "DATA_FILE": "data/database_HC_bbr.csv"
    }
}
```

**Parameters:**
-   `N_GENES`: Number of genes in each chromosome.
-   `POP_SIZE`: Size of the population for the genetic algorithm.
-   `MUTATION_RATE`: Probability of mutation for each gene.
-   `N_CROSSOVER_POINTS`: Number of crossover points during genetic recombination.
-   `NUM_CYCLES`: Number of genetic algorithm cycles to run.
-   `RANDOM_SEED`: Seed for the random number generator to ensure reproducibility.
-   `DATA_FILE`: Path to the database file (CSV or Excel) containing molecular data.

The script will perform a series of GA cycles, optimizing the FLP structures.

## Code Structure

-   `launcher.py`: The main script that orchestrates the genetic algorithm. It sets up the GA solver, runs the optimization, and handles logging and result saving.
-   `ga_core/`: Contains core GA-FLP functionalities.
    -   `ga_core/ga_flp.py`: Defines the `chromosome_to_smiles` and `overall_fitness_function` used by the GA.
    -   `ga_core/angle_measure.py`: Provides functions for calculating angles in molecular structures.
    -   `ga_core/ga_flp_fs.py`: Contains functions for molecular embedding, property calculations (B-N distance, HA, PA), and `xtb` optimizations.
    -   `ga_core/frustration_c.py`: Implements the `FrustrationPredictor` class for predicting molecular frustration using a pre-trained model. See the "Data and Model" section for more details.

## Data and Model

This project includes a pre-trained model and the dataset used to train it.

-   `models/classifier_trained.sav`: This is the pre-trained machine learning model (a scikit-learn classifier) used by the `FrustrationPredictor` for quick predictions of molecular frustration.
-   `data/selected_data_energies_distances.csv`: This CSV file contains the dataset used to train the frustration prediction model. It includes various molecular properties and the "Frustrated" class label, which is used to train the classifier.
-   `data/quenching/`: This directory contains the dataset for training the classifier quenching model.
    -   `data/quenching/bvs_energies_distances.csv`: A CSV file with energies, distances, and other molecular descriptors for a set of molecules.
    -   `data/quenching/xyz/`: A directory containing the corresponding molecular geometry `.xyz` files for the entries in the CSV.

## Logging

The `launcher.py` script uses Python's `logging` module with colored output for better readability. Key information for each GA cycle, such as max fitness, mean fitness, and the best individual's SMILES and chromosome, will be printed to the console.

## Troubleshooting

-   **`InconsistentVersionWarning` from `sklearn`**: This warning indicates that the `classifier_trained.sav` model was trained with a different version of scikit-learn than the one currently installed. While it might still work, it's recommended to retrain the model with your current scikit-learn version if you encounter issues or desire optimal performance.
-   **`Warning: Could not load frustration model from classifier_trained.sav. Frustration prediction will be skipped.`**: This means the `classifier_trained.sav` file was not found or could not be loaded. Ensure the file is in the correct directory or that its path is correctly specified. If frustration prediction is critical, you might need to train and save the model first.
-   **`TypeError: 'float' object is not subscriptable`**: This issue has been addressed in `ga_core/ga_flp.py`. Ensure you have the latest version of the code.
-   **Excessive "Randomizing starting chromosome." messages**: These messages have been suppressed by adjusting the logging level in the `navicatGA` library.
-   **Program times out**: The GA can be computationally intensive. If the program consistently times out (e.g., with `timeout 60`), consider increasing the `timeout` duration or reducing the `NUM_CYCLES` in `launcher.py` for quicker runs.


## Citing

If you use this code in your research, please cite the following publication:

    Coming soon...