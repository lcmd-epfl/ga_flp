# Genetic Algorithm for Frustrated Lewis Pairs (GA-FLP)

## Project Description
This project implements a Genetic Algorithm (GA) to optimize Frustrated Lewis Pairs (FLPs) for chemical applications. It leverages computational chemistry tools like RDKit and xtb for molecular property calculations and a pre-trained model for frustration prediction. The GA explores a chemical space defined by SMILES strings to discover novel FLP structures with desired characteristics.

## Setup

### Prerequisites
- Python 3.x
- `conda` or `pip` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ga_clean_flp
    ```

2.  **Create and activate a conda environment (recommended):**
    ```bash
    conda create -n ga_flp python=3.9 # or your preferred Python version
    conda activate ga_flp
    ```
    **Or, using pip:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    For `chimera`, this appears to be a local package. Ensure it is correctly installed or available in your Python path.

4.  **Install NaviCatGA:**
    ```bash
    pip install git+https://github.com/lcmd-epfl/NaviCatGA.git
    ```

4.  **Install `xtb`:**
    Follow the instructions on the `xtb` website to install it for your operating system and ensure it's available in your system's PATH.

## Usage

To run the genetic algorithm, execute the `launcher.py` script with the path to your database Excel file:

```bash
python launcher.py [data]
```

The script will perform a series of GA cycles, optimizing the FLP structures.

## Code Structure

-   `launcher.py`: The main script that orchestrates the genetic algorithm. It sets up the GA solver, runs the optimization, and handles logging and result saving.
-   `ga_core/`: Contains core GA-FLP functionalities.
    -   `ga_core/ga_flp.py`: Defines the `chromosome_to_smiles` and `overall_fitness_function` used by the GA.
    -   `ga_core/angle_measure.py`: Provides functions for calculating angles in molecular structures.
    -   `ga_core/ga_flp_fs.py`: Contains functions for molecular embedding, property calculations (B-N distance, HA, PA), and `xtb` optimizations.
-   `frustration_c.py`: Implements the `FrustrationPredictor` class for predicting molecular frustration using a pre-trained model.
-   `NaviCatGA/`: This directory appears to contain a custom genetic algorithm library (`navicatGA`) which is used by the `launcher.py`.
    -   `NaviCatGA/navicatGA/smiles_solver.py`: A specialized GA solver for SMILES strings.
    -   `NaviCatGA/navicatGA/selfies_solver.py`: A specialized GA solver for SELFIES strings.

## Logging

The `launcher.py` script uses Python's `logging` module with colored output for better readability. Key information for each GA cycle, such as max fitness, mean fitness, and the best individual's SMILES and chromosome, will be printed to the console.

## Troubleshooting

-   **`InconsistentVersionWarning` from `sklearn`**: This warning indicates that the `classifier_trained.sav` model was trained with a different version of scikit-learn than the one currently installed. While it might still work, it's recommended to retrain the model with your current scikit-learn version if you encounter issues or desire optimal performance.
-   **`Warning: Could not load frustration model from classifier_trained.sav. Frustration prediction will be skipped.`**: This means the `classifier_trained.sav` file was not found or could not be loaded. Ensure the file is in the correct directory or that its path is correctly specified. If frustration prediction is critical, you might need to train and save the model first.
-   **`TypeError: 'float' object is not subscriptable`**: This issue has been addressed in `ga_core/ga_flp.py`. Ensure you have the latest version of the code.
-   **Excessive "Randomizing starting chromosome." messages**: These messages have been suppressed by adjusting the logging level in the `navicatGA` library.
-   **Program times out**: The GA can be computationally intensive. If the program consistently times out (e.g., with `timeout 60`), consider increasing the `timeout` duration or reducing the `NUM_CYCLES` in `launcher.py` for quicker runs.
