#!/usr/bin/env python3

import pickle
import sys
import logging
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from morfeus import BuriedVolume, read_xyz
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.linear_model import SGDClassifier as sgdlogreg
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    train_test_split,
)

logger = logging.getLogger(__name__)

def train_and_evaluate_model(csv_path: str, model_save_path: str, verbosity: int, cross_validate: bool):
    """
    Trains and evaluates a machine learning model for frustration prediction.

    This function loads data from a CSV file, splits it into training and testing sets,
    and then uses GridSearchCV to find the best estimator among three different
    classifier types:
    - SGDClassifier (Stochastic Gradient Descent)
    - LogisticRegression
    - RandomForestClassifier

    The best-performing model based on accuracy is saved to a file. The function
    can also perform cross-validation and generate a classification report and
    confusion matrix plots for a more detailed evaluation.

    Args:
        csv_path (str): The path to the input CSV file containing the training data.
        model_save_path (str): The path where the trained model will be saved.
        verbosity (int): The verbosity level for logging and output messages.
                         Higher values result in more detailed output.
        cross_validate (bool): A flag to indicate whether to perform cross-validation
                               after finding the best model.
    """
    # Load data from the provided CSV file path
    df = pd.read_csv(csv_path, index_col=0)
    logger.info(df.head())

    # Set global font size for publication quality
    plt.rcParams.update({'font.size': 12})

    # Convert to numpy arrays
    ndf = df.iloc[:, 2:].values
    X = ndf[:, :-3]
    y = ndf[:, -1]

    # All formatted now, X contains the different buried volumes and y is 0 (not frustrated) and 1 (frustrated)
    logger.info(f"{X[0, :]}, {y[0]}")
    logger.info(f"{X[1, :]}, {y[1]}")
    acc_scorer = make_scorer(accuracy_score)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    # To select the best approach
    curr_acc = 0
    best_estimator = None

    # SGDClassifier
    sgd = sgdlogreg(n_jobs=-1)
    param_grid_sgd = {
        "loss": ["hinge", "log_loss", "modified_huber"],
        "shuffle": [True, False],
        "penalty": ["l2", "l1", "elasticnet"],
        "max_iter": [10000],
        "alpha": [0.0001, 0.0002, 0.001, 0.01],
    }
    gs_sgd = GridSearchCV(
        estimator=sgd, param_grid=param_grid_sgd, scoring=acc_scorer, cv=10, verbose=0
    )
    gs_sgd.fit(X_train, y_train)
    acc_sgd = gs_sgd.best_estimator_.score(X_test, y_test)
    logger.info(f"SGD linear model accuracy on test: {acc_sgd}")
    if acc_sgd > curr_acc:
        curr_acc = acc_sgd
        best_estimator = gs_sgd.best_estimator_

    # Logistic Regression
    log_reg = logreg(n_jobs=-1, solver="saga")
    param_grid_logreg = {
        "penalty": ["l2"],
        "max_iter": [2000],
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }
    gs_logreg = GridSearchCV(
        estimator=log_reg, param_grid=param_grid_logreg, scoring=acc_scorer, cv=10, verbose=0
    )
    gs_logreg.fit(X_train, y_train)
    acc_logreg = gs_logreg.best_estimator_.score(X_test, y_test)
    logger.info(f"Logreg accuracy on test: {acc_logreg}")
    if acc_logreg > curr_acc:
        curr_acc = acc_logreg
        best_estimator = gs_logreg.best_estimator_

    # Random Forest
    rand_forest = rf(n_jobs=-1)
    param_grid_rf = {
        "n_estimators": [50, 100, 200, 500],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
    }
    gs_rf = GridSearchCV(
        estimator=rand_forest, param_grid=param_grid_rf, scoring=acc_scorer, cv=10, verbose=0
    )
    gs_rf.fit(X_train, y_train)
    acc_rf = gs_rf.best_estimator_.score(X_test, y_test)
    logger.info(f"RF accuracy on test: {acc_rf}")
    if acc_rf > curr_acc:
        best_estimator = gs_rf.best_estimator_

    # Save the best model
    pickle.dump(best_estimator, open(model_save_path, "wb"))
    logger.info(f"Best model saved to {model_save_path}")

    if cross_validate:
        # Cross validation after reloading data
        kf = KFold(n_splits=20)
        y_pred_full = np.zeros_like(y)
        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_tr, X_te = X[train_index], X[test_index]
            y_tr, y_te = y[train_index], y[test_index]
            best_estimator.fit(X_tr, y_tr)
            y_pred = best_estimator.predict(X_te)
            y_pred_full[test_index] = y_pred

            if verbosity > 1:
                # Plot confusion matrix for each fold
                plt.figure(figsize=(8, 6))
                disp = ConfusionMatrixDisplay.from_predictions(
                    y_te,
                    y_pred,
                    display_labels=["Not frustrated", "Frustrated"],
                    cmap=plt.cm.Blues,
                    normalize=None,
                )
                disp.ax_.set_title(f"Fold {i+1} Confusion Matrix")
                plt.tight_layout()
                figname = f"confusion_matrix_fold_{i+1}.png"
                plt.savefig(figname, dpi=300)
                logger.info(f"Saved figure: {figname}")
                plt.close()

                plt.figure(figsize=(8, 6))
                disp_norm = ConfusionMatrixDisplay.from_predictions(
                    y_te,
                    y_pred,
                    display_labels=["Not frustrated", "Frustrated"],
                    cmap=plt.cm.Blues,
                    normalize="true",
                )
                disp_norm.ax_.set_title(f"Fold {i+1} Normalized Confusion Matrix")
                plt.tight_layout()
                figname_norm = f"confusion_matrix_normalized_fold_{i+1}.png"
                plt.savefig(figname_norm, dpi=300)
                logger.info(f"Saved figure: {figname_norm}")
                plt.close()

        if verbosity > 0:
            report = classification_report(
                y, y_pred_full, target_names=["Not frustrated", "Frustrated"]
            )
            logger.info(report)
            print(report)

        if verbosity > 1:
            titles_options = [
                ("Confusion matrix, without normalization", None, "confusion_matrix.png"),
                ("Normalized confusion matrix", "true", "confusion_matrix_normalized.png"),
            ]
            for title, normalize, figname in titles_options:
                plt.figure(figsize=(8, 6))
                disp = ConfusionMatrixDisplay.from_predictions(
                    y,
                    y_pred_full,
                    display_labels=["Not frustrated", "Frustrated"],
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                )
                disp.ax_.set_title(title)
                plt.tight_layout()
                plt.savefig(figname, dpi=300)
                logger.info(f"Saved figure: {figname}")
            plt.close('all')


class FrustrationPredictor:
    """
    A class to predict molecular frustration using a pre-trained machine learning model.

    The model is loaded from a specified file, and predictions can be made from XYZ molecular structures.
    """
    def __init__(self, model_filename="classifier_trained.sav"):
        """
        Initializes the FrustrationPredictor.

        Args:
            model_filename (str): The name of the file containing the pre-trained model.
        """
        self.model = None
        self.model_loaded = False
        self.model_filename = model_filename
        self._load_model()

    def _load_model(self):
        """
        Loads the pre-trained frustration prediction model from the specified file.
        If the file is not found or there's an error during loading, frustration prediction
        will be skipped.
        """
        try:
            with open(self.model_filename, "rb") as f:
                self.model = pickle.load(f)
            self.model_loaded = True
            logger.info(f"Frustration model loaded successfully from {self.model_filename}.")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load frustration model from {self.model_filename}. Frustration prediction will be skipped. Error: {e}")
            self.model = None
            self.model_loaded = False

    def predict_from_xyz(self, filename: str, idx_b: int, idx_n: int):
        """
        Predicts molecular frustration from an XYZ file using a pre-trained model.

        Args:
            filename (str): The path to the XYZ file.
            idx_b (int): The index of the boron atom.
            idx_n (int): The index of the nitrogen atom.

        Returns:
            The prediction of frustration (0 for not frustrated, 1 for frustrated) or None if the model is not loaded.
        """
        if not self.model_loaded:
            logger.warning("Frustration model not loaded. Skipping frustration prediction.")
            return None

        elements, coordinates = read_xyz(filename)
        b_vsbv = BuriedVolume(
            elements, coordinates, idx_b, radius=2.0
        ).fraction_buried_volume
        b_sbv = BuriedVolume(
            elements, coordinates, idx_b, radius=2.5
        ).fraction_buried_volume
        b_mbv = BuriedVolume(
            elements, coordinates, idx_b, radius=3.0
        ).fraction_buried_volume
        b_lbv = BuriedVolume(
            elements, coordinates, idx_b, radius=3.5
        ).fraction_buried_volume
        n_vsbv = BuriedVolume(
            elements, coordinates, idx_n, radius=2.0
        ).fraction_buried_volume
        n_sbv = BuriedVolume(
            elements, coordinates, idx_n, radius=2.5
        ).fraction_buried_volume
        n_mbv = BuriedVolume(
            elements, coordinates, idx_n, radius=3.0
        ).fraction_buried_volume
        n_lbv = BuriedVolume(
            elements, coordinates, idx_n, radius=3.5
        ).fraction_buried_volume
        repr = np.array([b_vsbv, b_sbv, b_mbv, b_lbv, n_vsbv, n_sbv, n_mbv, n_lbv])
        frustrated = self.model.predict(repr.reshape(1, -1))
        return frustrated

def run_frustration_prediction(args):
    """
    Runs the frustration prediction process for a single molecule.
    
    Args:
        args: Command line arguments from argparse.
    """
    predictor = FrustrationPredictor(args.model)
    logger.info(
        f"Filename is {args.xyz_file}, boron index is {args.idx_b} and nitrogen index is {args.idx_n}."
    )
    frustration = predictor.predict_from_xyz(args.xyz_file, args.idx_b, args.idx_n)
    logger.info(f"Frustration is {frustration}.")
    print(f"Predicted frustration: {frustration}")

def main():
    """Main function to handle command-line arguments and run the appropriate action."""
    parser = argparse.ArgumentParser(description="Train a frustration prediction model or predict frustration for a molecule.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train and evaluate a model.')
    train_parser.add_argument('csv_path', type=str, help='Path to the training data CSV file.')
    train_parser.add_argument('--model_save_path', type=str, default='classifier_trained.sav', help='Path to save the trained model.')
    train_parser.add_argument('--verbosity', type=int, default=2, help='Verbosity level.')
    train_parser.add_argument('--no-cross-validation', action='store_false', dest='cross_validate', help='Disable cross-validation.')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict frustration for a single molecule.')
    predict_parser.add_argument('xyz_file', type=str, help='Path to the XYZ file.')
    predict_parser.add_argument('idx_b', type=int, help='Index of the boron atom.')
    predict_parser.add_argument('idx_n', type=int, help='Index of the nitrogen atom.')
    predict_parser.add_argument('--model', type=str, default='classifier_trained.sav', help='Path to the trained model file.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.command == 'train':
        train_and_evaluate_model(args.csv_path, args.model_save_path, args.verbosity, args.cross_validate)
    elif args.command == 'predict':
        run_frustration_prediction(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()