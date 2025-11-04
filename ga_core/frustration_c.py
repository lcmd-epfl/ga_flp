#!/usr/bin/env python3

import pickle
import sys
import logging

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

# Sets the verbosity level for output messages. Higher values typically mean more verbose output.
verb = 5
# Flag to indicate whether the model should be re-trained (refitted) or not.
refit = False
# Flag to indicate whether cross-validation should be performed.
crossvalidate = False
# The filename for saving/loading the trained classifier model.
filename = "classifier_trained.sav"

# Model training and selection
def train_and_evaluate_model(filename: str, verb: int, crossvalidate: bool):
    """
    Trains and evaluates a machine learning model for frustration prediction.

    This function loads data, splits it into training and testing sets,
    performs GridSearchCV to find the best estimator among SGDClassifier,
    LogisticRegression, and RandomForestClassifier, and then saves the best model.
    Optionally performs cross-validation and displays classification reports and confusion matrices.

    Args:
        filename: The name of the file to save the trained model.
        verb: Verbosity level for output messages.
        crossvalidate: Flag to indicate whether cross-validation should be performed.
    """
    # Convert to numpy arrays
    df = pd.read_csv("selected_data_energies_distances.csv", index_col=0)
    logger.info(df.head())

    ndf = df.iloc[:, 13:].values
    X = ndf[:, :-1]
    y = ndf[:, -1]

    # All formatted now, X contains the different buried volumes and y is 0 (not frustrated) and 1 (frustrated)
    logger.info(f"{X[0, :]}, {y[0]}")
    logger.info(f"{X[1, :]}, {y[1]}")
    acc_scorer = make_scorer(accuracy_score)
    X, X_te, y, y_te = train_test_split(X, y, test_size=0.10)

    # To select the best approach
    curr_acc = 0

    a = sgdlogreg(n_jobs=-1)
    param_grid = {
        "loss": ["hinge", "log_loss", "modified_huber"],
        "shuffle": [True, False],
        "penalty": ["l2", "l1", "elasticnet"],
        "max_iter": [10000],
        "alpha": [0.0001, 0.0002, 0.001, 0.01],
    }
    gsa = GridSearchCV(
        estimator=a, param_grid=param_grid, scoring=acc_scorer, cv=10, verbose=0
    )
    gsa.fit(X, y)
    acc_a = gsa.best_estimator_.score(X_te, y_te)
    logger.info(f"SGD linear model accuracy on test: {acc_a}")
    if acc_a > curr_acc:
        curr_acc = acc_a
        c = gsa.best_estimator_

    b = logreg(n_jobs=-1, solver="saga")
    param_grid = {
        "penalty": ["l2"],
        "max_iter": [2000],
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }
    gsb = GridSearchCV(
        estimator=b, param_grid=param_grid, scoring=acc_scorer, cv=10, verbose=0
    )
    gsb.fit(X, y)
    acc_b = gsb.best_estimator_.score(X_te, y_te)
    logger.info(f"Logreg accuracy on test: {acc_b}")
    if acc_b > curr_acc:
        curr_acc = acc_b
        c = gsb.best_estimator_

    c = rf(n_jobs=-1)
    param_grid = {
        "n_estimators": [50, 100, 200, 500],
        "max_features": ["sqrt", 400, 600],
        "bootstrap": [True, False],
        "max_samples": [None],
    }
    gsc = GridSearchCV(
        estimator=c, param_grid=param_grid, scoring=acc_scorer, cv=10, verbose=0
    )
    gsc.fit(X, y)
    acc_c = gsc.best_estimator_.score(X_te, y_te)
    logger.info(f"RF accuracy on test: {acc_c}")
    if acc_c > curr_acc:
        curr_acc = acc_c
        c = gsc.best_estimator_
    pickle.dump(c, open(filename, "wb"))

    # We select the best model, which is the random forest typically
    logger.info(filename)

    if crossvalidate:
        # Cross validation after reloading data
        X = ndf[:, :-1]
        y = ndf[:, -1]
        kf = KFold(n_splits=20)
        y_pr = np.zeros_like(y)
        for train, test in kf.split(X, y):
            X_tr, X_te, y_tr, y_te = X[train], X[test], y[train], y[test]
            c.fit(X_tr, y_tr)
            y_pr[test] = c.predict(X_te)

        if verb > 0:
            logger.info(
                classification_report(
                    y, y_pr, target_names=["Not frustrated", "Frustrated"]
                )
            )
        if verb > 1:
            titles_options = [
                ("Confusion matrix, without normalization", None),
                ("Normalized confusion matrix", "true"),
            ]
            for title, normalize in titles_options:
                disp = ConfusionMatrixDisplay.from_estimator(
                    c,
                    X,
                    y,
                    display_labels=["Not frustrated", "Frustrated"],
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                )
                disp.ax_.set_title(title)
            plt.show()
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
            self.model = pickle.load(open(self.model_filename, "rb"))
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
            filename: The path to the XYZ file.
            idx_b: The index of the boron atom.
            idx_n: The index of the nitrogen atom.

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

def run_frustration_prediction():
    """
    Runs the frustration prediction process.
    Initializes the FrustrationPredictor, reads command-line arguments for
    filename, boron index, and nitrogen index, then prints the prediction.
    """
    predictor = FrustrationPredictor()
    idx_b = int(sys.argv[2])
    idx_n = int(sys.argv[3])
    logger.info(
        f"Filename is {sys.argv[1]}, boron index is {idx_b} and nitrogen index is {idx_n}."
    )
    logger.info(f"Frustration is {predictor.predict_from_xyz(sys.argv[1], idx_b, idx_n)}.")


if __name__ == "__main__":
    if refit:
        train_and_evaluate_model(filename, verb, crossvalidate)
    run_frustration_prediction()
