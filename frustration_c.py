#!/usr/bin/env python3

import pickle
import sys

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
    StratifiedKFold,
    train_test_split,
)

# Verbosity flag
verb = 5
refit = False
crossvalidate = False
filename = "classifier_trained.sav"

# Model training and selection
if refit:
    # Convert to numpy arrays
    df = pd.read_csv("selected_data_energies_distances.csv", index_col=0)
    print(df.head())

    ndf = df.iloc[:, 13:].values
    X = ndf[:, :-1]
    y = ndf[:, -1]

    # All formatted now, X contains the different buried volumes and y is 0 (not frustrated) and 1 (frustrated)
    print(X[0, :], y[0])
    print(X[1, :], y[1])
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
    print(f"SGD linear model accuracy on test: {acc_a}")
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
    print(f"Logreg accuracy on test: {acc_b}")
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
    print(f"RF accuracy on test: {acc_c}")
    if acc_c > curr_acc:
        curr_acc = acc_c
        c = gsc.best_estimator_
    pickle.dump(c, open(filename, "wb"))

# We select the best model, which is the random forest typically
print(filename)
frustration_model_loaded = False
try:
    c = pickle.load(open(filename, "rb"))
    frustration_model_loaded = True
except (FileNotFoundError, ValueError) as e:
    print(f"Warning: Could not load frustration model from {filename}. Frustration prediction will be skipped. Error: {e}")
    c = None # Set c to None if loading fails

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
        print(
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


def predict_from_xyz(filename: str, idx_b: int, idx_n: int):
    global c, frustration_model_loaded
    if not frustration_model_loaded:
        print("Warning: Frustration model not loaded. Skipping frustration prediction.")
        return None # Or some other appropriate placeholder

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
    frustrated = c.predict(repr.reshape(1, -1))
    return frustrated


if __name__ == "__main__":
    idx_b = int(sys.argv[2])
    idx_n = int(sys.argv[3])
    print(
        f"Filename is {sys.argv[1]}, boron index is {idx_b} and nitrogen index is {idx_n}."
    )
    print(f"Frustration is {predict_from_xyz(sys.argv[1], idx_b, idx_n)}.")
