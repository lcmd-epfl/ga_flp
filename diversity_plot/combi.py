#!/usr/bin/env python3

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import RDConfig
import sklearn.metrics.pairwise
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import product
import os
import sys
from sklearn.cluster import AgglomerativeClustering, OPTICS, MiniBatchKMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from openTSNE import TSNE
import multiprocessing
from joblib import parallel_backend
from scipy.spatial.distance import euclidean, squareform
from scipy.cluster.hierarchy import dendrogram
import matplotlib
from sklearn.neighbors import NearestCentroid

matplotlib.use("Agg")
verb = 2
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

nc = multiprocessing.cpu_count()

file_a = "total_raw_data.txt"


def plot_tsne(
    xy,
    colors=None,
    alpha=0.75,
    figsize=(8, 8),
    s=42,
    cmap="jet",
    filename="colored_tsne.png",
):
    if colors is not None:
        cmap = plt.cm.viridis
        if np.size(np.unique(colors)) == 2:
            colors = np.array(colors, dtype=int)
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "Custom cmap", cmaplist, cmap.N
            )
            bounds = range(3)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        else:
            norm = None
    else:
        norm = None

    plt.figure(figsize=figsize, facecolor="white")
    plt.margins(0)
    plt.axis("off")
    fig = plt.scatter(
        xy[:, 0],
        xy[:, 1],
        c=colors,  # set colors of markers
        cmap=cmap,  # set color map of markers
        s=s,
        alpha=alpha,  # set alpha of markers
        marker="o",  # use smallest available marker (square)
        lw=0,  # don't use edges
        edgecolor="black",
        norm=norm,
    )  # don't use edges
    plt.colorbar(fig)
    # remove all axes and whitespace / borders
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(filename)
    plt.close()


def plot_histogram(data, name, output_filename):
    """
    Plot a histogram of a NumPy array and save it as a PNG.

    Parameters:
    data (ndarray): The data to plot.
    output_filename (str): The name of the file to save the plot to.

    Returns:
    None
    """

    if name == "theta":
        name = r"$\Phi$"
    if name == "d_chem":
        name = r"$d_{chem}$"
    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the histogram
    n, bins, patches = ax.hist(
        data, bins=20, color="gray", edgecolor="black", alpha=0.7
    )

    # Add vertical lines for mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    ax.axvline(mean, color="red", linestyle="--", linewidth=2)
    ax.axvline(mean - std, color="green", linestyle="--", linewidth=2)
    ax.axvline(mean + std, color="green", linestyle="--", linewidth=2)

    # Set the title and labels
    # ax.set_title("Histogram", fontsize=16)
    ax.set_xlabel(f"{name}", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)

    # Add a legend with mean and standard deviation
    ax.legend(
        ["Mean: {:.2f}".format(mean), "± 1 SD: {:.2f}".format(std)],
        loc="upper right",
        fontsize=12,
    )

    # Set the tick font size
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Save the plot as a PNG
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")

    # Show the plot
    # plt.show()
    plt.close()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def check_arr(arr, name):
    print(f"{name} max value is {np.max(arr)} and min is {np.min(arr)}")


with open(file_a, "r") as f:
    lines = [k.rstrip() for k in list(f.readlines())]
    # header = lines.pop(0)
    smiles_a = [l.split(",")[0].rstrip() for l in lines]
    # f"{smiles},{dchem},{score3},{bn_distance},{angle},{fepa},{feha},{scs},{fr}",
    dchem_a = np.array([float(l.split(",")[1].rstrip()) for l in lines])
    d_a = np.array([float(l.split(",")[3].rstrip()) for l in lines])
    a_a = np.array([float(l.split(",")[4].rstrip()) for l in lines])
    fepa_a = np.array([float(l.split(",")[5].rstrip()) for l in lines])
    feha_a = np.array([float(l.split(",")[6].rstrip()) for l in lines])
    scs_a = np.array([float(l.split(",")[7].rstrip()) for l in lines])
    f_a = np.array([float(l.split(",")[8].rstrip()) for l in lines])
    mols_a = [Chem.MolFromSmiles(smi) for smi in smiles_a]
    fps_a = [
        np.array(
            AllChem.GetMorganFingerprintAsBitVect(
                mol, useChirality=True, radius=3, nBits=64, bitInfo={}
            ),
            dtype=bool,
        )
        for mol in mols_a
    ]
    X_a = np.array(fps_a)
    # db_a = AgglomerativeClustering(
    #    metric="rogerstanimoto",
    #    linkage="average",
    #    distance_threshold=0.25,
    #    n_clusters=None,
    # )
    # labels_a = db_a.fit_predict(X_a)
    # u_a, c_a = np.unique(labels_a, return_counts=True)
    # print(f"{u_a} and {c_a} retained")

# smiles = [a + "," + b for a, b, in product(smiles_a, smiles_b)]
smiles = np.array(smiles_a)
# fps = [np.concatenate((a, b)) for a, b in product(fps_a, fps_b)]
X = np.array(fps_a)
indices_1 = np.where(dchem_a < 100)
indices_2 = np.where(scs_a < 100)
indices_3 = np.where(feha_a > -1000)
indices_4 = np.where(fepa_a > -1000)
indices = np.intersect1d(
    np.intersect1d(np.intersect1d(indices_1, indices_2), indices_3), indices_4
)
# indices = np.unique(
#    np.concatenate((indices_1, indices_2, indices_3, indices_4), axis=1)
# )
dchem_a = dchem_a[indices]
d_a = d_a[indices]
a_a = a_a[indices]
feha_a = np.clip(feha_a[indices], a_min=np.min(feha_a), a_max=0)
fepa_a = np.clip(fepa_a[indices], a_min=np.min(fepa_a), a_max=0)
f_a = f_a[indices]
scs_a = scs_a[indices]
X = X[indices]
properties = [dchem_a, d_a, a_a, fepa_a, feha_a, scs_a, f_a]
property_names = ["d_chem", "d", "theta", "FEPA", "FEHA", "SCS", "QS"]

# print(X.shape)
np.save("X.npy", X)
# perps = [250]  # , 50, 100, 250, 500]  # [5, 10, 20, 50]  # [50, 100, 250]
# distances = [
#    "jaccard",
#    "rogerstanimoto",
#    "cosine",
#    "euclidean",
# ]  # ["cosine", "euclidean"]
# for i, perplexity in enumerate(perps):
#    for j, dist in enumerate(distances):
#        tsne = TSNE(
#            n_components=2,
#            initialization="pca",
#            perplexity=perplexity,
#            early_exaggeration=50.0,
#            learning_rate=200.0,
#            n_iter=500,
#            metric=dist,
#            n_jobs=nc,
#            random_state=42,
#        )
#        E = tsne.fit(X)  # .transform(X)
#
#        print(X.shape)
#        Y = E.transform(X)
#        print(Y.shape)
#
#        np.save(f"Y_{dist}_{perplexity}.npy", Y)
#        plt.figure()
#        plt.scatter(Y[:, 0], Y[:, 1])
#        plt.savefig(f"Y_cor_{dist}_{perplexity}.png")
#        plt.close()
#


def silhouette_scorer(estimator, X):
    clusters = estimator.fit_predict(X)
    score = silhouette_score(X, clusters)
    return score


dist = "cosine"  # "rogerstanimoto"
perplexity = "250"  # "100"
choice = f"{dist}_{perplexity}"
Y = np.load(f"Y_{choice}.npy")

for property, property_name in zip(properties, property_names):
    check_arr(property, property_name)
    plot_tsne(Y, property, filename=f"{property_name}_{choice}.png")
    plot_histogram(property, property_name, f"{property_name}_hist.png")

exit()
assert Y.shape[1] == 2
with parallel_backend("threading", n_jobs=nc):
    db = AgglomerativeClustering(
        n_clusters=None,
        metric="l2",  # "rogerstanimoto",
        linkage="average",
        memory="tmp",
        compute_full_tree=True,
    )
    # distance_threshold=7.5,
    # n_clusters=None,
    classifier = GridSearchCV(
        db,
        {"distance_threshold": range(1, 5)},
        scoring=silhouette_scorer,
        cv=[(slice(None), slice(None))],
    )
    classifier.fit(Y)
    db = classifier.best_estimator_
    labels = db.fit_predict(Y)
u, c = np.unique(labels, return_counts=True)

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(db, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.savefig(f"clustering_{choice}_dendrogram.png")
plt.close()

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Estimated number of clusters: %d" % n_clusters_)
print(f"With compositions {u} : {c}")
print("Estimated number of noise points: %d" % n_noise_)
clf = NearestCentroid()
clf.fit(Y, labels)

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
closest_pt_idx = []
clusters = []
for iclust in range(u.size):
    # get all points assigned to each cluster:
    clusters.append(np.where(db.labels_ == iclust)[0])

    # get all indices of points assigned to this cluster:
    cluster_pts_indices = np.where(db.labels_ == iclust)[0]

    # cluster_cen = db.cluster_centers_[iclust]
    cluster_cen = clf.centroids_[iclust]
    min_idx = np.argmin([euclidean(Y[idx], cluster_cen) for idx in cluster_pts_indices])

    if verb > 1:
        print(
            f"Closest index of point to cluster {iclust} center has index {cluster_pts_indices[min_idx]}, smiles {smiles[cluster_pts_indices[min_idx]]} and coordinates {Y[cluster_pts_indices[min_idx]]}"
        )
    closest_pt_idx.append(cluster_pts_indices[min_idx])
i_cluster_centers = np.array(closest_pt_idx, dtype=int)

core_samples_mask[i_cluster_centers] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k
    xy = Y[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )
    xy = Y[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "X",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=12,
    )

plt.savefig(f"clustering_{choice}.png")

f = open("Results.csv", "w+")
print("Smiles,x,y,cluster", file=f)
for i, _ in enumerate(smiles):
    print(f"{smiles[i]},{Y[i,0]},{Y[i,1]},{labels[i]}", file=f)

f.close()
