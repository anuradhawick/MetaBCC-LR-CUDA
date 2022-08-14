import logging
from random import sample
from typing import Counter

import matplotlib
from scipy.stats import multivariate_normal

try:
    import cudf
    import cupy as cp
    from cuml.cluster.hdbscan import HDBSCAN
    from cuml.manifold.t_sne import TSNE
    from cuml.manifold.umap import UMAP
    from cuml.metrics.cluster.silhouette_score import (
        cython_silhouette_score as silhouette_score,
    )

    CUDA = True
except:
    import numpy as cp
    import pandas as cudf
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    CUDA = False

# import umap
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

# from song.song import SONG
from tabulate import tabulate

# matplotlib.use("Agg")


logger = logging.getLogger("MetaBCC-LR")


def plot_cluster(features, clusters, labels, title=""):
    perplexity = 50
    # features_2D = (
    #     TSNE(method="barnes_hut", perplexity=perplexity, n_neighbors=3 * perplexity)
    #     .fit_transform(features)
    #     .get()
    # )
    features_2D = UMAP().fit_transform(features).get()

    palette = {k: f"C{n}" for n, k in enumerate(set(labels))}
    palette2 = {k: f"C{n}" for n, k in enumerate(set(clusters.get()))}

    plt.figure(figsize=(10, 10))
    plt.title(title + " truth")
    sns.scatterplot(x=features_2D.T[0], y=features_2D.T[1], hue=labels, palette=palette)

    plt.figure(figsize=(10, 10))
    plt.title(title + " clusters")
    sns.scatterplot(
        x=features_2D.T[0], y=features_2D.T[1], hue=clusters.get(), palette=palette2
    )
    # plt.show()


def detect_best_sampling_tsne(features):
    no_rows, _ = features.shape
    lower = 20000
    upper = no_rows // 10
    splits = 5
    split_size = (no_rows // 10 - lower) // splits + 1
    perplexity = 50
    best_score = float("-inf")
    best_clusters = None
    best_features = None

    for sampling_size in range(lower, upper, split_size):
        row_indices = cp.random.choice(no_rows, min(no_rows, sampling_size), replace=False)
        features_sampled = features[row_indices]
        features_2D = TSNE(
            method="barnes_hut", perplexity=perplexity, n_neighbors=3 * perplexity
        ).fit_transform(features_sampled)
        clusters = HDBSCAN(min_cluster_size=200).fit_predict(features_2D)
        score = silhouette_score(features_2D[clusters != -1], clusters[clusters != -1])

        if score > best_score:
            best_score = score
            best_clusters = clusters[clusters != -1]
            best_features = features_sampled[clusters != -1]

    return best_clusters, best_features


def detect_best_sampling_umap(features):
    no_rows, _ = features.shape
    lower = 50000
    upper = min(no_rows // 10, 500000)
    splits = 5
    split_size = (no_rows // 10 - lower) // splits + 1
    best_score = float("-inf")
    best_clusters = None
    best_features = None

    for sampling_size in range(lower, upper, split_size):
        row_indices = cp.random.choice(no_rows, min(no_rows, sampling_size), replace=False)
        features_sampled = features[row_indices]
        features_2D = UMAP().fit_transform(features_sampled)
        try:
            clusters = HDBSCAN(min_cluster_size=500).fit_predict(features_2D)
            score = silhouette_score(features_2D[clusters != -1], clusters[clusters != -1])
        except:
            score = float('-inf')
        if score > best_score:
            best_score = score
            best_clusters = clusters[clusters != -1]
            best_features = features_sampled[clusters != -1]

    return best_clusters, best_features


def compute_probability_density(features, mean, std):
    std += 0.00001
    features = features
    mean = mean
    std = std

    prob = (
        -0.5 * ((features - mean) / std) ** 2 - cp.log((2 * cp.pi) ** 0.5 * std)
    ).sum(axis=1)

    return prob.reshape(-1, 1)


class Cluster:
    def __init__(self, data, indices, classifier, truth):
        self.data = data
        self.indices = indices
        self.truth = truth
        self.classifier = classifier
        self.clusters = None

    def update(self, new_data):
        # only data changes, nothing else
        self.data = new_data[self.indices]

    def recluster(self):
        clusters, features = self.classifier(self.data)
        # plot_cluster(self.data[indices], clusters, self.truth[indices.get()], title=f"")

        probabilities = []
        # using the detected clusters, cluster all the points
        for cluster in set(clusters.tolist()):
            cluster_features = features[clusters == cluster]

            probabilities.append(
                compute_probability_density(
                    self.data, cluster_features.mean(), cluster_features.std()
                )
            )
        probabilities = cp.concatenate(probabilities, axis=1)
        self.clusters = cp.argmax(probabilities, axis=1)

    def get_clusters(self):
        for cluster in set(self.clusters.tolist()):
            cluster_indices = self.indices[self.clusters == cluster]
            cluster_data = self.data[self.clusters == cluster]
            cluster_truth = self.truth[(self.clusters == cluster).get()]

            yield Cluster(cluster_data, cluster_indices, self.classifier, cluster_truth)


def run_binner(composition, coverage, embedding):
    ids = (
        cudf.read_csv("/home/anuvini/Desktop/MetaBCC-LR/test2/ids.txt", header=None)
        .to_numpy()
        .ravel()
    )
    c = Cluster(coverage, cp.arange(len(composition)), detect_best_sampling_umap, ids)
    c.recluster()

    with open("test.out.txt", "w+") as fo, open("test.truth.txt", "w+") as ft:
        # cluster by coverage
        for m, cov_cluster in enumerate(c.get_clusters()):
            cov_cluster.update(composition)
            # cluster again using composition
            cov_cluster.recluster()
            for n, com_cluster in enumerate(cov_cluster.get_clusters()):
                for i in com_cluster.indices.tolist():
                    fo.write(f"{m}-{n}\n")
                for i in com_cluster.truth.tolist():
                    ft.write(f"{i}\n")

    # plt.show()
