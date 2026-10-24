import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
# files
P1_X = r"D:\COURSES\MIT STATISTICAL MODELING\data\release2\p1\X.npy"
P1_Y = r"D:\COURSES\MIT STATISTICAL MODELING\data\release2\p1\Y.npy"
P2_X = r"D:\COURSES\MIT STATISTICAL MODELING\data\release2\p2_unsupervised\X.npy"
# loading the numpy files
data_x = np.load(P1_X)
data_y = np.load(P1_Y)

# loading P2
p2_x = np.load(P2_X)

def log_transformation(data):
    # log2(X+1) transformation of data
    data_transform = np.log2(data + 1)
    return data_transform

# log_transform
log_transform = log_transformation(data_x)
p2_log_trans = log_transformation(p2_x)
# largest element
print(p2_log_trans[:, 0].max()) 

plt.scatter(log_transform[:, 0], log_transform[:, 1])

# PCA on log transformation and data x
def pca(data, n_components=None):
    pca = PCA(n_components=n_components)
    pca_fit = pca.fit(data)
    pca_trans = pca_fit.transform(data)
    return pca_trans


def scatter_plot(x, y, title, fontsize = 14, c=None):
    plt.scatter(x, y, c=c)
    plt.title(title, fontsize=fontsize)
    
# PCA visualtion
def pca_viz(data, n_components = None, c=None):
    #pca visualization
    x_pca_ = pca(data, n_components=n_components)
    scatter_plot(x_pca_[:,0], x_pca_[:,1], "PCA", c=c)
#MDS
def mds_viz(data, n_components=2, c=None):
    mds = MDS(n_components=n_components)
    x_mds_ = mds.fit_transform(data)
    scatter_plot(x_mds_[:, 0], x_mds_[:, 1], "MDS", c=c)
#TSNE
def tsne(data, pca_n=50, tsne_n=2, perplexity=40):
    #pcaxn as in pca20, 30, 40, 50 anything
    x_pca_xn = pca(data, n_components=pca_n)
    tsne = TSNE(n_components=tsne_n, perplexity=perplexity)

    x_tsne_ = tsne.fit_transform(x_pca_xn)
    print(tsne.kl_divergence_)
    return x_tsne_

# plot
def tsne_viz(data, pca_n=50, tsne_n=2, perplexity=40, c=None):
    x_tsne_ = tsne(data, pca_n=pca_n, tsne_n=tsne_n, perplexity=perplexity)  
    scatter_plot(x_tsne_[:, 0], x_tsne_[:, 1], "t-SNE Visualization", c=c)


def K_means(data, n_clusters, pca_n=50):
    x_pca_xn = pca(data, n_components=pca_n)
    k_means_x = KMeans(n_clusters=n_clusters)
    k_means_x.fit(x_pca_xn)
    cluster_labels = k_means_x.labels_

    return cluster_labels
KMeans(n_clusters=n_clusters).fit(x_pca_xn).labels_
# kmeans is for all of them
def K_means_pca(data, n_clusters, pca_n=50):
    cluster_labels  = K_means(data, n_clusters, pca_n=pca_n)
    pca_viz(data, c = cluster_labels)

def K_means_mds(data, n_clusters, pca_n=50):
    cluster_labels  = K_means(data, n_clusters, pca_n=pca_n)
    mds_viz(data, c = cluster_labels)
    
def K_means_tsne(data, n_clusters, pca_n=50):
    cluster_labels  = K_means(data, n_clusters)
    tsne_viz(data, pca_n=pca_n, c = cluster_labels)
    
def h_clustering(data, pca_n=50):
    x_pca_xn = pca(data, n_components=pca_n)
    hc = AgglomerativeClustering(
        n_clusters = 3,
        linkage="ward",
        metric="euclidean"
    )
    labels = hc.fit_predict(x_pca_xn)
    return labels

def h_pca(data, pca_n=50):
    cluster_labels  = h_clustering(data, pca_n=pca_n)
    pca_viz(data, c = cluster_labels)

def h_mds(data, pca_n=50):
    cluster_labels  = h_clustering(data, pca_n=pca_n)
    mds_viz(data, c = cluster_labels)
    
def h_tsne(data, pca_n=50):
    cluster_labels  = h_clustering(data, pca_n=pca_n)
    tsne_viz(data, pca_n=pca_n, c = cluster_labels)
def wgss():
    x_pca_50 = pca(p2_log_trans, n_components=50)
    wgss = []
    cluster_range = range(1, 30)
    for i in cluster_range:
        l_kmeans = KMeans(i)
        l_kmeans.fit_predict(x_pca_50)
        wgss.append(l_kmeans.inertia_)
    plt.plot(cluster_range, wgss, marker='o')
    plt.title('Elbow Method Using WGSS')
    return wgss
