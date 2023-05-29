import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
import base64
import tensorflow as tf
import numpy as np



def converter(instr):
    return np.fromstring(instr[1:-1],sep=' ')

def load_data():
    # Load the data
    file_path = "./data/cdr3_seqs.csv"
    cdr3_seqs_df = pd.read_csv(file_path, converters={'CDR3_al_integer':converter})
    return cdr3_seqs_df

def preprocess(data, user_data, model_name, N=100):
    X_test = None
    sample = None
    if model_name == "Simple AutoEncoder":
        if data["CDR3"].str.contains(user_data).any():
            # select one row only with user data
            row = data[data["CDR3"] == user_data]
            # add row at end 
            data = data.append(row)
            X_test = np.stack(data["CDR3_al_integer"].to_numpy())
            X_test = tf.constant(X_test)
            # select last N rows 
            X_test = X_test[-N:]
            sample = data[-N:]
        else:
            X_test = np.stack(data["CDR3_al_integer"].to_numpy())
            X_test = tf.constant(X_test)
            X_test = X_test[-N:]
            sample = data[-N:]
    return X_test, sample

def plot_clusters(data, prediction, model_name):
    # Plot clusters using UMAP for dimensionality reduction
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(prediction)

    # Cross validation for k based on silhouette score   
    # Split data into train and test sets
    X_train, X_test = train_test_split(embedding, test_size=0.2, random_state=42)
    # Create a list of possible k values
    k_values = list(range(2, 10))
    # Create a list to store the silhouette scores for each k
    silhouette_scores = []
    # For each k value
    for k in k_values:
        # Create a kmeans instance with k clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        # Fit the kmeans model to the training data
        kmeans.fit(X_train)
        # Use it to predict the labels of the test data
        labels = kmeans.predict(X_test)
        # Get the average silhouette score
        score = silhouette_score(X_test, labels)
        # Store the score for this k value
        silhouette_scores.append(score)

    # Get the k value with highest silhouette score
    max_score = max(silhouette_scores)
    max_score_index = silhouette_scores.index(max_score)
    best_k = k_values[max_score_index]

    # Create a kmeans instance with best k
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    # Fit the kmeans model to the data
    kmeans.fit(embedding)
    # Use it to predict the labels of the data
    labels = kmeans.predict(embedding)

    perform_stats = []
    perform_stats.append(silhouette_score(embedding, labels))
    perform_stats.append(calinski_harabasz_score(embedding, labels))
    perform_stats.append(davies_bouldin_score(embedding, labels))

    # Add column with embeddings to data
    data["embedding"] = list(embedding)

    # Select rows that have the same label as the predicted sequence
    cluster = data[labels == labels[-1]]
    print(cluster.value_counts("Amino Acids 1"))
    majority_target = cluster.value_counts("Amino Acids 1").max()
    majority_label = cluster.value_counts("Amino Acids 1").idxmax()
    print(cluster.shape[0])
    # Percentage of the majority target in the cluster
    majority_target_percentage = (majority_target / cluster.shape[0])*100
    # Get centroid of the cluster
    centroid = kmeans.cluster_centers_[labels[-1]]
    
    import matplotlib
    min_val, max_val = 0.5,1.0
    n = best_k
    orig_cmap = plt.cm.BuGn
    colors = orig_cmap(np.linspace(min_val, max_val, n))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)
    # Plot the clusters
    plt.switch_backend('Agg') 
    sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=10)
    plt.gca().set_aspect('equal', 'datalim')

    plt.colorbar(boundaries=np.arange(best_k+1)-0.5).set_ticks(np.arange(best_k))
    plt.title(f"UMAP projection of the {model_name} embeddings with labels from K-means clustering", fontsize=24)
    plt.text(centroid[0]+0.01, centroid[1], f"Predicted: {majority_label} ({majority_target_percentage:.2f}%)", fontsize=14)
    plt.scatter(embedding[-1, 0], embedding[-1, 1], c='red', s=50)

    plt.savefig(f"./plots/{model_name}.png", bbox_inches='tight')

    test_base64 = base64.b64encode(open(f"./plots/{model_name}.png", 'rb').read()).decode('ascii')
    img = 'data:image/png;base64,{}'.format(test_base64)
    
    return img, majority_label, majority_target_percentage, perform_stats
    
