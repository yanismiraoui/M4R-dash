import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import base64
import tensorflow as tf
import numpy as np
import pickle as pkl


def align_seqs(seq, target_len=20):
    initial_len = len(seq)
    if initial_len < target_len:
        # insert gaps in the middle of the sequence
        n_gaps = target_len - initial_len
        # insert gaps in the middle of the sequence
        seq = seq[:initial_len//2] + '-'*n_gaps + seq[initial_len//2:]
    return seq

def converter(instr):
    return np.fromstring(instr[1:-1].replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '),sep=' ')

def load_data():
    # Load the data
    file_path = "./data/cdr3_seqs.csv"
    cdr3_seqs_df = pd.read_csv(file_path, converters={'CDR3_al_integer':converter, 'CDR3_al_one_hot':converter, "v_gene_one_hot":converter, "j_gene_one_hot":converter})
    return cdr3_seqs_df

def preprocess(data, user_data, v_gene="", j_gene="", model_name="", N=100):
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
            # encode CDR3 sequences as integers
            with open("./encoders/integer_encoder.pkl", "rb") as f:
                integer_encoder = pkl.load(f)
            # encode user data
            user_data_integer = integer_encoder.transform(list(user_data))
            # add user data at end
            data = data.append({"CDR3": user_data, "CDR3_al_integer": user_data_integer}, ignore_index=True)
            print(data["CDR3_al_integer"])
            X_test = np.stack(data["CDR3_al_integer"].to_numpy())
            X_test = tf.constant(X_test)
            X_test = X_test[-N:]
            sample = data[-N:]

    elif model_name == "Optimized Deep AutoEncoder":
        # Load encoders
        with open("./encoders/integer_encoder.pkl", "rb") as f:
            integer_encoder = pkl.load(f)
        with open("./encoders/v_gene_encoder.pkl", "rb") as f:
            v_gene_encoder = pkl.load(f)
        with open("./encoders/j_gene_encoder.pkl", "rb") as f:
            j_gene_encoder = pkl.load(f)
        with open("./encoders/v_gene_one_hot_encoder.pkl", "rb") as f:
            v_gene_one_hot_encoder = pkl.load(f)
        with open("./encoders/j_gene_one_hot_encoder.pkl", "rb") as f:
            j_gene_one_hot_encoder = pkl.load(f)
        if data["CDR3"].str.contains(user_data).any():
            # select one row only with user data
            row = data[data["CDR3"] == user_data]
            # add row at end 
            data = data.append(row)
            X_test = data[['CDR3_al_integer', 'v_gene', 'j_gene', 'v_gene_one_hot', 'j_gene_one_hot']]
            X_test.reset_index(drop=True, inplace=True)
            for k in X_test.index:
                integer_encoded_v = v_gene_encoder.transform([X_test.loc[k, 'v_gene']]) 
                integer_encoded_j = j_gene_encoder.transform([X_test.loc[k, 'j_gene']])
                X_test.loc[k, 'v_gene_one_hot'] = v_gene_one_hot_encoder.transform(integer_encoded_v.reshape(-1,1))
                X_test.loc[k, 'j_gene_one_hot'] = j_gene_one_hot_encoder.transform(integer_encoded_j.reshape(-1,1)) 
            X_test["v_gene_one_hot"] = X_test["v_gene_one_hot"].apply(lambda x: x.reshape(54))
            X_test["j_gene_one_hot"] = X_test["j_gene_one_hot"].apply(lambda x: x.reshape(13))
            X_test = X_test[['CDR3_al_integer', 'v_gene_one_hot', 'j_gene_one_hot']]

            X_test = [np.stack(col.values) for _, col in X_test.items()]
            # select last N rows 
            for i in range(len(X_test)):
                X_test[i] = X_test[i][-N:]
            sample = data[-N:]
        else:
            # encode user data
            user_data_integer = integer_encoder.transform(list(user_data))
            # add user data at end
            data = data.append({"CDR3": user_data, "CDR3_al_integer": user_data_integer, "v_gene": v_gene, "j_gene": j_gene, "v_gene_one_hot": v_gene_one_hot_encoder.transform(v_gene_encoder.transform([v_gene]).reshape(-1,1)), "j_gene_one_hot": j_gene_one_hot_encoder.transform(j_gene_encoder.transform([j_gene]).reshape(-1,1))}, ignore_index=True)
            X_test = data[['CDR3_al_integer', 'v_gene', 'j_gene', 'v_gene_one_hot', 'j_gene_one_hot']]
            X_test.reset_index(drop=True, inplace=True)
            for k in X_test.index:
                integer_encoded_v = v_gene_encoder.transform([X_test.loc[k, 'v_gene']]) 
                integer_encoded_j = j_gene_encoder.transform([X_test.loc[k, 'j_gene']])
                X_test.loc[k, 'v_gene_one_hot'] = v_gene_one_hot_encoder.transform(integer_encoded_v.reshape(-1,1))
                X_test.loc[k, 'j_gene_one_hot'] = j_gene_one_hot_encoder.transform(integer_encoded_j.reshape(-1,1)) 
            X_test["v_gene_one_hot"] = X_test["v_gene_one_hot"].apply(lambda x: x.reshape(54))
            X_test["j_gene_one_hot"] = X_test["j_gene_one_hot"].apply(lambda x: x.reshape(13))
            X_test = X_test[['CDR3_al_integer', 'v_gene_one_hot', 'j_gene_one_hot']]

            X_test = [np.stack(col.values) for _, col in X_test.items()]
            # select last N rows 
            for i in range(len(X_test)):
                X_test[i] = X_test[i][-N:]
            sample = data[-N:]
        
    elif model_name == "Variational AutoEncoder":
        # Load encoders
        with open("./encoders/integer_encoder.pkl", "rb") as f:
            integer_encoder = pkl.load(f)
        with open("./encoders/onehot_encoder.pkl", "rb") as f:
            onehot_encoder = pkl.load(f)
        with open("./encoders/v_gene_encoder.pkl", "rb") as f:
            v_gene_encoder = pkl.load(f)
        with open("./encoders/j_gene_encoder.pkl", "rb") as f:
            j_gene_encoder = pkl.load(f)
        with open("./encoders/v_gene_one_hot_encoder.pkl", "rb") as f:
            v_gene_one_hot_encoder = pkl.load(f)
        with open("./encoders/j_gene_one_hot_encoder.pkl", "rb") as f:
            j_gene_one_hot_encoder = pkl.load(f)
        if data["CDR3"].str.contains(user_data).any():
            # select one row only with user data
            row = data[data["CDR3"] == user_data]
            # add row at end 
            data = data.append(row)
            X_test = data[['CDR3_al_integer', 'CDR3_al_one_hot', 'v_gene', 'j_gene', 'v_gene_one_hot', 'j_gene_one_hot']]
            X_test.reset_index(drop=True, inplace=True)
            for k in X_test.index:
                integer_encoded_v = v_gene_encoder.transform([X_test.loc[k, 'v_gene']]) 
                integer_encoded_j = j_gene_encoder.transform([X_test.loc[k, 'j_gene']])
                X_test.loc[k, 'v_gene_one_hot'] = v_gene_one_hot_encoder.transform(integer_encoded_v.reshape(-1,1))
                X_test.loc[k, 'j_gene_one_hot'] = j_gene_one_hot_encoder.transform(integer_encoded_j.reshape(-1,1))
                X_test.loc[k, 'CDR3_al_integer'] = np.array(X_test.loc[k, 'CDR3_al_integer'])
                X_test.loc[k, 'CDR3_al_one_hot'] = onehot_encoder.transform(X_test.loc[k, 'CDR3_al_integer'].reshape(len(X_test.loc[k, 'CDR3_al_integer']), 1))
            X_test["v_gene_one_hot"] = X_test["v_gene_one_hot"].apply(lambda x: x.reshape(54))
            X_test["j_gene_one_hot"] = X_test["j_gene_one_hot"].apply(lambda x: x.reshape(13))
            X_test = X_test[['CDR3_al_one_hot', 'v_gene_one_hot', 'j_gene_one_hot']]

            X_test = [np.stack(col.values) for _, col in X_test.items()]
            # select last N rows 
            for i in range(len(X_test)):
                X_test[i] = X_test[i][-N:]
            sample = data[-N:]
        else:
            # encode user data
            user_data_integer = integer_encoder.transform(list(user_data))
            # add user data at end
            data = data.append({"CDR3": user_data, "CDR3_al_integer": user_data_integer, "v_gene": v_gene, "j_gene": j_gene, "v_gene_one_hot": v_gene_one_hot_encoder.transform(v_gene_encoder.transform([v_gene]).reshape(-1,1)), "j_gene_one_hot": j_gene_one_hot_encoder.transform(j_gene_encoder.transform([j_gene]).reshape(-1,1))}, ignore_index=True)
            X_test = data[['CDR3_al_integer', 'CDR3_al_one_hot', 'v_gene', 'j_gene', 'v_gene_one_hot', 'j_gene_one_hot']]
            X_test.reset_index(drop=True, inplace=True)
            for k in X_test.index:
                integer_encoded_v = v_gene_encoder.transform([X_test.loc[k, 'v_gene']]) 
                integer_encoded_j = j_gene_encoder.transform([X_test.loc[k, 'j_gene']])
                X_test.loc[k, 'v_gene_one_hot'] = v_gene_one_hot_encoder.transform(integer_encoded_v.reshape(-1,1))
                X_test.loc[k, 'j_gene_one_hot'] = j_gene_one_hot_encoder.transform(integer_encoded_j.reshape(-1,1))
                X_test.loc[k, 'CDR3_al_integer'] = np.array(X_test.loc[k, 'CDR3_al_integer'])
                X_test.loc[k, 'CDR3_al_one_hot'] = onehot_encoder.transform(X_test.loc[k, 'CDR3_al_integer'].reshape(len(X_test.loc[k, 'CDR3_al_integer']), 1))
            X_test["v_gene_one_hot"] = X_test["v_gene_one_hot"].apply(lambda x: x.reshape(54))
            X_test["j_gene_one_hot"] = X_test["j_gene_one_hot"].apply(lambda x: x.reshape(13))
            X_test = X_test[['CDR3_al_one_hot', 'v_gene_one_hot', 'j_gene_one_hot']]

            X_test = [np.stack(col.values) for _, col in X_test.items()]
            # select last N rows 
            for i in range(len(X_test)):
                X_test[i] = X_test[i][-N:]
            sample = data[-N:]

    elif model_name == "Transformers (TCR-BERT)":
        user_data = user_data.replace("-", "").replace("*", "").replace(" ", "")
        if data["CDR3"].str.contains(user_data).any():
            # select one row only with user data
            row = data[data["CDR3"] == user_data]
            # add row at end 
            data = data.append(row)
            data["CDR3"] = data["CDR3"].apply(lambda x: " ".join(x))
            X_test = list(data["CDR3"])
            # select last N rows 
            X_test = X_test[-N:]
            sample = data[-N:]
        else:
            data = data.append({"CDR3": user_data}, ignore_index=True)
            data["CDR3"] = data["CDR3"].apply(lambda x: " ".join(x))
            X_test = list(data["CDR3"])
            X_test = X_test[-N:]
            sample = data[-N:]


    return X_test, sample

def plot_clusters(data, prediction, model_name):
    # Plot clusters using UMAP for dimensionality reduction
    reducer = umap.UMAP(random_state=42)
    print(prediction)
    print(prediction.shape)
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
    


def build_vae():
    from keras import backend as K
    from tensorflow.keras.layers import Dense, Reshape, Activation, Input, Lambda
    from vampire_custom_keras import BetaWarmup, EmbedViaMatrix
    import keras
    from keras import regularizers
    from tensorflow.keras.models import Model
    params = {
            "latent_dim": 50,
            "dense_nodes": 75,
            "aa_embedding_dim": 21,
            "v_gene_embedding_dim": 54,
            "j_gene_embedding_dim": 13,
            "beta": 0.2,
            "max_cdr3_len": 20,
            "n_aas": 21,
            "n_v_genes": 54,
            "n_j_genes": 13,
            "stopping_monitor": "val_loss",
            "batch_size": 50,
            "pretrains": 2,
            "warmup_period": 3,
            "epochs": 4,
            "patience": 20,
            "n_inputs" : 20,
            "v_inputs" : 54,
            "j_inputs" : 13
        }

    beta = K.variable(params['beta'])

    def sampling(args):
        """
        This function draws a sample from the multivariate normal defined by
        the latent variables.
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(params['batch_size'], params['latent_dim']), mean=0.0, stddev=1.0)
        print(epsilon.shape)
        # Reparameterization trick!
        return (z_mean + K.exp(z_log_var / 2) * epsilon)

    def vae_cdr3_loss(io_encoder, io_decoder):
        """
        The loss function is the sum of the cross-entropy and KL divergence. KL
        gets a weight of beta.
        """
        # Here we multiply by the number of sites, so that we have a
        # total loss across the sites rather than a mean loss.
        xent_loss = params['max_cdr3_len'] * K.mean(losses.categorical_crossentropy(io_encoder, io_decoder))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss *= beta
        return (xent_loss + kl_loss)

    # Input:
    cdr3_input_shape = (params['max_cdr3_len'], params['n_aas'])
    cdr3_input = Input(shape=cdr3_input_shape, name='cdr3_input')
    v_gene_input = Input(shape=(params['n_v_genes'], ), name='v_gene_input')
    j_gene_input = Input(shape=(params['n_j_genes'], ), name='j_gene_input')

    # Encoding layers:
    cdr3_embedding = EmbedViaMatrix(params['aa_embedding_dim'], name='cdr3_embedding')(cdr3_input)
    cdr3_embedding_flat = Reshape([params['aa_embedding_dim'] * params['max_cdr3_len']],
                                  name='cdr3_embedding_flat')(cdr3_embedding)
    v_gene_embedding = Dense(params['v_gene_embedding_dim'], name='v_gene_embedding')(v_gene_input)
    j_gene_embedding = Dense(params['j_gene_embedding_dim'], name='j_gene_embedding')(j_gene_input)
    merged_embedding = keras.layers.concatenate([cdr3_embedding_flat, v_gene_embedding, j_gene_embedding],
                                                name='merged_embedding')
    encoder_dense_1 = Dense(params['dense_nodes'], activation='elu', kernel_regularizer=regularizers.l2(0.001), name='encoder_dense_1')(merged_embedding)
    encoder_dense_2 = Dense(params['dense_nodes'], activation='elu', kernel_regularizer=regularizers.l2(0.001), name='encoder_dense_2')(encoder_dense_1)
    encoder_dense_3 = Dense(params['dense_nodes'], activation='elu', kernel_regularizer=regularizers.l2(0.001), name='encoder_dense_1')(encoder_dense_2)
    encoder_dense_4 = Dense(params['dense_nodes'], activation='elu', kernel_regularizer=regularizers.l2(0.001), name='encoder_dense_2')(encoder_dense_3)
    # Latent layers:
    z_mean = Dense(params['latent_dim'], name='z_mean')(encoder_dense_2)
    z_log_var = Dense(params['latent_dim'], name='z_log_var')(encoder_dense_2)

    # Decoding layers:
    z_l = Lambda(sampling, output_shape=(params['latent_dim'], ), name='z')
    decoder_dense_1_l = Dense(params['dense_nodes'], activation='elu', name='decoder_dense_1')
    decoder_dense_2_l = Dense(params['dense_nodes'], activation='elu', name='decoder_dense_2')
    cdr3_post_dense_flat_l = Dense(np.array(cdr3_input_shape).prod(), activation='linear', name='cdr3_post_dense_flat')
    cdr3_post_dense_reshape_l = Reshape(cdr3_input_shape, name='cdr3_post_dense')
    cdr3_output_l = Activation(activation='softmax', name='cdr3_output')
    v_gene_output_l = Dense(params['n_v_genes'], activation='softmax', name='v_gene_output')
    j_gene_output_l = Dense(params['n_j_genes'], activation='softmax', name='j_gene_output')

    post_decoder = decoder_dense_2_l(decoder_dense_1_l(z_l([z_mean, z_log_var])))
    cdr3_output = cdr3_output_l(cdr3_post_dense_reshape_l(cdr3_post_dense_flat_l(post_decoder)))
    v_gene_output = v_gene_output_l(post_decoder)
    j_gene_output = j_gene_output_l(post_decoder)

    # Define the decoder components separately so we can have it as its own model.
    z_mean_input = Input(shape=(params['latent_dim'], ))
    decoder_post_decoder = decoder_dense_2_l(decoder_dense_1_l(z_mean_input))
    decoder_cdr3_output = cdr3_output_l(cdr3_post_dense_reshape_l(cdr3_post_dense_flat_l(decoder_post_decoder)))
    decoder_v_gene_output = v_gene_output_l(decoder_post_decoder)
    decoder_j_gene_output = j_gene_output_l(decoder_post_decoder)

    encoder = Model([cdr3_input, v_gene_input, j_gene_input], [z_mean, z_log_var])
    decoder = Model(z_mean_input, [decoder_cdr3_output, decoder_v_gene_output, decoder_j_gene_output])
    vae = Model([cdr3_input, v_gene_input, j_gene_input], [cdr3_output, v_gene_output, j_gene_output])
    vae.compile(
        optimizer="adam",
        loss={
            'cdr3_output': vae_cdr3_loss,
            'v_gene_output': keras.losses.categorical_crossentropy,
            'j_gene_output': keras.losses.categorical_crossentropy,
        },
        loss_weights={
            # Keep the cdr3_output weight to be 1. The weights are relative
            # anyhow, and buried inside the vae_cdr3_loss is a beta weight that
            # determines how much weight the KL loss has. If we keep this
            # weight as 1 then we can interpret beta in a straightforward way.
            "cdr3_output": 1,
            "j_gene_output": 0.1305,
            "v_gene_output": 0.8138
        })

    callbacks = [BetaWarmup(beta, params['beta'], params['warmup_period'])]

    return {'encoder': encoder, 'decoder': decoder, 'vae': vae, 'callbacks': callbacks}