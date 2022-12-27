"""
This script evaluates a multilayer perceptron on the given dataset.
A number of hyperparameters are chosen randomly.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import utils

from sklearn.model_selection import train_test_split

# This little helper allows us to avoid some repetition later on
def pop_label(data, label="subject", features_to_remove=["paper_id"]):
    return data.drop(axis="columns", labels=[label] + features_to_remove).values, pd.get_dummies(data[label]).values


DATA_PATH = "../cora/"

# Load preprocessed data
with open(DATA_PATH + "papers.pkl", "rb") as f:
    papers = pickle.load(f)
with open(DATA_PATH + "graph_info.pkl", "rb") as f:
    graph_info = pickle.load(f)

vectors, labels = pop_label(papers)

# Hyperparameters
h_use_lap = [True, False]
h_powers = [0,1,2,3]
h_mp_powers = [0,1,2,3]
h_svectors = range(10,1000)
h_n_layers = [1,2,3,4]
h_units = range(32,512)

from numpy.linalg import svd

# Main loop to search over the hyperparameter space
while True:
    use_lap = np.random.choice(h_use_lap)
    powers = np.random.choice(h_powers)
    mp_powers = np.random.choice(h_mp_powers)
    svectors = np.random.choice(h_svectors)
    n_layers = np.random.choice(h_n_layers)
    units = []
    skip_connections = []
    for _ in range(n_layers):
        units.append(np.random.choice(h_units))
        skip_connections.append(np.random.choice([False, True]))
        
    print("Configuration:")
    print(f"Use Laplacian: {use_lap}")
    print(f"Adj. powers: {powers}")
    print(f"Message passing order: {mp_powers}")
    print(f"Singular vectors: {svectors}")
    print(f"Number of layers: {n_layers}")
    print(f"Layers: {units}")
    print(f"Skip connections: {skip_connections}")

    # Build the adjacency matrix
    edges = graph_info[1]
    x=np.array(edges).T
    A = utils.build_adj_matrix(x, graph_info[0].shape[0], symmetrize=True, normalize=True, mformat="numpy")

    # SVD of data
    U,d,Vt = svd(vectors)
    vectors = U[:,:svectors]
    
    # Laplacian
    Uk = A
    if use_lap:
        L = utils.compute_graph_laplacian(A, normalize=True)
        Uk = L
        U,d,Vt = svd(L)
        Uk = U[:,:svectors]
        
    graph_features = Uk

    # Adjacency matrix
    Uk = A
    U,d,Vt = svd(A)
    Uk = U[:,:svectors]
    graph_features = np.hstack([graph_features, Uk])

    # Adjacency matrix powers
    Ap = np.copy(A)
    for _ in range(powers):
        Ap = Ap.dot(A)
        Uk = Ap
        U,d,Vt = svd(Ap)
        Uk = U[:,:svectors]
        graph_features = np.hstack([graph_features, Uk])

    # Message-passing
    U,d,Vt = svd(papers.values)
    Uk = U[:,:svectors]
    Uk = papers.values
    mp = A.dot(Uk)    
    for _ in range(mp_powers):
        graph_features = np.hstack([graph_features, mp])
        mp = A.dot(mp)
    
    print(f"Graph features: {graph_features.shape}")

    # Incorporate all built features into the data matrix
    X = np.hstack([vectors, graph_features])

    # Train-test split
    TRAINING_SIZE = 0.5
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(X, labels, train_size=TRAINING_SIZE, random_state=42)
                                   
    print(f"Train:{train_vectors.shape}")
    print(f"Test:{test_vectors.shape}")

    print("Shapes")
    print(train_vectors.shape, train_labels.shape, test_vectors.shape, test_labels.shape)
    
    inputs = tf.keras.Input(shape=(train_vectors.shape[1],))
    x = inputs
    x = tf.keras.layers.Dense(units[0], activation='relu')(x)
    skip = x
    x = tf.keras.layers.Dropout(0.5)(x)
    print("skip", skip)
    i=1
    for u in units[1:]:
        x = tf.keras.layers.Dense(u, activation='gelu')(x)
        if skip_connections[i]:
            x = tf.keras.layers.Concatenate()([x, skip])
        x = tf.keras.layers.Dropout(0.5)(x)
    print("skipafter", skip)

    outputs = tf.keras.layers.Dense(7)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=np.sum(units), restore_best_weights=True)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(
        train_vectors, 
        train_labels, 
        epochs=1000, 
        validation_split=0.15, 
        batch_size=256,
        callbacks=[early_stopping])

    test_loss, test_acc = model.evaluate(test_vectors,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    with open("results.csv", "a") as out:
        out.write(f"{use_lap},{powers},{mp_powers},{svectors},{units},{test_acc}\n")
    
