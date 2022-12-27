import numpy as np
import tensorflow as tf
import scipy as sp
import time
import pandas as pd

def sp_coo_to_sparse_tensor(coo):
    print(coo.indices, coo.data, coo.shape)
    return tf.SparseTensor(coo.indices, coo.data, coo.shape)


def compute_graph_laplacian(A, normalize=False):
    d = np.sum(A, axis=1)
    D = np.diag(d)
    L = D-A
    if normalize:
        sdi = np.sqrt(1/d)
        sdi[np.isnan(sdi)] = 0
        sdi[np.isinf(sdi)] = 0
        Di = np.diag(sdi)
        L = Di.dot(L).dot(Di)
    return L
    

def build_adj_matrix(edges, N, symmetrize=False, normalize=False, mformat="tf"):
    print(edges.shape, N)
    if symmetrize:
        edges = np.vstack([edges, edges[:,[1,0]]])
        edges = np.unique(edges, axis=0)
        
    A = sp.sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
                  shape=(N,N), dtype=np.float32)

    if normalize:        
        d = 1/np.sqrt(np.squeeze(np.array(sp.sum(A, axis=1))))
        D = sp.sparse.diags(d, 0) 
        A = sp.dot(D, sp.dot(A,D))
    if mformat == "numpy":
        return np.array(A.todense())
    elif mformat == "tf":
        return tf.SparseTensor(edges, A.data, (N,N))


def get_data_splits(papers):
    NUM_CLASSES = 7
    train_data, test_data = [], []

    for _, group_data in papers.groupby("subject"):
        # Select around 50% of the dataset for training.
        random_selection = np.random.rand(len(group_data.index)) <= 0.5
        train_data.append(group_data[random_selection])
        test_data.append(group_data[~random_selection])

    train_data = pd.concat(train_data).sample(frac=1)
    test_data = pd.concat(test_data).sample(frac=1)

    class_values = sorted(papers["subject"].unique())
    class_idx = {name: id for id, name in enumerate(class_values)}
    
    feature_names = set(papers.columns) - {"paper_id", "subject"}
    num_features = len(feature_names)
    num_classes = len(class_idx)
    
    # Create train and test features as a numpy array.
    x_train = train_data[feature_names].to_numpy()
    x_test = test_data[feature_names].to_numpy()
    # Create train and test targets as a numpy array.
    y_train = train_data["subject"]
    y_test = test_data["subject"]

    x_train = train_data.paper_id.to_numpy()
    x_test = test_data.paper_id.to_numpy()
    y_train = tf.one_hot(train_data["subject"], NUM_CLASSES)
    y_test = tf.one_hot(test_data["subject"], NUM_CLASSES)
    
    return x_train, x_test, y_train, y_test
    

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.process_time()
    def on_epoch_begin(self,epoch,logs = {}):
        self.timetaken = time.process_time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append(time.process_time() - self.timetaken)
    def on_train_end(self,logs = {}):
        print('Time per epoch (mean, st.dev.): {}, {}'.format(np.mean(self.times), np.std(self.times)))



