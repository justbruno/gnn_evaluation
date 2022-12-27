import os
import pandas as pd
import tensorflow as tf
import pickle

DATA_PATH = "../"

# This script is based on the code provided by Khalid Salama at https://keras.io/examples/graph/gnn_citations/

citations = pd.read_csv(DATA_PATH + "cora.cites", sep="\t", header=None, names=["target", "source"])
print("Citations shape:", citations.shape)

column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers = pd.read_csv("cora/cora.content", sep="\t", header=None, index_col=0)
print("Papers shape:", papers.shape)

# Renumber ids from 0 to N
class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])

feature_names = set(papers.columns) - {"paper_id", "subject"}

# Create an edges array (sparse adjacency matrix) of shape [2, num_edges].
edges = citations[["source", "target"]].to_numpy().T
# Create a node features array of shape [num_nodes, num_features].
node_features = tf.cast(
    papers.sort_values("paper_id")[feature_names].to_numpy(), dtype=tf.dtypes.float32
)
# Create graph info tuple with node_features and edges.
graph_info = (node_features, edges)

print("Edges shape:", edges.shape)
print("Nodes shape:", node_features.shape)


with open(DATA_PATH + "graph_info.pkl", "wb") as out:
    pickle.dump(graph_info, out)
    
with open(DATA_PATH + "papers.pkl", "wb") as out:
    pickle.dump(papers, out)
