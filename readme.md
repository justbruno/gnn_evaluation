# Graph Neural Networks evaluation

Can we match the performance of a GNN with a simple multilayer perceptron?

This repository provides a sandbox to test this question. The code is currently tailored to the CORA citations dataset.

The script ``mlp_functional.py`` randomly searches over a space of hyperparameters for a simple MLP, including a wide array of features engineered from the input graph.

``gnn_models.py`` and ``gnn_layers.py`` provide implementations of the Kipf & Welling GCN model. See ``run_experiment.py`` for a usage example.
