import sys
import numpy as np
import tensorflow as tf
import utils
import pandas as pd
import getopt
import time
import pickle

from scipy import sparse as sp

import gnn_models
import gnn_layers

DATA_PATH = "../cora/"
def printUsage():
    print("python run_experiment.py -c|--conv_layers=<number of convolutional layers> -h|--hidden_units=<number of units in dense layers> -d|--dropout_rate=<dropout rate>")

def main(argv):

    # Fetch command line arguments
    try:
        opts, args = getopt.getopt(argv,"c:h:d:",["conv_layers=", "hidden_units=", "dropout_rate="])
    except getopt.GetoptError:
        printUsage()
        sys.exit(1)
    for opt, arg in opts:
        if opt in ("-c", "--conv_layers"):
            CONV_LAYERS = int(arg)
        elif opt in ("-h", "--hidden_units"):
            HIDDEN_UNITS = int(arg)
        elif opt in ("-d", "--dropout_rate"):
            DROPOUT_RATE = float(arg)
    
    # Load preprocessed data
    with open(DATA_PATH + "papers.pkl", "rb") as f:
        papers = pickle.load(f)
    with open(DATA_PATH + "graph_info.pkl", "rb") as f:
        graph_info = pickle.load(f)

    edges = graph_info[1]
    x=np.array(edges).T
    # Sort the edges for faster aggregation (by using segment_sum instead of unsorted_segment_sum)
    # This is only used for the segment_sum-based model.
    new_edges = x[x[:, 0].argsort()].T
    graph_info = (graph_info[0], new_edges)

    # Build an adjacency matrix encoding the input graph
    A = utils.build_adj_matrix(x, graph_info[0].shape[0], symmetrize=True, normalize=True)

    # High-resolution timestamp to identify this run. This allows us to identify experiments with consistent setups.
    exp_id = int(time.time()*1e6)
    NUM_CLASSES = 7
    REPS = 10
    test_accs = []
    times = []
    for i in range(REPS):

        starting_time = time.time()
        print("Iteration " + str(i))

        x_train, x_test, y_train, y_test = utils.get_data_splits(papers)

        print("Train set size: {}".format(x_train.shape[0]))
        print("Test set size: {}".format(x_test.shape[0]))


        learning_rate = 0.01
        num_epochs = 1000
        batch_size = 256

        gnn_model = gnn_models.GNNModelMat(
            graph_info=graph_info,
            A=A,
            num_classes=NUM_CLASSES,
            conv_layers=CONV_LAYERS,
            hidden_units=HIDDEN_UNITS,
            dropout_rate=DROPOUT_RATE,
            name="gnn_model",
        )


        
        gnn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
            run_eagerly=False
        )
        # Create an early stopping callback.
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)


        
        timetaken = utils.timecallback()
        history = gnn_model.fit(
            x=x_train,
            y=y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[early_stopping, timetaken],
        )

        _, test_accuracy = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
        print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
        test_accs.append(test_accuracy)

        print(gnn_model.summary())

        
        total_time = time.time() - starting_time
        times.append(total_time)

        # We use a single file for accuracies, and separate files to store the training history
        with open("results/results.txt", "a") as out:
            out.write("{},{},{},{},{}\n".format(exp_id, CONV_LAYERS, HIDDEN_UNITS, DROPOUT_RATE, test_accuracy, total_time))

    print("="*50)
    print("Results")
    print("="*50)
    print("Mean (st. dev.) acc. on the test set: {} ({})".format(np.mean(test_accs), np.std(test_accs)))
        


    # Write results to file.
    with open("results/history_{}.pkl".format(exp_id), "wb") as out:
        pickle.dump(history, out)
        
if __name__ == "__main__":
   main(sys.argv[1:])
