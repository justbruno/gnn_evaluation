import tensorflow as tf
from tensorflow.keras import layers
import gnn_layers
import numpy as np
import utils

class GNNModelMat(tf.keras.Model):
    """
    A graph-neural network model for classification.
    """
    def __init__(
        self,
        graph_info,
        A,
        num_classes,
        conv_layers,
        hidden_units,
        dropout_rate=0.5,
        normalize=True,
        *args,
        **kwargs,
    ):
        super(GNNModelMat, self).__init__(*args, **kwargs)

        self.conv_layers = conv_layers
        self.hidden_units = hidden_units
        
        node_features, edges = graph_info
        self.node_features = node_features
        self.edges = edges
        self.A = A
        
        self.d1 = layers.Dense(self.hidden_units, activation=tf.nn.gelu)

        self.dropout1 = layers.Dropout(0.5)
        
        self.conv = []
        for i in range(self.conv_layers):
            self.conv.append(gnn_layers.GraphConvLayerMat(
                        hidden_units,
                        graph_info,
                        dropout_rate,
                    ))

        
        self.d2 = layers.Dense(self.hidden_units, activation=tf.nn.gelu)

        self.dropout2 = layers.Dropout(0.5)

        self.d3 = layers.Dense(units=num_classes, name="logits")
        
        
    def call(self, input_node_indices):
        # To build a reusable pipeline we could consider refactoring this model.
        # Since this is just an exercise, we lay out the model explicitly for easier scrutiny.
        x = self.d1(self.node_features)
        x = self.dropout1(x)
        
        for i in range(self.conv_layers):
            x1 = self.conv[i]((x, self.A))
            x = x1 + x # Skip connection                    
            
        x = self.d2(x)
        x = self.dropout2(x)

        # Fetch node embeddings for the input node_indices.
        node_embeddings = tf.gather(x, input_node_indices)
        return self.d3(node_embeddings)




class GNNModel(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        conv_layers,
        hidden_units,
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super(GNNModel, self).__init__(*args, **kwargs)

        self.conv_layers = conv_layers
        self.hidden_units = hidden_units
        
        node_features, edges = graph_info
        self.node_features = node_features
        self.edges = edges

        self.d1 = layers.Dense(self.hidden_units, activation=tf.nn.gelu)
        self.dropout1 = layers.Dropout(0.5)
        
        self.conv = []
        for i in range(self.conv_layers):
            self.conv.append(gnn_layers.GraphConvLayer(
                        hidden_units,
                        graph_info,
                        dropout_rate,
                    ))

        
        self.d2 = layers.Dense(self.hidden_units, activation=tf.nn.gelu)
        self.dropout2 = layers.Dropout(0.5)

        self.d3 = layers.Dense(units=num_classes, name="logits")
        
        
    def call(self, input_node_indices):
        x = self.d1(self.node_features)

        x = self.dropout1(x)
        
        for i in range(self.conv_layers):
            x1 = self.conv[i]((x, self.edges))
            x = x1 + x
            
        x = self.d2(x)
        x = self.dropout2(x)

        # Fetch node embeddings for the input node_indices.
        node_embeddings = tf.gather(x, input_node_indices)
        # Compute logits
        return self.d3(node_embeddings)

    
