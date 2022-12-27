import tensorflow as tf
from tensorflow.keras import layers

def create_ffn(hidden_units, dropout_rate, name=None):
    dense_layers = []

    for units in hidden_units:
        dense_layers.append(layers.BatchNormalization())
        dense_layers.append(layers.Dropout(dropout_rate))
        dense_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return tf.keras.Sequential(dense_layers, name=name)

class GraphConvLayerMat(layers.Layer):
    """
    A graph convolutional network layer based on the work of Kipf and Welling (2016).
    """
    def __init__(
        self,
        hidden_units,
        graph_info,
        dropout_rate=0.2,
        *args,
        **kwargs,
    ):
        super(GraphConvLayerMat, self).__init__(*args, **kwargs)
        self.dense_out = create_ffn([hidden_units], dropout_rate)

    def call(self, inputs):
        """
        inputs: a tuple of two elements: H (node features), A (sparse adjacency matrix)
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """
        H, A = inputs        
        h = tf.sparse.sparse_dense_matmul(A, H)                
        return self.dense_out(h)               


class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        graph_info,
        dropout_rate=0.2,
        normalize=False,
        *args,
        **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)
        self.normalize = normalize

        self.node_features, self.edges = graph_info
        
        self.update_fn = create_ffn([hidden_units], dropout_rate)
            
    def call(self, inputs):
        """
        inputs: a tuple of three elements: H, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        H, edges = inputs
        
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(H, neighbour_indices)
                        
        neighbour_messages = neighbour_repesentations
        
        num_nodes = self.node_features.shape[0]
        aggregated_messages = tf.math.segment_sum(neighbour_messages, node_indices)
        
        h = H + aggregated_messages
        node_embeddings = self.update_fn(h)
        node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)

        return node_embeddings

