import tensorflow as tf 

from layers import pos_embedding, MultiHeadAttention



class Model(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, num_heads, num_fields, num_neighbors, padding_val):
        super(Model, self).__init__()
        # Text embedding layer for neighbors
        self.text_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True)
        # Embedding Layer for candidate positions
        self.cand_pos_emb = tf.keras.layers.Dense(emb_dim)
        # Embedding layer for relative positions of neighbors.
        self.neigh_pos_emb = pos_embedding(dim=emb_dim)
        # Field ID Embedding Layer
        self.field_emb = tf.keras.layers.Embedding(input_dim=num_fields, output_dim=emb_dim, mask_zero=False)
        # Self Attention layer
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(2*emb_dim, self.num_heads)
        # Linear Projection Layer for Neighborhood Encoding+Candidate Pos Embedding
        self.projection = tf.keras.layers.Dense(emb_dim)
        # Max pooling layer for neighborhood embedding
        self.max_pool = tf.keras.layers.MaxPool1D(strides=num_neighbors, padding='same')
        # Cosine Similarity
        self.cosine_sim = tf.keras.losses.CosineSimilarity(axis=1, reduction='none')

        self.padding_val = padding_val

    def call(self, field_id, cand_pos, neighbors_text, neighbors_pos):
        # Value of the field_id must be less than or equal to  num_fields-1
        field_emb = self.field_emb(field_id) # Sample Input -> [[2], [1], [0]] # (batch_size, 1) # Output dim (batch_size, 1, emb_dim)

        cand_emb = self.cand_pos_emb(cand_pos)# Sample Input -> [[0.2, 0.3], [0.4, 0.5]] #(batch_size, 2) # Output dim (batch_size, emb_dim)

        text_emb = self.text_emb(neighbors_text) # Input Size -> (batch_size, N, 1)  # Output dim (batch_size, N, emb_dim)

        #TODO how to apply mask here?
        pos_emb = self.neigh_pos_emb(neighbors_pos) # Input Size -> (batch_size, N, 2) # Output dim (batch_size, N, emb_dim)
        
        concat_vector = tf.concat([tf.squeeze(text_emb), pos_emb], axis=-1) #(batch_size, N, 2*emb_dim)

        # We can't trust that zeros in the neighbors pos are masked values vs genuine zeros, so use neighbor text token
        # values to make a mask for the attention mechanism. 1 means ignore, 0 means keep.
        # Shape (batch_size, N, 1)
        mask = tf.dtypes.cast(tf.dtypes.cast(tf.math.equal(neighbors_text, self.padding_val), tf.bool), tf.float32)
        mask = tf.squeeze(mask)     # (batch_size, N)
        # Each mask is 1d, but we want to turn that into a matrix by crossing against itself
        # For example instead of [0, 0, 1, 1], we want:
        # array([[0., 0., 0., 0.],
        #        [0., 0., 0., 0.],
        #        [0., 0., 1., 1.],
        #        [0., 0., 1., 1.]])
        mask = tf.map_fn(lambda instance_mask: tf.tensordot(instance_mask, instance_mask, axes=0), elems=mask)
        mask = tf.expand_dims(mask, 1)      # (batch_dim, 1, N, N)
        mask = tf.repeat(mask, self.num_heads, axis=1)      # (batch_dim, num_heads, N, N)
        att_vector, weights = self.mha(concat_vector, concat_vector, concat_vector, mask) # Output dim (batch_dim, N , 2*emb_dim)

        neighbor_enc = self.max_pool(att_vector) # Output dim (batch_dim, 1, 2*emb_dim)

        candidate_enc = self.projection(tf.concat([cand_emb, tf.squeeze(neighbor_enc)], axis=-1)) #(batch_size, 2, emb_dim)

        sim_values = self.cosine_sim(candidate_enc, tf.squeeze(field_emb))
        return sim_values



       




