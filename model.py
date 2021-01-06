import tensorflow as tf 

from layers import pos_embedding, MultiHeadAttention



class Model(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, num_heads, num_fields, num_neighbors):
        super(Model, self).__init__()
        # Text embedding layer for neighbors
        self.text_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dim)
        # Embedding layer for relative positions of neighbors and candidate pos embedding.
        self.pos_emb = pos_embedding(dim=emb_dim)
        # Field ID Embedding Layer
        self.field_emb = tf.keras.layers.Embedding(input_dim=num_fields, output_dim=emb_dim)
        # Self Attention layer
        self.mha = MultiHeadAttention(2*emb_dim, num_heads)
        # Linear Projection Layer for Neighborhood Encoding+Candidate Pos Embedding
        self.projection = tf.keras.layers.Dense(emb_dim)
        # Max pooling layer for neighborhood embedding
        self.max_pool = tf.keras.layers.MaxPool1D(strides=num_neighbors, padding='same')
        # Cosine Similarity
        self.cosine_sim = tf.keras.losses.CosineSimilarity()

    def call(self, field_id, cand_pos, neighbors_text, neighbors_pos, mask):
        # Value of the field_id must be less than or equal to  num_fields-1
        field_emb = self.field_emb(field_id) # Sample Input -> [[2], [1], [0]] # (batch_size, 1) # Output dim (batch_size, 1, emb_dim)

        cand_emb = self.pos_emb(cand_pos)# Sample Input -> [[0.2, 0.3], [0.4, 0.5]] #(batch_size, 2) # Output dim (batch_size, emb_dim)

        text_emb = self.text_emb(neighbors_text) # Input Size -> (batch_size, N, 1)  # Output dim (batch_size, N, emb_dim)

        pos_emb = self.pos_emb(neighbors_pos) # Input Size -> (batch_size, N, 2) # Output dim (batch_size, N, emb_dim)
        
        concat_vector = tf.concat([tf.squeeze(text_emb), pos_emb], axis=-1) #(batch_size, N, 2*emb_dim)

        att_vector, weights = self.mha(concat_vector, concat_vector, concat_vector, mask) # Output dim (batch_dim, N , 2*emb_dim)

        neighbor_enc = self.max_pool(att_vector) # Output dim (batch_dim, 1, 2*emb_dim)

        candidate_enc = self.projection(tf.concat([cand_emb, tf.squeeze(neighbor_enc)], axis=-1)) #(batch_size, 2, emb_dim)

        sim_value = self.cosine_sim(candidate_enc, field_emb)
        return sim_value



       




