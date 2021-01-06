import tensorflow as tf
import tensorflow_addons as tfa


from model import Model


sample_model = Model(vocab_size=512, emb_dim=128, num_heads=8,
	                 num_fields=4, num_neighbors=10)

temp_fieldid = tf.random.uniform((16, 1), dtype=tf.int64, minval=0, maxval=3)
temp_cand_pos = tf.random.uniform((16, 2), dtype=tf.float32, minval=0, maxval=1)

temp_textemb = tf.random.uniform((16, 10, 1), dtype=tf.int64, minval=0, maxval=512-1)
temp_posemb = tf.random.uniform((16, 10, 2), dtype=tf.float32, minval=0, maxval=1)

sample_output = sample_model(temp_fieldid, temp_cand_pos,
	                         temp_textemb, temp_posemb, None)

print(sample_output)




# Optimizer Rectified Adam(from paper)
optimizer = tfa.optimizers.RectifiedAdam(0.001)

# Loss function Binary Cross Entropy
loss = tf.keras.losses.BinaryCrossentropy()
