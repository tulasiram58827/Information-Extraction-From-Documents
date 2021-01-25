import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import json
import wandb
from pathlib import Path

from model import Model

# Set CPU as available physical device
tf.config.set_visible_devices([], 'GPU')

wandb.init(project="information_extraction")


model = Model(vocab_size=512, emb_dim=128, num_heads=8,
	                 num_fields=4, num_neighbors=30)

# temp_fieldid = tf.random.uniform((4, 1), dtype=tf.int64, minval=0, maxval=3)
# temp_cand_pos = tf.random.uniform((4, 2), dtype=tf.float32, minval=0, maxval=1)

# temp_textemb = tf.random.uniform((4, 10, 1), dtype=tf.int64, minval=0, maxval=512-1)
# temp_posemb = tf.random.uniform((4, 10, 2), dtype=tf.float32, minval=0, maxval=1)

# sample_output = sample_model(temp_fieldid, temp_cand_pos,
# 	                         temp_textemb, temp_posemb, None)
# print(temp_fieldid.shape, temp_cand_pos.shape, temp_textemb.shape, temp_posemb.shape)
# print(sample_output)

batch_size = 16

Epochs = 50

loss_function = tf.keras.losses.BinaryCrossentropy()

# Optimizer Rectified Adam(from paper)
optimizer = tfa.optimizers.RectifiedAdam(0.001)

def generate_data():
    csv_files_path = Path('new_processed_files/')
    for csv_file in csv_files_path.glob('*.csv'):
        json_filename = str(csv_file).split('/')[-1]
        json_filename = json_filename[:-4]+".json"
        json_file_path = Path('/home/ram/Projects/OCR/ICDAR-2019-SROIE/data/key/')/json_filename
        with open(json_file_path) as f:
            temp = f.read()
        json_data = json.loads(temp)
        data = pd.read_csv(csv_file)
        for idx, row in data.iterrows():
            field_id = 0 #Assume it is date instance
            if row['total_candidate'] == 'NA':
                field_id = 1
            ground_truth = 0 # Assume gt is date
            if json_data['total'].replace('RM','').replace('$','').strip() == str(row['total_candidate']) or\
                json_data['date'] == str(row['date_candidate']):
                ground_truth = 1 # Change gt if it is total amount
            try:
                cand_pos = [(row['xmax']-row['xmin'])/2, (row['ymax']-row['ymin'])/2]
            except KeyError:
                print(csv_file)
                break
            try:
                row['neighbors'].split('  ')
            except AttributeError:
                continue
            neighbor_textid = row['neighbors'].split('  ')[:30]
            # neighbor_textid = list(filter(lambda a: a != '', neighbor_textid))[:50]
            neighbor_pos = row['neigh_pos'].split('  ')[:30]
            # neighbor_pos = list(filter(lambda a: a != '', neighbor_pos))[:50]
            new_pos = []
            for val in neighbor_pos:
                out = val.replace('(', '').replace(')', '').split(',')
                new_pos.append(out)
            yield field_id, cand_pos, neighbor_textid, new_pos, ground_truth

#Create dataset from generator
dataset = tf.data.Dataset.from_generator(generate_data, output_types=(tf.int32, tf.float32, tf.int32, tf.float32, tf.int32),
                                         output_shapes=((), (2,), (None,), (None, 2), ()))

dataset = dataset.cache()
dataset = dataset.prefetch(buffer_size=100)
dataset = dataset.padded_batch(batch_size)

train_loss = tf.keras.metrics.Mean(name='train_loss')

def train_step(input_vals):
    (field_id, cand_pos, neig_text, neigh_pos, gt) = input_vals
    field_id = tf.expand_dims(field_id, axis=-1)
    neig_text = tf.expand_dims(neig_text, axis=-1)
    # print(field_id.shape, cand_pos.shape, neig_text.shape, neigh_pos.shape)
    with tf.GradientTape() as tape:
        score = model(field_id, cand_pos, neig_text, neigh_pos)
        score = (score+1)/2
        loss_value = loss_function([gt], [score])
    gradients = tape.gradient(loss_value, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss_value)

for epoch in range(Epochs):
    for (batch, input_vals) in enumerate(dataset):
        
        train_step(input_vals)

        if batch%50 == 0:
            wandb.log({'loss': train_loss.result()})
            print(f'Epoch {epoch} Batch {batch} Loss {train_loss.result()}')

    if (epoch+1)%10 == 0:
        # Save the weights
        model.save_weights(f'checkpoint_{epoch}')
       