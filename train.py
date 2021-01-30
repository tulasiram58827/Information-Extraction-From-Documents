import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import json
import wandb
from pathlib import Path
from tqdm import tqdm

from model import Model
from utils import neighbor_sep


key_path = Path('data/key/')
preprocessed_path = Path('data/new_processed_files')
models_path = Path('data/models/')
models_path.mkdir(exist_ok=True)

# Set CPU as available physical device
tf.config.set_visible_devices([], 'GPU')

use_wandb = False

if use_wandb:
    wandb.init(project="information_extraction")

field_id_money = 0
field_id_date = 1
#TODO read vocab size from somewhere instead of hard-coding
vocab_size = 512
# This comes from TextVectorizer's default value (used in data_processing.py) and shouldn't be changed.
# It allows us to use keras.Embedding mask_zero flag, and to compute a mask to give the attention mechanism.
padding_val = 0

model = Model(vocab_size=vocab_size, emb_dim=128, num_heads=8,
              num_fields=4, num_neighbors=30, padding_val=padding_val)

# temp_fieldid = tf.random.uniform((4, 1), dtype=tf.int64, minval=0, maxval=3)
# temp_cand_pos = tf.random.uniform((4, 2), dtype=tf.float32, minval=0, maxval=1)

# temp_textemb = tf.random.uniform((4, 10, 1), dtype=tf.int64, minval=0, maxval=512-1)
# temp_posemb = tf.random.uniform((4, 10, 2), dtype=tf.float32, minval=0, maxval=1)

# sample_output = sample_model(temp_fieldid, temp_cand_pos,
# 	                         temp_textemb, temp_posemb, None)
# print(temp_fieldid.shape, temp_cand_pos.shape, temp_textemb.shape, temp_posemb.shape)
# print(sample_output)

batch_size = 16

epochs = 100

loss_function = tf.keras.losses.BinaryCrossentropy()

# Optimizer Rectified Adam(from paper)
optimizer = tfa.optimizers.RectifiedAdam(0.001)

def generate_data():
    for csv_file in preprocessed_path.glob('*.csv'):
        key_file_path = f'{key_path / csv_file.stem}.json'
        with open(key_file_path, 'r') as f:
            key_dict = json.loads(f.read())
        data = pd.read_csv(csv_file)
        for idx, row in data.iterrows():
            field_id = field_id_date if row['total_candidate'] == 'NA' else field_id_money
            ground_truth = 0
            if key_dict['total'].replace('RM','').replace('$','').strip() == str(row['total_candidate']) or\
                key_dict['date'] == str(row['date_candidate']):
                ground_truth = 1
            try:
                cand_pos = [(row['xmax']-row['xmin'])/2, (row['ymax']-row['ymin'])/2]
            except KeyError:
                print(f"Coordinates missing from {csv_file}")
                break
            # Skip any without neighbors
            try:
                row['neighbors'].split(neighbor_sep)
            except AttributeError:
                continue
            neighbor_textid = row['neighbors'].split(neighbor_sep)[:30]
            # neighbor_textid = list(filter(lambda a: a != '', neighbor_textid))[:50]
            neighbor_pos = row['neigh_pos'].split(neighbor_sep)[:30]
            # neighbor_pos = list(filter(lambda a: a != '', neighbor_pos))[:50]
            new_pos = []
            for val in neighbor_pos:
                out = val.replace('(', '').replace(')', '').split(',')
                new_pos.append(out)
            yield field_id, cand_pos, neighbor_textid, new_pos, ground_truth

# Read dataset into list first, so we have the size
dataset_list = list(generate_data())
dataset = tf.data.Dataset.from_generator(lambda: dataset_list,
                                         output_types=(tf.int32, tf.float32, tf.int32, tf.float32, tf.int32),
                                         output_shapes=((), (2,), (None,), (None, 2), ()))

dataset = dataset.cache()
dataset = dataset.prefetch(buffer_size=100)
# Note this will use zero as the padding value, consistent with padding_val above
dataset = dataset.padded_batch(batch_size)

train_loss = tf.keras.metrics.Mean(name='train_loss')

def train_step(input_vals):
    (field_id, cand_pos, neigh_text, neigh_pos, gt) = input_vals
    field_id = tf.expand_dims(field_id, axis=-1)
    neigh_text = tf.expand_dims(neigh_text, axis=-1)
    # print(field_id.shape, cand_pos.shape, neigh_text.shape, neigh_pos.shape)
    with tf.GradientTape() as tape:
        score = model(field_id, cand_pos, neigh_text, neigh_pos)
        score = (score+1)/2
        loss_value = loss_function([gt], [score])
    gradients = tape.gradient(loss_value, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss_value)

for epoch in range(epochs):
    for (batch, input_vals) in tqdm(enumerate(dataset), desc=f"Epoch {epoch}", total=(len(dataset_list) // batch_size)):
        train_step(input_vals)

        if batch%50 == 0:
            if use_wandb:
                wandb.log({'loss': train_loss.result()})
            print(f'Epoch {epoch} Batch {batch} Loss {train_loss.result()}')

    if (epoch+1)%10 == 0:
        # Save the weights
        model.save_weights(models_path / f'checkpoint_{epoch}')
