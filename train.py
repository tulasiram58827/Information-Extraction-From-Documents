from argparse import ArgumentParser
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

field_id_money = 0
field_id_date = 1


def generate_data(max_neighbors):
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
            neighbor_textid = row['neighbors'].split(neighbor_sep)[:max_neighbors]
            neighbor_pos = row['neigh_pos'].split(neighbor_sep)[:max_neighbors]
            new_pos = []
            for val in neighbor_pos:
                out = val.replace('(', '').replace(')', '').split(',')
                new_pos.append(out)
            yield field_id, cand_pos, neighbor_textid, new_pos, ground_truth


def train_step(input_vals, model, optimizer, loss_function):
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

    return loss_value

def train(dataset, dataset_len, batch_size, epochs, model, optimizer, use_wandb):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    loss_function = tf.keras.losses.BinaryCrossentropy()
    for epoch in range(epochs):
        for (batch, input_vals) in tqdm(enumerate(dataset), desc=f"Epoch {epoch}", total=(dataset_len // batch_size)):
            loss = train_step(input_vals, model, optimizer, loss_function)
            train_loss(loss)

            if batch%50 == 0:
                if use_wandb:
                    wandb.log({'loss': train_loss.result()})
                print(f'Epoch {epoch} Batch {batch} Loss {train_loss.result()}')

        if (epoch+1)%10 == 0:
            # Save the weights
            model.save_weights(models_path / f'checkpoint_{epoch}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--skip_wandb', dest='skip_wandb', action='store_true',
                        help="To skip tracking experiments with weights&biases")
    parser.set_defaults(skip_wandb=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--emb_dim', type=int, default=128,
                        help="Embedding dimension. Used for both token IDs and position")
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--max_neighbors', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    print(vars(args))

    # Set CPU as available physical device by specifying no GPUs
    tf.config.set_visible_devices([], 'GPU')

    # Read dataset into list first, so we have the size
    dataset_list = list(generate_data(args.max_neighbors))
    dataset = tf.data.Dataset.from_generator(lambda: dataset_list,
                                             output_types=(tf.int32, tf.float32, tf.int32, tf.float32, tf.int32),
                                             output_shapes=((), (2,), (None,), (None, 2), ()))
    dataset = dataset.cache()
    #dataset = dataset.prefetch(buffer_size=100)
    # Note this will use zero as the padding value, consistent with padding_val above
    dataset = dataset.padded_batch(args.batch_size)

    if not args.skip_wandb:
        wandb.init(project="information_extraction")
        wandb.config.update(vars(args))

    # TODO read vocab size from somewhere instead of hard-coding
    vocab_size = 512
    # This comes from TextVectorizer's default value (used in data_processing.py) and shouldn't be changed.
    # It allows us to use keras.Embedding mask_zero flag, and to compute a mask to give the attention mechanism.
    padding_val = 0

    model = Model(vocab_size=vocab_size, emb_dim=args.emb_dim, num_heads=args.attention_heads,
                  num_fields=4, num_neighbors=args.max_neighbors, padding_val=padding_val)

    # Optimizer Rectified Adam(from paper)
    optimizer = tfa.optimizers.RectifiedAdam(args.lr)

    train(dataset, len(dataset_list), args.batch_size, args.epochs, model, optimizer, not args.skip_wandb)
