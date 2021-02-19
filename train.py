from argparse import ArgumentParser
import math
import random
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import json
import wandb
from pathlib import Path
from tqdm import tqdm

from model import Model
from utils import neighbor_sep

# Set Random Seed
random.seed(127)

key_path = Path('data/key/')
preprocessed_path = Path('data/new_processed_files')
models_path = Path('data/models/')
models_path.mkdir(exist_ok=True)

field_id_money = 0
field_id_date = 1
# Ground thruth labels:
incorrect_id = 0
correct_id = 1


hyperparameter_defaults = dict(
    emb_dim = 128,
    attention_heads = 8,
    max_neighbors=30,
    lr = 0.001,
    epochs = 10,
    )


def generate_data(max_neighbors, files_list):
    for i, csv_file in enumerate(files_list):
        key_file_path = f'{key_path / csv_file.stem}.json'
        with open(key_file_path, 'r') as f:
            key_dict = json.loads(f.read())
        data = pd.read_csv(csv_file)
        for idx, row in data.iterrows():
            field_id = field_id_date if row['total_candidate'] == 'NA' else field_id_money
            ground_truth = incorrect_id
            if key_dict['total'].replace('RM','').replace('$','').strip() == str(row['total_candidate']) or\
                key_dict['date'] == str(row['date_candidate']):
                ground_truth = correct_id
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


def train_step(input_vals, model, optimizer, loss_function, debug):
    (field_id, cand_pos, neigh_text, neigh_pos, gt) = input_vals
    field_id = tf.expand_dims(field_id, axis=-1)
    neigh_text = tf.expand_dims(neigh_text, axis=-1)
    # print(field_id.shape, cand_pos.shape, neigh_text.shape, neigh_pos.shape)
    if debug:
        print(f"Ground truth: {gt}")
    with tf.GradientTape() as tape:
        scores = model(field_id, cand_pos, neigh_text, neigh_pos)
        scores = (scores+1)/2
        loss_value = loss_function([gt], [scores])
        if debug:
            print(f"Cosine sim scores: {scores}")
    gradients = tape.gradient(loss_value, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_value, gt, scores

def test_step(input_vals, model, loss_function, debug):
    (field_id, cand_pos, neigh_text, neigh_pos, gt) = input_vals
    field_id = tf.expand_dims(field_id, axis=-1)
    neigh_text = tf.expand_dims(neigh_text, axis=-1)
    if debug:
        print(f"Ground truth: {gt}")
    scores = model(field_id, cand_pos, neigh_text, neigh_pos)
    scores = (scores+1)/2
    loss_value = loss_function([gt], [scores])
    if debug:
        print(f"Cosine sim scores: {scores}")
    return loss_value, gt, scores

    

def train(dataset, test_data, n_instances, batch_size, epochs, model, optimizer, reports_per_epoch, use_wandb, debug):
    loss_function = tf.keras.losses.BinaryCrossentropy()
    batches_per_epoch = math.ceil(n_instances / batch_size)
    eval_indices = [round((i+1) * (batches_per_epoch / reports_per_epoch)) for i in range(reports_per_epoch)]

    for epoch in range(epochs):
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_acc = tf.keras.metrics.BinaryAccuracy(name='train_acc', threshold=0.5)
        train_prec = tf.keras.metrics.Precision(name='train_pos_prec')
        train_rec = tf.keras.metrics.Recall(name='train_pos_rec')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_prec = tf.keras.metrics.Precision(name='test_pos_prec')
        test_rec = tf.keras.metrics.Recall(name='test_pos_rec')

        pbar_desc = "Epoch %d avg loss [%.3f], acc [%.3f], prec/rec [%.3f/%.3f]"
        pbar = tqdm(enumerate(dataset),
                    desc=(pbar_desc % (epoch, 0.0, 0.0, 0.0, 0.0)),
                    total=(n_instances // batch_size))
        
        for (batch_i, input_vals) in pbar:
            loss, ground_truth, scores = train_step(input_vals, model, optimizer, loss_function, debug)

            train_loss(loss)
            train_acc.update_state(ground_truth, scores)
            train_prec.update_state(ground_truth, scores, sample_weight=(ground_truth == correct_id))
            train_rec.update_state(ground_truth, scores, sample_weight=(ground_truth == correct_id))
            pbar.set_description(
                pbar_desc % (epoch, train_loss.result(), train_acc.result(),
                             train_prec.result(), train_rec.result()))

            if batch_i in eval_indices:
                results_dict = dict()
                for metric in [train_loss, train_acc, train_prec, train_rec]:
                    results_dict[metric.name] = metric.result().numpy()
                    metric.reset_states()
                 
                print("Evaluating test data......")
                for i, batch_data in enumerate(test_data):
                    loss, gt, scores = test_step(batch_data, model, loss_function, debug)
                    test_loss(loss)
                    test_prec.update_state(gt, scores, sample_weight=(gt == correct_id))
                    test_rec.update_state(gt, scores, sample_weight=(gt == correct_id))
                print("Test data Evaluated.")

                test_results = dict()
                for metric in [test_loss, test_prec, test_rec]:
                    test_results[metric.name] = metric.result().numpy()
                    metric.reset_states()

                # TODO evaluate test set, add to results_dict
                #  instead of just measuring accuracy of each prediction, what is our accuracy when taking the highest
                #  scoring candidate for each (doc, field_id) pair? How often do we guess the right candidate per doc?

                if use_wandb:
                    wandb.log(results_dict)
                    wandb.log(test_results)
                print(f'\nEpoch {epoch} batch {batch_i}:\n{results_dict}')
                print(f'Test results: \n {test_results}')

        if (epoch+1)%10 == 0:
            # Save the weights
            model.save_weights(models_path / f'checkpoint_{epoch}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--skip_wandb', dest='skip_wandb', action='store_true',
                        help="To skip tracking experiments with weights&biases")
    parser.set_defaults(skip_wandb=False)
    parser.add_argument('--debug', dest='debug', action='store_true', help='tiny dataset, no wandb, more printing')
    parser.set_defaults(debug=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--emb_dim', type=int, default=128,
                        help="Embedding dimension. Used for both token IDs and position")
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--max_neighbors', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--reports_per_epoch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--test_size', type=int, default=0.1)
    args = parser.parse_args()

    # Set CPU as available physical device by specifying no GPUs
    #tf.config.set_visible_devices([], 'GPU')
    
    files = list(preprocessed_path.glob('*.csv'))
    train_size = int(len(files)*(1-args.test_size))

    random.shuffle(files)

    train_files = files[:train_size]
    test_files = files[train_size:]

    # Read dataset into list first, so we have the size
    train_dataset_list = list(generate_data(args.max_neighbors, train_files))
    test_dataset_list = list(generate_data(args.max_neighbors, test_files))
    print("Check......", len(train_dataset_list), len(test_dataset_list))
    if args.debug:
        args.skip_wandb = True
        args.batch_size = 8
        train_dataset_list = train_dataset_list[:32]
    train_dataset = tf.data.Dataset.from_generator(lambda: train_dataset_list,
                                             output_types=(tf.int32, tf.float32, tf.int32, tf.float32, tf.int32),
                                             output_shapes=((), (2,), (None,), (None, 2), ()))
    
    test_dataset = tf.data.Dataset.from_generator(lambda: test_dataset_list,
                                                 output_types=(tf.int32, tf.float32, tf.int32, tf.float32, tf.int32),
                                                 output_shapes=((), (2,), (None,), (None, 2), ()))

    train_dataset = train_dataset.cache()
    test_dataset = test_dataset.cache()

    # Note this will use zero as the padding value, consistent with padding_val below
    train_dataset = train_dataset.padded_batch(args.batch_size)
    test_dataset = test_dataset.padded_batch(args.batch_size)

    if not args.skip_wandb:
        wandb.init(config=hyperparameter_defaults, project="information_extraction")
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
    train(train_dataset, test_dataset, len(train_dataset_list), args.batch_size, args.epochs, model, optimizer,
          args.reports_per_epoch, not args.skip_wandb, args.debug)
