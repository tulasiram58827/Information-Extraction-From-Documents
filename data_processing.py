import cv2
import glob
import re
import json
import imutils
import numpy as np
from pathlib import Path

# https://stackoverflow.com/a/51855662
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from imutils import paths
#from google.colab.patches import cv2_imshow
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from utils import *

images_path = 'data/img/'
json_path = 'data/key/'
output_path = Path('data/new_processed_files')

images = sorted(list(paths.list_images(images_path)))
csv_files = sorted(glob.glob("data/ocr_modified_files/*.csv"))

def get_token(value):
    if is_date(value):
        return 'date_token'
    elif is_number_tryexcept(value):
        return 'amount_token'
    # elif check_numeric(en):
    #     data.append('numeric_token')
    # elif check_pad(en):
    #     data.append('pad_token')
    elif check_social(value):
        return 'social_token'
    else:
        return str(value)


# Create voabulary
#TODO separate [UNK] from [PAD]
data = []
for idx, entry in enumerate(csv_files):
    sample_csv = pd.read_csv(entry)
    for value in sample_csv['Object'].to_list():
        value = str(value)
        if len(value) < 2:
            continue
        for val in value.split():
            val = val.replace('dale', 'date')
            data.append(get_token(val))
        

vectorize_layer = TextVectorization(max_tokens=512, output_mode='int', output_sequence_length=1)

#Feed the data to layer
vectorize_layer.adapt(data)

vocab = vectorize_layer.get_vocabulary()
vocab ="\n".join(word for word in vocab)
with open('vocab_new.txt', 'w') as f:
    f.write(vocab)



#Filter dataframes using candidate generators

date_detected = 0
amount_detected = 0
count = 0

for file_idx, csv_file in enumerate(csv_files):
    file_name = csv_file.split('/')[-1]
    image_name = file_name.split('.')[0]+'.jpg'
    json_filename = file_name.split('.')[0]+'.json'
    image_path = images_path+image_name
    json_filepath = json_path+json_filename
    sample_csv = pd.read_csv(csv_file)
    filtered = []
    count+=1
    for idx, row in sample_csv.iterrows():
        new_row = list(row.copy())
        if is_date(row['Object']):
            new_row.append(str(row['Object']).strip())
        else:
            new_row.append('NA')
        if is_number_tryexcept(row['Object']):
            new_row.append(str(row['Object']).strip())
        else:
            new_row.append('NA')
        if new_row[-1] == 'NA' and new_row[-2] == 'NA':
            continue
        filtered.append(new_row)
    names = ['idx', 'xmin', 'ymin', 'xmax', 'ymax', 'Object', 'date_candidate', 'total_candidate']
    sample_csv_filtered = pd.DataFrame(filtered, columns=names)
    try:
        image_shape = cv2.imread(image_path).shape
    except AttributeError:
        count-=1
        continue

    # Caluclate Recall of candidate Generators
    # out = calculate_recall(json_filepath, sample_csv_filtered)
    # date_detected+=out[0]
    # amount_detected+=out[1]

    # Generate Neighbors
    new_df = generate_neighbors(sample_csv, sample_csv_filtered, image_shape, vectorize_layer)
    if new_df is None:
        continue

    # Check neighbors whether they are calculate correctly
    # After this call temp.jpg will be generated with candidate and neighbors drawn. Visualize it and confirm
    # sample_image = cv2.imread(image_path)
    # neighbors = new_df.iloc[5]['neighbor_rows'].split('  ')
    # row = new_df.iloc[5]
    # check_neighbors(sample_image, row, neighbors)
    
    #Write to csv file
    print(csv_file)
    output_path.mkdir(exist_ok=True)
    new_df.to_csv(output_path / file_name)
