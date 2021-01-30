import re

from dateutil.parser import parse

neighbor_sep = '  '


# Candidate Generator to check a numeric value
def is_number_tryexcept(s):
    s = str(s)
    s = s.replace('$', '').replace(',', '').replace('RM', '')
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_date(value):
    if is_number_tryexcept(value) or len(value) < 6:
        return False
    try:
        parse(value)
        return True
    except:
        return False
    
# def check_amount(en):
#     if "." in en and any([x.replace(".","",1).isdigit() for x in [en, en.split()[0], en.split()[-1], en[2:], en[:-2]]]):
#         return True
#     return False

# def check_numeric(en):
#     if en.isdigit() or bool(re.match("([a-z'\"]{0,2})(\d)+[-+,x/]*(\d)*", en)):
#         return True
#     return False

# def check_pad(en):
#     if bool(re.match("[\\*\\-,:\\.=()#_&><!]+", en)):
#         return True
#     return False

def check_social(value):
    if any([value.startswith(x) for x in ["facebook", "instagram", "linkedin", "http", "www", "twitter"]]):
        return True
    return False

# Reference : https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    try:
        iou = interArea/boxBArea
    except ZeroDivisionError:
        return None
    # return the intersection over union value
    return iou

# Generate Neighbors
def generate_neighbors(orig_csv_df, filt_csv_df, image_shape, layer):
    for idx, row in filt_csv_df.iterrows():
        co_ord = [0, row['ymin']-(10*image_shape[0]/100), row['xmax'], row['ymax']]
        # co_ord = [0, row['ymin']-(10*image_shape[0]/100), row['xmax'], row['ymax']]
        neighbors = list()
        neighbor_rows = list()
        neigh_pos = list()
        for new_idx, new_row in orig_csv_df.iterrows():
            if new_idx == idx or new_row['Object'] == row['Object']:
                continue
            new_co_ord = [new_row['xmin'], new_row['ymin'], new_row['xmax'], new_row['ymax']]
            area = bb_intersection_over_union(co_ord, new_co_ord)
            if area is None:
                return None
            elif area >= 0.5:
                neighbor_rows.append(new_co_ord)
                neighbors.append(apply_vocab(layer, str(new_row['Object'])))
                xx = (new_row['xmin']+new_row['xmax']) - (row['xmin']+row['xmax'])/2
                yy = (new_row['ymin']+new_row['ymax']) - (row['ymin']+row['ymax'])/2
                neigh_pos.append(f"({'%.3f' % (xx/image_shape[1])},{'%.3f' % (yy/image_shape[0])})")
        neighbors = neighbor_sep.join(str(i) for i in neighbors)
        neigh_pos = neighbor_sep.join(str(x) for x in neigh_pos)
        neighbor_rows = neighbor_sep.join(str(tuple(x)) for x in neighbor_rows)
        filt_csv_df.loc[idx, 'neighbors'] = neighbors
        filt_csv_df.loc[idx, 'neigh_pos'] = neigh_pos
        filt_csv_df.loc[idx, 'xmin'] = '%.3f' % (new_row['xmin']/image_shape[1])
        filt_csv_df.loc[idx, 'xmax'] = '%.3f' % (new_row['xmax']/image_shape[1])
        filt_csv_df.loc[idx, 'ymin'] = '%.3f' % (new_row['ymin']/image_shape[0])
        filt_csv_df.loc[idx, 'ymax'] = '%.3f' % (new_row['ymax']/image_shape[0])
        filt_csv_df.loc[idx, 'neighbor_rows'] = neighbor_rows
    return filt_csv_df

## Apply vocabulary for each word
def apply_vocab(layer, value):
    value = value.replace('dale', 'date')
    if is_date(value):
        return layer(['date_token']).numpy()[0][0]
    elif is_number_tryexcept(value):
        return layer(['numeric_token']).numpy()[0][0]
    # elif check_numeric(value):
    #     return layer(['numeric_token']).numpy()[0][0]
    # elif check_pad(value):
    #     return layer(['pad_token']).numpy()[0][0]
    elif check_social(value):
        return layer(['social_token']).numpy()[0][0]
    else:
        return layer([str(value)]).numpy()[0][0]


def calculate_recall(json_filepath, sample_csv_filtered):
    with open(json_filepath) as f:
        temp = f.read()
    data = json.loads(temp)
    date = data['date'].strip()
    amount = data['total'].replace('$', '').replace('RM', '').strip()

    total_objects = sample_csv_filtered['total_candidate'].tolist()
    date_objects = sample_csv_filtered['date_candidate'].tolist()
    total_objects = list(map(lambda x: x.replace(',', '').replace('$', '').replace('RM','').replace('NA', '-100000'), total_objects))
    out = []
    if date in date_objects:
        out.append(1)
    else:
        out.append(0)
        # print(date, date_objects, file_name)
    if amount in total_objects:
        out.append(1)
    else:
        out.append(0)
    return out

def check_neighbors(sample_image, row, neighbors):
    image_shape = cv2.imread(sample_image).shape
    cv2.rectangle(sample_image, (int(float(row["xmin"])*image_shape[1]), int(float(row["ymin"])*image_shape[0])), 
              (int(float(row["xmax"])*image_shape[1]), int(float(row["ymax"])*image_shape[0])), 
              (0, 255, 0), 2)
    for neighbor in neighbors:
        neighbor = ast.literal_eval(neighbor)
        cv2.rectangle(sample_image, (int(neighbor[0]), int(neighbor[1])), (int(neighbor[2]), int(neighbor[3])), (255, 0, 0), 2)
    cv2.imwrite('tempp.jpg', sample_image)