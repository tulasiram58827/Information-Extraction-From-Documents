import io
import sys
import dateparser
import pandas as pd

from datetime import date as today_date
from enum import Enum
from dateutil.parser import parse
from google.cloud import vision


# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ubuntu/.gcloud_ap_aiml_config.json'

client = vision.ImageAnnotatorClient()

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5

final_dates = list()

def get_document_bounds(path, filename):
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    document = response.full_text_annotation
    texts = response.text_annotations
    bounds = list()
    text_output = list()
    paragraph_bounds = list()
    paragraph_texts = list()
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                paragraph_text = ''
                for word in paragraph.words:
                    v = word.bounding_box.vertices
                    # print(word)
                    bounds.append([[v[0].x, v[0].y], [v[1].x, v[1].y], [v[2].x, v[2].y], [v[3].x, v[3].y]])
                    word_text = ''
                    for symbol in word.symbols:
                        word_text += symbol.text
                    paragraph_text += word_text + ' '
                paragraph_text = paragraph_text.replace(' - ', '-').replace(' / ', '/')
                y = paragraph.bounding_box.vertices
                paragraph_bounds.append([[y[0].x, y[0].y], [y[1].x, y[1].y], [y[2].x, y[2].y], [y[3].x, y[3].y]])
                paragraph_texts.append(paragraph_text)
    for text in texts:
        text_output.append(text.description)
    df_list = []

    for bound, text in zip(bounds, text_output[1:]):
        df_list.append([min(bound[0][0], bound[3][0]), min(bound[0][1], bound[1][1]), max(bound[1][0], bound[2][0]), max(bound[2][1], bound[3][1]), text])
    for bound, text in zip(paragraph_bounds, paragraph_texts):
        df_list.append([min(bound[0][0], bound[3][0]), min(bound[0][1], bound[1][1]), max(bound[1][0], bound[2][0]), max(bound[2][1], bound[3][1]), text])
    df = pd.DataFrame(df_list,columns=['xmin', 'ymin', 'xmax', 'ymax', 'Object'])
    df.to_csv(filename)    

def is_number_tryexcept(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def normalize_dates(dates):
    today = today_date.today()
    today_month_first = str(today.strftime("%Y-%m-%d"))
    today_day_first = str(today.strftime("%Y-%d-%m"))
    new_dates = list()
    for date in dates:
        if date is None or date == today_month_first or date == today_day_first:
            continue
        try:
            d = parse(str(date), dayfirst=True)
            new_dates.append(d.strftime("%d/%m/%Y"))
        except (ValueError, OverflowError):
            continue
    return new_dates

def remove_duplicates(nodes):
    seen = set()
    seen_add = seen.add
    return [x for x in nodes if not (x in seen or seen_add(x))]

def is_date(string):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    # string = string.replace('.', '').replace(':', '')
    splitted = string.split(' ')
    # print(string, splitted)
    dates = []
    splitted.append(string)
    for val in splitted:
        try: 
            output = parse(val)
            # print(output, val)
            if output is None or is_number_tryexcept(val) or len(val) < 6:
                continue
            # print(output.date())
            dates.append(str(output.date()))
        except:
            continue
    final_dates.extend(dates)



if __name__ == '__main__':
    image_path = sys.argv[1]
    output_filepath = image_path.split('.')[0]+'.csv'
    get_document_bounds(image_path, 'temp.csv')
    data = pd.read_csv('temp.csv')
    data['Object'].apply(is_date)
    normalized  = normalize_dates(final_dates)
    print(remove_duplicates(normalized))