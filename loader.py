import xml.etree.ElementTree as et
import os
import random
from skimage import io
import numpy as np
from tensorflow import keras

path_to_data = os.path.join('.', 'data')
path_to_xml = os.path.join(path_to_data, 'xml')
path_to_raw_jpg = os.path.join(path_to_data, 'raw_jpg')
path_to_preprocessed = os.path.join(path_to_data, 'preprocessed')
path_to_inception = os.path.join(path_to_data, 'inception')

fields = ['number', 'age', 'sex', 'composition',
          'echogenicity', 'margins', 'calcifications', 'tirads']


def load(mode='preprocessed'):
    path_to_images = path_to_preprocessed
    if mode == 'raw':
        path_to_images = path_to_raw_jpg
    if mode == 'inception':
        path_to_images = path_to_inception
    xml_filenames = os.listdir(path_to_xml)
    image_filenames = os.listdir(path_to_images)
    cases = []
    for filename in xml_filenames:
        tree = et.parse(os.path.join(path_to_xml, filename))
        root = tree.getroot()
        case = {}
        for field in fields:
            case[field] = root.find(field).text
        case_image_filenames = list(filter(lambda x: x.startswith(str(case['number']) + '_'), image_filenames))
        for image_filename in case_image_filenames:
            image = io.imread(os.path.join(path_to_images, image_filename))
            case['image'] = image
            cases.append(case)
    return cases


def label(data):
    data = list(filter(lambda x: x['tirads'] is not None, data))
    for item in data:
        item['label'] = False if item['tirads'] == '2' or item['tirads'] == '3' else True
    return data


def make_dataset(data):
    data = label(data)
    benign_cases = list(filter(lambda x: not x['label'], data))
    malign_cases = list(filter(lambda x: x['label'], data))
    random.shuffle(benign_cases)
    random.shuffle(malign_cases)

    benign_limit = int(len(benign_cases) * 0.1)
    malign_limit = int(len(malign_cases) * 0.1)
    train_benign = benign_cases[0: -benign_limit]
    test_benign = benign_cases[-benign_limit:]
    train_malign = malign_cases[0: -malign_limit]
    test_malign = malign_cases[-malign_limit:]
    train = train_benign + train_malign
    test = test_benign + test_malign
    random.shuffle(train)
    random.shuffle(test)

    train_labels = keras.utils.to_categorical([x['label'] for x in train], 2)
    train_images = np.array([x['image'] for x in train]).astype('float32') / 255
    test_labels = keras.utils.to_categorical([x['label'] for x in test], 2)
    test_images = np.array([x['image'] for x in test]).astype('float32') / 255

    return train_images, train_labels, test_images, test_labels
