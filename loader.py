import xml.etree.ElementTree as et
import os
from skimage import io

path_to_data = os.path.join('.', 'data')
path_to_xml = os.path.join(path_to_data, 'xml')
path_to_jpg = os.path.join(path_to_data, 'preprocessed')
fields = ['number', 'age', 'sex', 'composition',
          'echogenicity', 'margins', 'calcifications', 'tirads']


def load():
    filenames = os.listdir(path_to_xml)
    filenames.sort()
    cases = []
    for filename in filenames:
        tree = et.parse(os.path.join(path_to_xml, filename))
        root = tree.getroot()

        case = {}

        for field in fields:
            case[field] = root.find(field).text
        for mark in root.findall('mark'):
            image_name = case['number'] + '_' + str(mark.find('image').text) + '.jpg'
            image = io.imread(os.path.join(path_to_jpg, image_name))
            case['image'] = image
            cases.append(case)
    return cases
