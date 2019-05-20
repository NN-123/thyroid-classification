import os
import xml.etree.ElementTree as et
from skimage import io


class Loader:
    default_fields = ['number', 'age', 'sex', 'composition',
                      'echogenicity', 'margins', 'calcifications', 'tirads']

    def __init__(self, path='data', extension='jpg', fields=None):
        self.path = path
        self.extension = extension
        if fields is None:
            self.fields = self.default_fields

    def initialize_xml_fields(self, fields):
        self.fields = fields

    def load(self, xml_folder='xml', image_folder='image'):
        image_path = os.path.join(self.path, image_folder)
        xml_path = os.path.join(self.path, xml_folder)

        xml_files = os.listdir(xml_path)
        image_files = os.listdir(image_path)

        cases = []

        for xml_file in xml_files:
            tree = et.parse(os.path.join(xml_path, xml_file))
            root = tree.getroot()
            case = {}
            for field in self.fields:
                case[field] = root.find(field).text
            case_image_files = list(filter(lambda x: x.startswith(str(case['number']) + '_'), image_files))
            for image_file in case_image_files:
                image = io.imread(os.path.join(image_path, image_file))
                case['image'] = image
                cases.append(case)
        print('Successfully imported {} images from {}'.format(len(cases), image_path))
        return cases
