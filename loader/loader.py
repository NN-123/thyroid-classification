import xml.etree.ElementTree as et
import os

path_to_data = os.path.join('..', 'data')
path_to_xml = os.path.join(path_to_data, 'xml')
path_to_jpg = os.path.join(path_to_data, 'jpg')
fields = ['number', 'age', 'sex', 'composition',
          'echogenicity', 'margins', 'calcifications', 'tirads',
          'reportbacaf', 'reportbacaf']

filenames = os.listdir(path_to_xml)
filenames.sort()

forest = []
for filename in filenames:
    tree = et.parse(os.path.join(path_to_xml, filename))
    root = tree.getroot()
    temp = {}
    for field in fields:
        temp[field] = root.find(field).text
    forest.append(temp)

print(forest)
