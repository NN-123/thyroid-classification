import tensorflow as tf
from utils import load
import copy


class Labeler:
    def __init__(self, data):
        self._input = data
        self._data = self._input
        self._labeled = None

    def filter(self, predicate, incremental=False):
        if not incremental:
            self._data = list(filter(predicate, self._input))
        else:
            self._data = list(filter(predicate, self._data))
        return self

    def binary_label(self, predicate, preserve_original=False):
        def map_func(item):
            if predicate(item):
                item['label'] = True
            else:
                item['label'] = False
            return item

        if preserve_original:
            self._labeled = copy.deepcopy(self._data)
        else:
            self._labeled = self._data

        self._labeled = list(map(map_func, self._labeled))
        return self

    def get(self, labeled=True):
        if labeled:
            if self._labeled is None:
                raise Exception('Data unlabeled. Try calling labelling methods')
            return self._labeled
        else:
            return self._data
