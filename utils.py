import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sktime.transformations.panel.rocket import Rocket
import pandas as pd
from sktime.utils import data_io

def transform_series(series_dat, split_dat, column_names, subgroup = None):
    if(subgroup is None):
        series = {'X_long': series_dat.reset_index()}
    else:
        series = {
            'X_long':
                series_dat.merge(split_dat[split_dat.group == subgroup], on='id').reset_index()
        }
    series.update(column_names)
    series = data_io.from_long_to_nested(**series)
    rocket = Rocket()
    rocket.fit(series)

    return rocket.transform(series)

class AMI:
    def __init__(self, config):
        self.input = config['input']
        self.classes = config['class_index']
        self.options = config['options']
        self.column_names = config['column_names']
    def load_data(self):
        series_dat, class_dat, split_dat = [
            pd.read_csv(p).set_index('id').sort_index() for p in self.input.values()
        ]
        series_dat['dim_id'] = 1

        if (self.options['sample']):
            print('Taking 20% subsample of data.')
            class_dat = class_dat.sample(frac=0.20)
            split_dat = split_dat.merge(class_dat, on='id').drop(columns='class')
            series_dat = series_dat.merge(class_dat, on='id').drop(columns='class')

        self.class_dat = class_dat
        self.split_dat = split_dat

        class_match = np.all(np.sort(series_dat.index.unique()) == np.sort(class_dat.index.unique()))
        split_match = np.all(np.sort(series_dat.index.unique()) == np.sort(split_dat.index.unique()))
        if (not class_match):
            print('Warning: Class index does not match series index!')
        if (not split_match):
            print('Warning: Splits index does not match series index!')
        return series_dat
    def train(self, series_dat):
        series_train = transform_series(series_dat, self.split_dat, self.column_names, 'train')
        class_array = self.class_dat. \
            replace(self.classes). \
            merge(self.split_dat[self.split_dat['group'] == 'train'], on='id'). \
            drop(columns='group'). \
            sort_index(). \
            to_numpy()
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier.fit(series_train, class_array)
        return classifier

    def predict(self, model, series_dat):

        series_trans = transform_series(series_dat, self.split_dat, self.column_names, subgroup = None)
        return model.predict(series_trans)
