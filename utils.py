import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from rocket import generate_kernels, apply_kernels

def df_to_array(df):
    return np.asarray(
        df.reset_index(). \
        pivot(index='id', columns='time', values='use'). \
        sort_index())

class AMI:
    def __init__(self, config):
        self.input = config['input']
        self.classes = config['class_index']
        self.options = config['options']
    def load_data(self):
        series_dat, class_dat, split_dat = [pd.read_csv('input/' + p).set_index('id').sort_index() for p in self.input.values()]
        if (self.options['sample']):
            print('Taking 20% subsample of data.')
            split_dat = split_dat.sample(frac=0.20)
        class_dat = class_dat.merge(split_dat, on='id').drop(columns='group')
        series_dat = series_dat.merge(split_dat, on='id').drop(columns='group')

        class_match = np.all(np.sort(series_dat.index.unique()) == np.sort(class_dat.index.unique()))
        split_match = np.all(np.sort(series_dat.index.unique()) == np.sort(split_dat.index.unique()))
        if (not class_match):
            print('Warning: Class index does not match series index!')
        if (not split_match):
            print('Warning: Splits index does not match series index!')
        self.split_assignment = split_dat.sort_index()
        self.class_assignment = class_dat.sort_index()
        return series_dat

    def train(self, df):
        id_index = list(df.index.unique())
        y_train = self.class_assignment.loc[id_index]. \
            replace(self.classes). \
            sort_index(). \
            astype(np.int32)

        X_train = df_to_array(df)
        kernels = generate_kernels(X_train.shape[-1], 10_000)
        X_train_transform = apply_kernels(X_train, kernels)

        classifier = CalibratedClassifierCV(
            RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True))
        classifier.fit(X_train_transform, y_train)
        return classifier, kernels

    def predict(self, df, classifier, kernels):
        id_index = df.index.unique()
        X_test = df_to_array(df)
        X_test_transform = apply_kernels(X_test, kernels)
        pred_array = classifier.predict(X_test_transform)
        prob_array = classifier.predict_proba(X_test_transform)
        pred_dat = pd.concat([
            pd.DataFrame(prob_array, columns=self.classes),
            pd.DataFrame(pred_array, columns=['pred'])], axis=1). \
            set_index(id_index)
        return pred_dat

    def subset(self, df, subgroup):
        df_group = df.merge(
            self.split_assignment[self.split_assignment['group'].isin(subgroup)],  # << change to self.split_assignment
            on='id'). \
            groupby('group', group_keys=False)
        return {k: df.drop(columns='group') for k, df in df_group}

    def evaluate(self, df):
        recall = df.groupby('class'). \
            apply(lambda x: np.round(100 * np.mean(x['class'] == x['pred']), 4)). \
            to_dict()
        classes = list(self.classes.keys())
        return [f'{classes[np.int(k)]} : {value}' for k, value in recall.items()]