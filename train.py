#!/usr/bin/env python

from time import time
from datetime import timedelta
import yaml
import pickle
from utils import transform_series, AMI
import argparse

parser = argparse.ArgumentParser('Train end-use classification model to AMI data.')
parser.add_argument('--cfg', dest='cfg', default='cfg.yaml', action='store')
args = parser.parse_args()

cfg = yaml.load(open(args.cfg))
# cfg = yaml.load(open('cfg.yaml')) #<< use interactively

startTime = time()

print('Loading data...')

ami = AMI(cfg)
series_dat = ami.load_data()

print('Training Model...')
model = ami.train(series_dat)


filehandler = open(cfg['output']['model'], 'wb')
pickle.dump(model, filehandler)

elapse = timedelta(seconds=time() - startTime)

print('Time elapsed: {}'.format(elapse))

##############
# filehandler = open(cfg['paths']['model'], 'rb')
# model_read = pickle.load(filehandler)
# conf_dat = pd.DataFrame({'actual': class_array, 'prediction': predictions})
# conf_dat.groupby('actual').apply(lambda x: np.mean(x.actual == x.prediction))
# conf_dat.groupby('actual').apply(lambda x: np.mean(x.actual != x.prediction))
