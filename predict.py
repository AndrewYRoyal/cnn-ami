#!/usr/bin/env python

from time import time
from datetime import timedelta
import yaml
import pickle
from utils import transform_series, AMI
import argparse
import pandas as pd

parser = argparse.ArgumentParser('Predict meter end-use using AMI data.')
parser.add_argument('--cfg', dest='cfg', default='cfg.yaml', action='store')
args = parser.parse_args()

cfg = yaml.load(open(args.cfg))
# cfg = yaml.load(open('cfg.yaml')) #<< use interactively

startTime = time()
print('Loading Data...')

ami = AMI(cfg)
series_dat = ami.load_data()
filehandler = open(cfg['output']['model'], 'rb')
model = pickle.load(filehandler)

pred = ami.predict(model, series_dat)

out_dat = ami.class_dat.replace(cfg['class_index']).sort_index()
out_dat['prediction'] = pred

# TODO: export class predictions and probabilities

elapse = timedelta(seconds=time() - startTime)

print('Time elapsed: {}'.format(elapse))