#!/usr/bin/env python

from time import time
from datetime import timedelta
import yaml
import pickle
from utils import AMI
import argparse
import pandas as pd
import subprocess

parser = argparse.ArgumentParser('Predict meter end-use using AMI data.')
parser.add_argument('--cfg', dest='cfg', default='cfg.yaml', action='store')
args = parser.parse_args()
cfg = yaml.load(open(args.cfg))
# cfg = yaml.load(open('cfg.yaml')) #<< use interactively

startTime = time()
print('Loading Data...')

ami = AMI(cfg)
series_dat = ami.load_data()
filehandler = open('output/' + cfg['output']['model'], 'rb')
model_bundle = pickle.load(filehandler)

print('Predicting:')
predictions = ami.predict(series_dat, model_bundle['classifier'], model_bundle['kernels'])

print('Exporting:')
predictions.to_csv('output/' + cfg['output']['predictions'])

if cfg['options']['s3']['use_s3']:
    try:
        subprocess.run(['./export_results.sh'])
        print('Results synced to s3.')
    except:
        print('Unable to sync results to s3 bucket.')


elapse = timedelta(seconds=time() - startTime)
print('Time elapsed: {}'.format(elapse))