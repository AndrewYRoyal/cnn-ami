#!/usr/bin/env python

from time import time
from datetime import timedelta
import yaml
import pickle
from utils import AMI
import argparse
import os
import boto3

parser = argparse.ArgumentParser('Train end-use classification model to AMI data.')
parser.add_argument('--cfg', dest='cfg', default='cfg.yaml', action='store')
args = parser.parse_args()
cfg = yaml.load(open(args.cfg))
# cfg = yaml.load(open('cfg.yaml')) #<< use interactively
startTime = time()

if(not os.path.exists('input')):
    os.mkdir('input')
if(not os.path.exists('output')):
    os.mkdir('output')
if(cfg['options']['s3']['use_s3']):
    try:
        s3 = boto3.client('s3')
        for k, f in cfg['input'].items():
            s3.download_file(cfg['options']['s3']['import'], f, 'input/' + f)
        print('Input data imported successfully.')
    except:
        sys.exit('Error: s3 data not found.')


print('Loading data...')
ami = AMI(cfg)
series_data = ami.load_data()
series_data = ami.subset(series_data, ['train', 'holdout'])

print('Training Model...')
classifier, kernels = ami.train(series_data['train'])

print('Class recall:')
holdout_predictions = ami.predict(series_data['holdout'], classifier, kernels) .\
    merge(ami.class_assignment.replace(ami.classes), on='id')
recall = ami.evaluate(holdout_predictions)
print(recall)

# Export
#========================================
filehandler = open('output/' + cfg['output']['model'], 'wb')
pickle.dump({'classifier': classifier, 'kernels': kernels}, filehandler)
elapse = timedelta(seconds=time() - startTime)
print('Time elapsed: {}'.format(elapse))

