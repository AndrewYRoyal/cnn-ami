#!/bin/sh
aws s3 sync ./output/ s3://lightgbm-output/ --delete
