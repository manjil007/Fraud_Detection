#!/usr/bin/env bash

# file: combine_data.sh
# author: Manjil Pradhan
#date: 03/07/2024

#Description: This file merges the fraud and not fraud date into a single csv file.

fa=is_fraud_overSampled.csv
fb=train_isnot_fraud.csv
fc=balancedTrain.csv

rm -f $fc
cat $fa >> $fc
cat $fb >> $fc

echo "status created new file $fc"