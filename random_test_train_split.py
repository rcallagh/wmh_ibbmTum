#!/usr/bin/env python

from os.path import join
import argparse
import glob
from random import shuffle

parser = argparse.ArgumentParser(description='HDF5-Creation')

parser.add_argument('--csv_name', type=str, default="subjects",
                        help='path and base name of csv file. Outputs will be csv_name_train.csv and csv_name_test.csv (default: subjects)')
parser.add_argument('--data_dir', type=str, default="./testsuite", help="Directory with images to load")
parser.add_argument('--pattern', type=str, default="*", help="Pattern to match files in directory. (default: *)")
parser.add_argument('--test_frac', type=float, default=0.2, help="Fraction of subjects to be held in training set. (default: 0.2)")


args = parser.parse_args()

search_pattern = join(args.data_dir, args.pattern)
subject_dirs = glob.glob(search_pattern)

num_subject = len(subject_dirs)

num_test = int(args.test_frac * num_subject)
num_train = num_subject - num_test

print('Num Test: ', num_test)
print('Num Train: ', num_train)

#Shuffle list for random order
shuffle(subject_dirs)

test_name = args.csv_name + '_test.csv'
train_name = args.csv_name + '_train.csv'

f = open(test_name, 'w')
for i in range(num_test):
    f.write(subject_dirs[i])
    f.write('\n')
f.close()

f = open(train_name, 'w')
for i in range(num_train):
    f.write(subject_dirs[num_test + i])
    f.write('\n')
f.close()
