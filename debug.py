__author__ = 'jingyuan'
from New_Dataset import *
import multiprocessing as mp
from Model import *
from Test_Dataset import *

dataset = "Pinterest"

train_file = "/home/jie/sigir/data/train_v3"
test_file = "/home/jie/sigir/data/test_v3"

splitter = "\t"
hold_k_out = 1
batch_size = 512
learning_rate = 0.01
num_epoch = 400
dim = 128
trainset = New_Dataset(train_file, splitter, batch_size)
users, pos_items, neg_items,mask,feats = trainset.get_batch(24)
print mask.shape
print users.shape
print feats.shape
users, pos_items, neg_items,mask,feats = trainset.get_batch(25)
print mask.shape
print users.shape
print feats.shape
