__author__ = 'jingyuan'
from New_Dataset import *
import multiprocessing as mp
from Model import *
from Test_Dataset import *

dataset = "Vine"

#train_file = "/home/jie/jingyuan/sigir/data/train_vine"
train_file = ""
test_file = ""

splitter = "\t"
hold_k_out = 1
batch_size = 256

num_epoch = 400
dim = 128

trainset = New_Dataset(train_file, splitter, batch_size)
testset = Test_Dataset(test_file, splitter,batch_size,trainset)

lr = 0.000001
reg = 0.01

print("Load data (%s) done." %(dataset))

model = Model(trainset,testset,trainset.num_user, trainset.num_item, dim, reg, lr, '/Users/jingyuan/86_')
print 'model loaded'
model.test(100)

