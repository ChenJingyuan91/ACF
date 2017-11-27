__author__ = 'jingyuan'
from New_Dataset import *
from Model import *
from Test_Dataset import *

dataset = "Vine"

train_file = ""
test_file = ""

splitter = "\t"
hold_k_out = 1
batch_size = 256

num_epoch = 400
dim = 128

trainset = New_Dataset(train_file, splitter, batch_size)
testset = Test_Dataset(test_file, splitter,batch_size,trainset)

lr = 0.01
reg = 0.01

print("Load data (%s) done." %(dataset))

model = Model(trainset,testset,trainset.num_user, trainset.num_item, dim, reg, lr, None)
model.train(trainset.epoch)
model.save('./model/'+dataset+'/0_')
print '****************************************************************************'


print '****************************************************************************'
print '\n'
for i in xrange(1,num_epoch):
	model.train(trainset.epoch)
	print '****************************************************************************'
	#print 'test',i+1
	#newresult=model.test()
	print '****************************************************************************'
	#print '\n'
	#if newresult[0]>result[0] :
	#	result=newresult
	model.save('./model/'+dataset+'/luck1/'+str(i)+'_')
print 'bestmodel saved!'
