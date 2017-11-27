__author__ = 'jingyuan'
import numpy as np
from sets import Set
import h5py
import ast
from random import randint
import cPickle
import random
from copy import deepcopy

class New_Dataset(object):
    def __init__(self, filename, splitter, batch_size): #line formart: [usr]\t[i]\t[j]\t[list]\t
        self.maxbatch = batch_size

        lines = map(lambda x: x.strip().split(splitter), open(filename).readlines())
        self.usr = map(lambda line: line[0], lines)
        self.v_i = map(lambda line: line[1], lines)
        self.num_video_u = map(lambda line: int(line[2]), lines)

        self.num_user = len(set(self.usr))
        self.num_item = len(set(self.v_i))

        print 'num_user ',self.num_user
        print 'num_item ',self.num_item

        self.epoch = len(self.usr) / self.maxbatch
        if len(self.usr) % self.maxbatch != 0:
            self.epoch += 1

        self.video_features = self.load_frame_feat()
        self.frame_mask = self.load_frame_mask()

        self.u_list_map = self.load_u_list_map()

    def shuffle_data(self):
        c = list(zip(self.usr, self.v_i, self.num_video_u))
        random.shuffle(c)
        self.usr, self.v_i, self.num_video_u = zip(*c)

    def load_frame_feat(self):
#        frame_feat = cPickle.load(open('/home/jie/jingyuan/sigir/data/frame_feat.p','rb'))
        frame_feat = cPickle.load(open('/Users/jingyuan/Downloads/NSC-master/HAN/Data_preparing/frame_feat.p','rb'))
        return np.array(frame_feat,dtype='float32')

    def load_frame_mask(self):
#        frame_mask = cPickle.load(open('/home/jie/jingyuan/sigir/data/frame_mask.p','rb'))
        frame_mask = cPickle.load(open('/Users/jingyuan/Downloads/NSC-master/HAN/Data_preparing/frame_mask.p','rb'))
        return np.array(frame_mask,dtype='int32')

    def load_u_list_map(self):
#        u_list_map = cPickle.load(open('/home/jie/jingyuan/sigir/data/user_all_vines.p','rb'))
        u_list_map = cPickle.load(open('/Users/jingyuan/Downloads/NSC-master/HAN/Data_preparing/user_all_vines.p','rb'))
        return np.array(u_list_map)

    def gen_batch(self,datas):
        dd = deepcopy(datas)
        #max_item_count = len(dd[0])
        max_item_count = max(len(xx) for xx in dd)
        for uu in dd:
            for i in xrange(max_item_count-len(uu)):
                uu.append(-1)
 #           uu = np.array(uu,dtype='int32')
        return dd

    def genitemmask(self,itemnum):

        maxnum = max(xx for xx in itemnum)
        mask = np.asarray(map(lambda num:[1]*num + [0]*(maxnum-num),itemnum),dtype='int32')
        return mask

    def genvideofeature(self,temp_u_list):
        return self.video_features[temp_u_list]

    def genneg(self,user_list,temp):
        neg_items = []
        for i in xrange(len(user_list)):
            i_id = randint(0,self.num_item-1)
            while i_id in temp[i]:
                i_id = randint(0,self.num_item-1)
            neg_items.append(i_id)
        return neg_items

    def get_u_list(self,user_list):
        return self.u_list_map[user_list]

    def get_batch(self,i):

        user_list = self.usr[i*self.maxbatch:(i+1)*self.maxbatch]
        users = np.asarray(user_list,dtype=np.int32)
        pos_items = np.asarray(self.v_i[i*self.maxbatch:(i+1)*self.maxbatch],dtype=np.int32)
        temp = self.get_u_list(users)
        temp_u_list = self.gen_batch(temp) #[user,item] after append -1
        mask = self.genitemmask(self.num_video_u[i*self.maxbatch:(i+1)*self.maxbatch]) # similar as above
        neg_items = np.asarray(self.genneg(user_list,temp),dtype=np.int32)
        feats_idx = np.vstack(temp_u_list[:]).astype(np.int32) #object to int
        mask_frame = np.take(self.frame_mask,feats_idx,axis=0)

        return (users, pos_items, neg_items,mask_frame,mask,feats_idx)
