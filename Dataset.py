__author__ = 'jingyuan'
import numpy as np
from sets import Set
import h5py

class Dataset(object):
    def __init__(self, dataset, splitter, hold_k_out, batch_size, dim):
        train = []
        test = []
        filename = dataset
        K = hold_k_out
        self.dim = dim

        self.batch_size = batch_size

        self.num_ratings = 0
        self.num_item = 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(splitter)
                if (len(arr) < 4):
                    continue
                user, item, time = int(arr[0]), int(arr[1]), long(arr[3])
                if (len(train) <= user):
                    train.append([])
                train[user].append([item, time])
                self.num_ratings += 1
                self.num_item = max(item, self.num_item)
                line = f.readline()
        self.num_user = len(train)
        self.num_item = self.num_item + 1

    # sort ratings of each user by time
        def getTime(item):
            return item[-1];
        for u in range (len(train)):
            train[u] = sorted(train[u], key = getTime)

    # split into train/test
        maxlen= 0
        #minlen = 10000
        for u in range (len(train)):
            for k in range(K):
                if (len(train[u]) == 0):
                    break
                if len(train[u]) > maxlen:
                    maxlen = len(train[u])
                #if len(train[u]) < minlen:
                #    minlen = len(train[u])
                test.append([u, train[u][-1][0], train[u][-1][1]])
                del train[u][-1]    # delete the last element from train

    # sort the test ratings by time
        print maxlen
        #print minlen
        self.test = sorted(test, key = getTime)

        self.epoch = (self.num_ratings/batch_size) + 1

        self.train = train
        self.items_of_user = self.get_items_of_user()

        hf = h5py.File('data/image_data.h5','r')
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get('res5c')
        np_data = np.array(data)
        print('Shape of the array dataset_1: \n', np_data.shape) #(n,2048,7,7)
        num_video = np_data.shape[0]
        num_dim = np_data.shape[1]
        np_data = np_data.reshape(num_video,num_dim,-1)
        np_data = np.transpose(np_data,(0,2,1))
        self.video_features = np_data

        #self.test_u = []
        #self.test_i = []
        #self.test_j = []
        #self.test_items_u = []
        #for i in xrange(len(test)):
        #    self.test_u.append(test[i][0])
        #    self.test_i.append(test[i][1])
        #    self.test_j.append(test[i][2])
        #    self.test_items_u.append(self.items_of_user[test[i][0]])

    def prepare_data_for_epoch(self):
        users = []
        pos_items = []
        neg_items = []
        items_u = []
        for i in xrange(self.epoch):
            users_batch, pos_items_batch, neg_items_batch,items_u_batch = self.get_batch()
            users.append(users_batch)
            pos_items.append(pos_items_batch)
            neg_items.append(neg_items_batch)
            items_u.append(items_u_batch)
        return (users, pos_items, neg_items,items_u)
        #print("#users: %d, #items: %d, #ratings: %d" %(self.num_user, self.num_item, self.num_ratings))


    def get_items_of_user(self):
        items_of_user = np.zeros((self.num_user, self.num_item))

        for u in xrange(len(self.train)):
            #items_of_user.append(Set([]))
            for i in xrange(len(self.train[u])):
                item = self.train[u][i][0]
                items_of_user[u][item]=1
        return items_of_user

    def get_videos_u(self, item_u):
        feat_u = []
        item_idexes = list(np.where(item_u)[0])
        for index in item_idexes:
            feat_u.append(self.video_features[index])

    def get_batch(self):
        users_b, pos_items_b, neg_items_b, items_u, mask = [], [], [], [], []
        items_feature = []
        max_item = 0,
        for iii in xrange(self.batch_size):
            # sample a user
            u = np.random.randint(0, self.num_user)
            # sample a positive item
            i = self.train[u][np.random.randint(0, len(self.train[u]))][0]
            # sample a negative item
            j = np.random.randint(0, self.num_item)
            #while j in self.items_of_user[u]:
            while self.items_of_user[u][j]==1:
                j = np.random.randint(0, self.num_item)
            users_b.append(u)
            pos_items_b.append(i)
            neg_items_b.append(j)

            item_u = self.items_of_user[u]
            if max_item < len(np.where(item_u)[0]):
                max_item = len(np.where(item_u)[0])
            items_u.append(item_u)
            items_feature.append(self.get_videos_u(item_u))

        new_items_u = []

        for i in xrange(len(items_u)):
            mask_i = np.ones(max_item)
            item_idex = list(np.where(items_u[i])[0])
            while len(item_idex) < max_item:
                mask_i[len(item_idex)] = 0
                item_idex.append(self.num_item)
            new_items_u.append(item_idex)
            mask.append(mask_i)
        return (users_b, pos_items_b, neg_items_b,new_items_u,mask,items_feature)