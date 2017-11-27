__author__ = 'jingyuan'
import numpy as np
import multiprocessing as mp
from Test_Dataset import *
from New_Dataset import *
import math
import heapq
_testRatings = None
_K = None


def test(model_folder,id_epoch,topK):


    test_file = "/home/jie/jingyuan/sigir/data/test_vine"
    splitter = "\t"
    testset = Test_Dataset(test_file, splitter)
    train_file = "/home/jie/jingyuan/sigir/data/train_vine"
    batch_size = 512
    global trainset
    trainset = New_Dataset(train_file, splitter, batch_size)

    f = file(model_folder + str(id_epoch)+ '_usremblayer.save', 'rb')
    global U_np
    U_np = np.asarray(cPickle.load(f).eval())

    f = file(model_folder + str(id_epoch) + '_videmblayer.save', 'rb')
    global V_np
    V_np = np.asarray(cPickle.load(f).eval())

    f = file(model_folder + str(id_epoch) + '_attentionlayer_feat.save', 'rb')
    global Wu_F_np
    Wu_F_np  = np.asarray(cPickle.load(f).eval())
    global Wv_F_np
    Wv_F_np = np.asarray(cPickle.load(f).eval())
    global b_F_np
    b_F_np = np.asarray(cPickle.load(f).eval())
    global c_F_np
    c_F_np = np.asarray(cPickle.load(f).eval())

    f = file(model_folder + str(id_epoch) + '_attentionlayer_item.save', 'rb')
    global Wu_I_np
    Wu_I_np  = np.asarray(cPickle.load(f).eval())
    global Wv_I_np
    Wv_I_np = np.asarray(cPickle.load(f).eval())
    global b_I_np
    b_I_np = np.asarray(cPickle.load(f).eval())
    global c_I_np
    c_I_np = np.asarray(cPickle.load(f).eval())

    global _testRatings
    _testRatings = testset.testRatings
    global _K
    _K = topK
    num_ratings = len(_testRatings)
    num_thread = mp.cpu_count()

    pool = mp.Pool(processes=num_thread)
    res = pool.map(eval_one_rating, range(num_ratings))
    pool.close()
    pool.join()

    hits = [r[0] for r in res]
    ndcgs = [r[1] for r in res]

    return np.array(hits).mean(), np.array(ndcgs).mean()

def eval_one_rating(idx):
    rating = _testRatings[idx]
    hr = ndcg = 0
    u = rating[0]
    gtItem = rating[1]
    map_item_score = {}

    maxScore, usr_vec = predict(u, gtItem)

    countLarger = 0

    for i in xrange(49358):
        early_stop = False
        score = predict_user(usr_vec,i)
        map_item_score[i] = score
        #print score
        if score > maxScore:
            countLarger += 1
        if countLarger > _K:
            hr = ndcg = 0
            early_stop = True
            break
    if early_stop == False:
        ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        ndcg = getNDCG(ranklist, gtItem)

    return (hr, ndcg)

def softmask(x):
    y = np.exp(x)
    sumx = np.sum(y,axis=0)
    x = y/sumx
    return x

def softmask_feat(atten, mask):
    atten = np.where(mask,atten,np.NINF)


def predict(user_idx, item):
    uu1 = U_np[user_idx]
    vv = V_np[item]
    index_u = trainset.u_list_map[user_idx]
    frame_feat = trainset.video_features[index_u]
    mask = trainset.frame_mask[index_u]
    #frame_level
    frame_feat_tran = np.dot(frame_feat, Wv_F_np)
    attenu = np.dot(uu1, Wu_F_np)
    attenu = np.reshape(attenu,(attenu[0],1,attenu[1]))
    attenu = frame_feat_tran + attenu + b_F_np
    atten = np.maximum(attenu, 0, attenu)
    atten = np.sum(atten * c_F_np, axis=2)
    atten = softmask_feat(atten, mask)

    temp1 = np.dot(vv_feat_tran, Wv_I_np)
    #print temp1.shape

    attenu = np.dot(uu1, Wu_I_np)
    #print attenu.shape


    temp2 = temp1 + attenu + b_I_np
    atten = temp2 * (temp2 > 0)
    atten = np.sum(atten*c_I_np,axis=1)
    atten = softmask(atten)
    #VV = atten.reshape(atten.shape[0],atten.shape[1],1) * vv_feat_tran

    #vv = self.V_np[item]
    #attenu = attenu.reshape(attenu.shape[0],1,attenu.shape[1])
    #temp1 = np.dot(VV, self.Wv_np)
    #temp2 = temp1 + attenu + self.b_np
    #atten = temp2 * (temp2 > 0)
    #atten = np.sum(atten*self.c_np,axis=1)
    #atten = softmask(atten)

    oo = atten.reshape(-1,1) * temp1
    uu2 = np.sum(oo, axis=0)
    #    print uu1.shape
#    print uu2.shape
    #print vv.shape
    u_vec = uu1 + uu2
#    print u_vec.shape

    return np.inner(u_vec, vv), u_vec

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def predict_user(user, item):
    u_vec = user
    vv = V_np[item]
    return np.inner(u_vec, vv)

hits,ndcgs = test(100)
print 'hits', hits
print 'ndcgs', ndcgs