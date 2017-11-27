__author__ = 'jingyuan'
import theano
import theano.tensor as T
from UsrEmblayer import *
from VidEmblayer import *
from GetuEmbLayer import *
from GetvEmbLayer import *
from AttentionLayer_Feat import *
from AttentionLayer_Item import *
from ContentEmbLayer import *
import numpy as np
import math

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

def softmask(x):
    y = np.exp(x)
    #y = y * mask
    sumx = np.sum(y,axis=0)
    #x = y/sumx.dimshuffle(0,'x')
    x=y/sumx
    return x

class Model(object):
    def __init__(self,trainset,testset,num_user,num_item,dim,reg,lr,prefix):
        self.trainset = trainset
        self.testset = testset
        self.reg = numpy.float32(reg)
        self.lr = numpy.float32(lr)
        self.num_item = num_item
        self.video_features = theano.shared(value=self.trainset.video_features, name='video_features', borrow=True)

        T.config.compute_test_value = 'warn'

        u = T.ivector('u') #[num_sample,]
        iv = T.ivector('iv') #[num_sample,]
        jv = T.ivector('jv') #[num_sample,]
        mask_frame = T.itensor3('mask_frame')  #[num_sample, num_video, num_frame]
        mask = T.imatrix('mask') #[num_sample, num_video]
        feat_idx = T.imatrix('feat_idx')  #[num_sample, num_video]


        u.tag.test_value = np.asarray([0,1,2],dtype='int32')
        iv.tag.test_value = np.asarray([4,5,2],dtype='int32')
        jv.tag.test_value = np.asarray([1,3,0],dtype='int32')
        mask.tag.test_value = np.asarray([[1,1,0],[1,0,0],[1,1,1]],dtype='int32')
        feat_idx.tag.test_value = np.asarray([[3,4,-1],[5,-1,-1],[6,2,4]],dtype='int32')
        mask_frame.tag.test_value = self.trainset.frame_mask.take(feat_idx.tag.test_value,axis=0)

        rng = np.random
        layers = []

        Uemb = UsrEmblayer(rng,num_user,dim,'usremblayer',prefix)
        Vemb = VidEmblayer(rng,num_item,dim,'videmblayer',prefix)
        feat = self.video_features.take(feat_idx,axis=0) #[num_sample, num_video,dim_feat]
        layers.append(Uemb)
        layers.append(Vemb)
        uemb_vec = GetuEmbLayer(u,Uemb.output,'uemb',prefix)
        iemb_vec = GetvEmbLayer(iv,Vemb.output,'v1emb',prefix)
        jemb_vec = GetvEmbLayer(jv,Vemb.output,'v2emb',prefix)

        layers.append(AttentionLayer_Feat(rng, 2048, uemb_vec.output, feat, dim, dim, mask_frame, 'attentionlayer_feat',prefix))

        layers.append(AttentionLayer_Item(rng, uemb_vec.output, layers[-1].output,dim,dim,mask,'attentionlayer_item',prefix))

        u_vec = uemb_vec.output + layers[-1].output
        self.layers = layers
        y_ui = T.dot(u_vec, iemb_vec.output.T).diagonal()
        y_uj = T.dot(u_vec, jemb_vec.output.T).diagonal()
        self.params = []
        loss = - T.sum(T.log(T.nnet.sigmoid(y_ui - y_uj)))
        for layer in layers:
            self.params += layer.params #[U,V,W_Tran,Wu,Wv,b,c]
        #regularizer = self.reg * ((uemb_vec.output ** 2).sum() + (iemb_vec.output ** 2).sum() + (jemb_vec.output ** 2).sum() +
        #                          (self.params[2] ** 2).sum() + (self.params[3] ** 2).sum() + (self.params[4] ** 2).sum() +
        #                            (self.params[5] ** 2).sum())

        regularizer = self.reg * ((uemb_vec.output ** 2).sum() + (iemb_vec.output ** 2).sum() + (jemb_vec.output ** 2).sum() )

        for param in self.params[2:]:
            regularizer += self.reg * (param ** 2).sum()

        loss = regularizer + loss

        updates = [(param, param-self.lr*T.grad(loss,param)) for param in self.params]

        self.train_model = theano.function(
            inputs = [u,iv,jv,mask_frame,mask,feat_idx],
            outputs = loss,
            updates=updates
        )

        self.test_model = theano.function(
            inputs = [u,mask_frame,mask,feat_idx],
            outputs= [u_vec,Vemb.output],
        )

    def train(self, iters):
        self.trainset.shuffle_data()
        lst = np.random.randint(self.trainset.epoch, size=iters)
        n = 0
        for i in lst:
            n += 1
            users, pos_items, neg_items, mask_frame, mask, feat_idx = self.trainset.get_batch(i)
            out = self.train_model(users,pos_items,neg_items,mask_frame,mask,feat_idx)
            print n, 'cost:', out

    def test(self,topK):
        for i in xrange(self.testset.epoch):
            user_list, mask_frame,mask,feats_idx = self.testset.get_batch(i)
            [user_vector, V_matrix] = self.test_model(user_list, mask_frame,mask,feats_idx)
            #V_value = np.asarray(V_matrix.eval())
            score_maxtrix = np.dot(user_vector,V_matrix.tranpose())
            index_top_K = score_maxtrix.argsort()[:,-topK:][:,::-1]
            hr = 0
            ndcg = 0
            for kk in xrange(len(user_vector)):
                hr += getHitRatio(index_top_K[kk],self.testset.v[i][kk])
                ndcg += getNDCG(index_top_K[kk],self.testset.v[i][kk])
            return hr/len(user_vector),ndcg/len(user_vector)

    def save(self, prefix):
        for layer in self.layers:
            layer.save(prefix)

