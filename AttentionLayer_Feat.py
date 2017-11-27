__author__ = 'jingyuan'
import numpy as np
import theano
import theano.tensor as T
import cPickle
#from theano import pp

def softmask(x,mask):
    #y = T.exp(x-x.max(axis=2,keepdims=True))
    #y = y*mask
    #sumx = T.sum(y,axis=2)
    #x=y/sumx.dimshuffle(0,1,'x')
    #return x
    x_mask = T.switch(mask, x, np.NINF)
    sh = x_mask.shape
    x_mask = x_mask.dimshuffle(2,0,1)
    x_mask = x_mask.reshape((sh[2],-1))
    x_mask = x_mask.dimshuffle(1,0)
    xx = T.nnet.softmax(x_mask)
    xx = xx.dimshuffle(1,0)
    xx = xx.reshape((sh[2],sh[0],sh[1]))
    xx = xx.dimshuffle(1,2,0)
    return xx

class AttentionLayer_Feat(object):
    def __init__(self,rng, dim_feat, uemb_vec, new_Vemb, n_wordin, n_out,mask,name,prefix=None):
        self.inputu = uemb_vec
        self.name = name

        if prefix is None:
            Wu_F_values = np.asarray(
            rng.uniform(
                    low = -np.sqrt(6./(n_wordin+n_out)),
                    high=  np.sqrt(6./(n_wordin+n_out)),
                    size= (n_wordin,n_out)
                ),
                dtype=np.float32
            )
            Wu_F = theano.shared(value=Wu_F_values, name="Wu_F", borrow=True)

            Wv_F_values = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6./(dim_feat+n_out)),
                    high=  np.sqrt(6./(dim_feat+n_out)),
                    size=  (dim_feat,n_out)
                    ),
                    dtype=np.float32
            )
            Wv_F = theano.shared(value=Wv_F_values, name="Wv_F", borrow=True)

            b_F_values = np.zeros((n_out,), dtype='float32')
	        #b_F_values = np.asarray(
		    #  rng.normal(scale=0.1, size=(n_out,)),
		    #  dtype=np.float32
	        #)
            b_F = theano.shared(value=b_F_values, name='b_F', borrow=True)

            c_F_values = np.asarray(
                rng.normal(scale=0.1, size=(n_out,)),
                dtype=np.float32
            )
            c_F = theano.shared(value=c_F_values, name="c_F", borrow=True)

        else:
            f = file(prefix + name + '.save','rb')
            Wu_F = cPickle.load(f)
            Wv_F = cPickle.load(f)
            b_F = cPickle.load(f)
            c_F = cPickle.load(f)

        self.Wu_F = Wu_F
        self.Wv_F = Wv_F
        self.b_F = b_F
        self.c_F = c_F

        items_emb = T.dot(new_Vemb, self.Wv_F)

        attenu = T.dot(self.inputu, self.Wu_F).dimshuffle(0,'x','x',1)
        atten = T.nnet.relu(items_emb + attenu + self.b_F)
        atten = T.sum(atten * self.c_F, axis=3, acc_dtype='float32')
        atten = softmask(atten,mask)

        output = atten.dimshuffle(0,1,2,'x') * items_emb

        self.output = T.sum(output, axis=2, acc_dtype='float32')

        self.params = [self.Wu_F, self.Wv_F, self.b_F, self.c_F]
        self.atten = atten
        self.name = name
        self.mask = mask

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
