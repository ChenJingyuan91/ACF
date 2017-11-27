__author__ = 'jingyuan'
import numpy as np
import theano
import theano.tensor as T
import cPickle
#from theano import pp

def softmask(x,mask):
    y = T.exp(x-x.max(axis=1, keepdims=True))
    y = y * mask
    sumx = T.sum(y,axis=1,acc_dtype='float32')
    #x = y/sumx.dimshuffle(0,'x')
    x=y/sumx.dimshuffle(0,'x')
    return x

def softmask_content(x):
    y = T.exp(x-x.max(axis=2,keepdims=True))
    #pp(y)
    sumx = T.sum(y,axis=2)
    #pp(sumx)
    x=y/sumx.dimshuffle(0,1,'x')
    return x

class AttentionLayer_Relu(object):
    def __init__(self,rng, uemb_vec, new_Vemb, n_wordin, n_out,mask,name,prefix=None):
        self.inputu = uemb_vec
        self.name = name

        if prefix is None:
            Wu_values = np.asarray(
            rng.uniform(
                    low = -np.sqrt(6./(n_wordin+n_out)),
                    high=  np.sqrt(6./(n_wordin+n_out)),
                    size= (n_wordin,n_out)
                ),
                dtype=np.float32
            )
            Wu = theano.shared(value=Wu_values, name="Wu", borrow=True)

            Wv_values = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6./(n_wordin+n_out)),
                    high=  np.sqrt(6./(n_wordin+n_out)),
                    size=  (n_wordin,n_out)
                    ),
                    dtype=np.float32
            )
            Wv = theano.shared(value=Wv_values, name="Wv", borrow=True)

            #b_values = np.zeros((n_out,), dtype='float32')
	    b_values = np.asarray(
		rng.normal(scale=0.1, size=(n_out,)),
		dtype=np.float32
	    )
            b = theano.shared(value=b_values, name='b', borrow=True)

            c_values = np.asarray(
                rng.normal(scale=0.1, size=(n_out,)),
                dtype=np.float32
            )
            c = theano.shared(value=c_values, name="c", borrow=True)

        else:
            f = file(prefix + name + '.save','rb')
            Wu = cPickle.load(f)
            Wv = cPickle.load(f)
            b = cPickle.load(f)
            c = cPickle.load(f)

        self.Wu = Wu
        self.Wv = Wv
        self.b = b
        self.c = c

        if self.name == 'attentionlayer_item':

            items_emb = new_Vemb

            attenu = T.dot(self.inputu, self.Wu).dimshuffle(0, 'x', 1)

            atten = T.nnet.relu(T.dot(items_emb, self.Wv) + attenu + self.b)
            atten = T.sum(atten * self.c, axis=2, acc_dtype='float32')
            atten = softmask(atten,mask)

            output = atten.dimshuffle(0,1,'x') * items_emb

            self.output = T.sum(output, axis=1, acc_dtype='float32')

        if self.name == 'attentionlayer_cont':
            items_emb = new_Vemb

            attenu = T.dot(self.inputu, self.Wu).dimshuffle(0,'x','x',1)
            atten = T.nnet.relu(T.dot(items_emb, self.Wv) + attenu + self.b)
            atten = T.sum(atten * self.c, axis=3, acc_dtype='float32')
            atten = softmask_content(atten)

            output = atten.dimshuffle(0,1,2,'x') * items_emb

            self.output = T.sum(output, axis=2, acc_dtype='float32')

        self.params = [self.Wu, self.Wv, self.b, self.c]
        self.atten = atten
        self.name = name
        self.mask = mask

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
