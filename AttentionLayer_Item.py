__author__ = 'jingyuan'
import numpy as np
import theano
import theano.tensor as T
import cPickle
#from theano import pp

def softmask(x,mask):
    #y = T.exp(x-x.max(axis=1, keepdims=True))
    #y = y * mask
    #sumx = T.sum(y,axis=1,acc_dtype='float32')
    #x=y/sumx.dimshuffle(0,'x')
    #return x
    x_mask = T.switch(mask, x, np.NINF)
    xx = T.nnet.softmax(x_mask)
    return xx

class AttentionLayer_Item(object):
    def __init__(self,rng, uemb_vec, new_Vemb, n_wordin, n_out,mask,name,prefix=None):
        self.inputu = uemb_vec
        self.name = name

        if prefix is None:
            Wu_I_values = np.asarray(
            rng.uniform(
                    low = -np.sqrt(6./(n_wordin+n_out)),
                    high=  np.sqrt(6./(n_wordin+n_out)),
                    size= (n_wordin,n_out)
                ),
                dtype=np.float32
            )
            Wu_I = theano.shared(value=Wu_I_values, name="Wu_I", borrow=True)

            Wv_I_values = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6./(n_wordin+n_out)),
                    high=  np.sqrt(6./(n_wordin+n_out)),
                    size=  (n_wordin,n_out)
                    ),
                    dtype=np.float32
            )
            Wv_I = theano.shared(value=Wv_I_values, name="Wv_I", borrow=True)

            b_I_values = np.zeros((n_out,), dtype='float32')
	    	#b_values = np.asarray(
			#rng.normal(scale=0.1, size=(n_out,)),
			#dtype=np.float32
	    	#)
            b_I = theano.shared(value=b_I_values, name='b_I', borrow=True)

            c_I_values = np.asarray(
                rng.normal(scale=0.1, size=(n_out,)),
                dtype=np.float32
            )
            c_I = theano.shared(value=c_I_values, name="c_I", borrow=True)

        else:
            f = file(prefix + name + '.save','rb')
            Wu_I = cPickle.load(f)
            Wv_I = cPickle.load(f)
            b_I = cPickle.load(f)
            c_I = cPickle.load(f)

        self.Wu_I = Wu_I
        self.Wv_I = Wv_I
        self.b_I = b_I
        self.c_I = c_I

        items_emb = new_Vemb
        attenu = T.dot(self.inputu, self.Wu_I).dimshuffle(0, 'x', 1)
        atten = T.nnet.relu(T.dot(items_emb, self.Wv_I) + attenu + self.b_I)
        atten = T.sum(atten * self.c_I, axis=2, acc_dtype='float32')
        atten = softmask(atten,mask)
        output = atten.dimshuffle(0,1,'x') * items_emb
        self.output = T.sum(output, axis=1, acc_dtype='float32')
        self.params = [self.Wu_I, self.Wv_I, self.b_I, self.c_I]
        self.atten = atten
        self.name = name
        self.mask = mask

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
