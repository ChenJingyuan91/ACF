__author__ = 'jingyuan'
import numpy as np
import theano
import theano.tensor as T
import cPickle

class ContentEmbLayer(object):
    def __init__(self, rng, orginal_feat, dim_in, dim_out, name, prefix= None):
        self.name = name
        self.orginal_feat = orginal_feat

        if prefix == None:
            W_Tran_values = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6./(dim_in+dim_out)),
                    high=  np.sqrt(6./(dim_in+dim_out)),
                    size= (dim_in, dim_out)
                ),
                dtype=np.float32
            )
            W_Tran = theano.shared(value=W_Tran_values, name="W_Tran", borrow=True)

        else:
            f = file(prefix + name +'.save','rb')
            W_Tran = cPickle.load(f)
            f.close()

        self.W_Tran = W_Tran

        out_feat = T.dot(self.orginal_feat, self.W_Tran)

        self.output = out_feat
        self.params = [self.W_Tran]

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()