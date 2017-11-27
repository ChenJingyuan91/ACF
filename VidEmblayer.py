__author__ = 'jingyuan'
import numpy
import theano
import cPickle

class VidEmblayer(object):
    def __init__(self, rng, n_vid, dim, name, prefix=None):
        self.name = name
        if prefix == None:
            #V_values = numpy.asarray(
            #    rng.normal(scale=0.1, size=(n_vid,dim)),
            #dtype=numpy.float32
            #)
            V_values= numpy.zeros((n_vid,dim),dtype=numpy.float32)
            V = theano.shared(value=V_values, name='V', borrow=True)
        else:
            f = file(prefix + name + '.save', 'rb')
            V = cPickle.load(f)
            f.close()

        self.V = V
        self.output = self.V
        self.params = [self.V]

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()