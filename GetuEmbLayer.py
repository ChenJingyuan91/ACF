__author__ = 'jingyuan'
class GetuEmbLayer(object):
    def __init__(self,u,Uemb,name,prefix=None):
        self.input = u
        self.name = name

        self.output = Uemb[u]
        self.params = []

    def save(self, prefix):
        pass