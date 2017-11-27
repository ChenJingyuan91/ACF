__author__ = 'jingyuan'
class GetvEmbLayer(object):
    def __init__(self,vid,Vemb,name,prefix=None):
        self.input = vid
        self.name = name

        self.output = Vemb[vid]
        self.params = []

    def save(self, prefix):
        pass