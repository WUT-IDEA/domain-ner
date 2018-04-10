import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict
from is13.examples.prepare_data_for_rnn import get_emb
class model(object):

    def __init__(self, nh, nc, ne, de, cs):

        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # p = 'is13/rnn/elman-train_ribao/'
        # parameters of the model
        # self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,(ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        self.emb = theano.shared(0.2 * get_emb(ne+1,de).astype(theano.config.floatX)) # add one for PADDING at the end
        self.Wx = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,(de * cs, nh)).astype(theano.config.floatX))
        self.Wh = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nh)).astype(theano.config.floatX))
        # self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nc)).astype(theano.config.floatX))
        self.bh = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        # self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.h0 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        #add one layer:h2
        self.V1=theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nh)).astype(theano.config.floatX))
        self.V2=theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,(de * cs, nh)).astype(theano.config.floatX))
        self.V3=theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,(nh, nc)).astype(theano.config.floatX))
        self.bv1=theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.bv2=theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        
        # self.emb = theano.shared(0.2 * get_emb(ne + 1, de).astype(theano.config.floatX))  # add one for PADDING at the end
        # self.Wx = theano.shared(0.2 * numpy.load(p + 'Wx.txt.npy').astype(theano.config.floatX))
        # self.Wh = theano.shared(0.2 * numpy.load(p + 'Wh.txt.npy').astype(theano.config.floatX))
        # self.bh = theano.shared(numpy.load(p + 'bh.txt.npy').astype(theano.config.floatX))
        # self.h0 = theano.shared(numpy.load(p + 'h0.txt.npy').astype(theano.config.floatX))
        # # add one layer:h2
        # self.V1 = theano.shared(0.2 * numpy.load(p + 'V1.txt.npy').astype(theano.config.floatX))
        # self.V2 = theano.shared(0.2 * numpy.load(p + 'V2.txt.npy').astype(theano.config.floatX))
        # self.V3 = theano.shared(0.2 * numpy.load(p + 'V3.txt.npy').astype(theano.config.floatX))
        # self.bv1 = theano.shared(numpy.load(p + 'bv1.txt.npy').astype(theano.config.floatX))
        # self.bv2 = theano.shared(numpy.load(p + 'bv2.txt.npy').astype(theano.config.floatX))
        # bundle
        self.params = [self.Wx, self.Wh, self.bh, self.h0 ,self.V1,self.V2, self.V3,self.bv1,self.bv2]
        self.names  = [ 'Wx', 'Wh', 'bh', 'h0','V1','V2','V3','bv1','bv2']
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        u_idxs = T.imatrix() #ribao input x
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        u = self.emb[u_idxs].reshape((u_idxs.shape[0], de * cs))
        y    = T.iscalar('y') # label
        u_y=T.iscalar('u_y')
        def sigmoid_sigmoid(x):
            return 0.8 * T.nnet.sigmoid(x) + 0.2 * T.nnet.hard_sigmoid(x)

        def recurrence(x_t,u, h_tm1):
            temp=T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh
            # h_t = T.nnet.sigmoid(temp)  # the t moment output of the hidden layer
            h_t = sigmoid_sigmoid(temp)
            h2=sigmoid_sigmoid(T.dot(h_t, self.V1) + T.dot(u, self.V2) + self.bv1 )
            s_t = T.nnet.softmax(T.dot(h2, self.V3) + self.bv2)  # the t moment output of the output layer
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=[x,u], outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        #print 'y_pred', y_pred
        #print ' p_y_given_x_sentence', p_y_given_x_sentence


        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.log(p_y_given_x_lastword)[y] #negative log-likelihood(NLL)
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))

        # theano functions
        self.myclassify = theano.function(inputs=[idxs,u_idxs], outputs=p_y_given_x_sentence)
        self.classify = theano.function(inputs=[idxs,u_idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs, u_idxs,y,u_y, lr],
                                      outputs = nll,
                                      updates = updates ,on_unused_input='ignore')

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            # numpy.save(os.path.join(folder, name + '.txt'), param.get_value())
            numpy.savetxt(os.path.join(folder,name+'.txt'), param.get_value())
