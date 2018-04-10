import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict
from theano import scalar
from keras import backend as K
from keras.layers import activations as A
class model(object):

    def __init__(self, nh, nc, ne, de, cs):

        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary  572
        de :: dimension of the word embeddings    100
        cs :: word window context size 
        '''
        # parameters of the model
        self.emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y    = T.iscalar('y') # label

        def Relu(x):
            out_dtype = scalar.upgrade_to_float(scalar.Scalar(dtype=x.dtype))[0].dtype
            a = T.constant(0.5, dtype=out_dtype)
            # ab = T.constant(abs(x), dtype=out_dtype)
            # x = (x * slope) + shift
            y=(x + abs(x)) * a
            r= T.clip(y, 0, 1)
            return r

        def PRelu(x):
            out_dtype = scalar.upgrade_to_float(scalar.Scalar(dtype=x.dtype))[0].dtype
            a = T.constant(0.625, dtype=out_dtype)
            b = T.constant(0.375, dtype=out_dtype)
            # x = (x * slope) + shift
            y = x * a + abs(x) * b
            r= T.clip(y, 0, 1)
            return r
        def my_tanh(x):
            #return 2*T.nnet.sigmoid(2*x)-1
            return T.nnet.sigmoid(x)
        def sigmoid_sigmoid(x):
            return 0.8*T.nnet.sigmoid(x)+0.2*T.nnet.hard_sigmoid(x)

        def recurrence(x_t, h_tm1):
            temp=T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh
            #h_t = T.nnet.hard_sigmoid(temp)  # the t moment output of the hidden layer
            #h_t = T.tanh(temp)

            h_t=T.nnet.sigmoid(temp)

            #h_t=T.nnet.relu(temp,0.2)#relu=T.maximum(0, temp)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)  # the t moment output of the output layer
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
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
        self.myclassify = theano.function(inputs=[idxs], outputs=p_y_given_x_sentence)
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs, y, lr],
                                      outputs = nll,
                                      updates = updates )

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.txt'), param.get_value())
