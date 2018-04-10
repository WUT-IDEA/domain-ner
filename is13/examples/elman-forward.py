import numpy
import time
import sys
import subprocess
import os
import random
from theano import tensor as T


from is13.data import load
from is13.rnn.elman import model
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle, minibatch, contextwin

if __name__ == '__main__':

    s = {'fold':3, # 5 folds 0,1,2,3,4
         'lr':0.0627142536696559,
         'verbose':1,
         'decay':False, # decay on the learning rate if improvement stops
         'win':9, # number of words in the context window
         'bs':9, # number of backprop through time steps
         'nhidden':100, # number of hidden units
         'seed':345,
         'emb_dimension':200, # dimension of word embedding
         'nepochs':50}

    folder = os.path.basename(__file__).split('.')[0]
    print 'folder=', folder
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    print 'load the dataset...'
    train_set, valid_set, test_set, dic = load.atisfold(s['fold'])
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())



    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex,  test_ne,  test_y  = test_set

    vocsize = len(dic['words2idx'])
    print 'vosize=', vocsize #572
    nclasses = len(dic['labels2idx'])
    print nclasses #127
    nsentences = len(train_lex)
    print 'train data length:',nsentences #3983 to train;  test_lex:893

    # instanciate the model
    print 'instanciate the model'
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    ne = vocsize,
                    de = s['emb_dimension'],
                    cs = s['win'] )

    # train with early stopping on validation set
    print 'train with set...'
    print time.localtime(time.time())
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    for e in xrange(s['nepochs']):
        predictions_test = []
        # shuffle
        shuffle([train_lex, train_ne, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
            #print 'i=', i
            cwords = contextwin(train_lex[i], s['win'])
            words  = map(lambda x: numpy.asarray(x).astype('int32'),\
                         minibatch(cwords, s['bs']))
            labels = train_y[i]
            #print 'label=', labels
            for word_batch , label_last_word in zip(words, labels):
                t=rnn.train(word_batch, label_last_word, s['clr'])
                rnn.normalize()
            if s['verbose']:
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic)
                sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        print 'evaluation step1: back into the real world : idx -> words'
        #for x in test_lex:
            #scores=rnn.myclassify(numpy.asarray(contextwin(x, s['win'])).astype('int32')) #each word's 127 lebels score in each line
           # final_label = map(lambda x:listmax(x),scores)
            #final_label=T.argmax(scores,axis=1)
           # predictions_test.append(map(lambda x:idx2label[x],final_label))

        predictions_test = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
                             for x in test_lex ]

        groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y]

        words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

        predictions_valid = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))\
                             for x in valid_lex ]
        groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]
        words_valid = [ map(lambda x: idx2word[x], w) for w in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        print 'evaluation step2...compute the accuracy using conlleval.pl'
        res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

        if res_valid['f1'] > best_f1:
            rnn.save(folder)
            best_f1 = res_valid['f1']
            print 'now,best_f1=', best_f1
            if s['verbose']:
                tempstr= 'NEW BEST: epoch '+str(e)+', valid F1 '+ str(res_valid['f1'])+', best test F1 '+str(res_test['f1'])
                f = open('different-size-result.txt', 'a')
                f.write(tempstr + '\n')
                f.close()
                print tempstr #'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20

            s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
            s['tf1'], s['tp'], s['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
            s['be'] = e
           # subprocess.call(['rename', folder + '/current.test.txt', folder + '/best.test.txt']) #mv->rename
            #subprocess.call(['rename', folder + '/current.valid.txt', folder + '/best.valid.txt'])
            if os.path.isfile(folder+'/best.test.txt'):
                os.remove(folder+'/best.test.txt')
            os.rename(folder + '/current.test.txt', folder + '/best.test.txt')
            if os.path.isfile(folder + '/best.valid.txt'):
                os.remove(folder + '/best.valid.txt')
            os.rename(folder + '/current.valid.txt', folder + '/best.valid.txt')
            #print 'test.... test.... test.... test....'
        else:
            print ''

        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5
        if s['clr'] < 1e-5: break

    print 'BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder
    print time.localtime(time.time())

