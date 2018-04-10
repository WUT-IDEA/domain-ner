import numpy
import time
import sys
import subprocess
import os
import random
from theano import tensor as T


from is13.data import load
from is13.rnn.sougou_elman import model
from is13.metrics.accuracy import conlleval,my_conll
from is13.utils.tools import shuffle, minibatch, contextwin,listmax
from is13.examples.prepare_data_for_rnn import label2idx,dictionary,get_datalist,transforx,transfory


if __name__ == '__main__':

    s = {
         'lr':0.0627142536696559,
         'verbose':1,
         'decay':False, # decay on the learning rate if improvement stops
         'win':1, # number of words in the context window
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
    idx2label = dict((k,v) for v,k in label2idx.iteritems())
    idx2word  = dict((k,v) for v,k in dictionary.iteritems())

    #initial running, obtain zhengzhi trainset
    alltrain_x = get_datalist('is13/data/sougou/trainx.txt')
    alltrain_y = get_datalist('is13/data/sougou/trainy.txt')

    train_lex=alltrain_x[0:245077]
    test_lex=alltrain_x[245077:]
    train_y = alltrain_y[0:245077]
    test_y = alltrain_y[245077:]
    # test_lex = get_datalist('../data/cotrain/testx.txt')
    # test_y = get_datalist('../data/cotrain/testy.txt')

    vocsize = len(dictionary)
    print 'vosize=', vocsize #572
    nclasses = len(label2idx)
    print 'classes:', nclasses #127
    nsentences = len(train_lex)
    print 'train data length:',nsentences #3983 to train;  test_lex:893
    print 'test data length:',len(test_lex)
    # instanciate the model
    print 'instanciate the model'
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    ne = vocsize,
                    de = s['emb_dimension'],
                    cs = s['win'] )

   rnn.save(folder)
    #
    # #score to zhengzhi_idx
    # zhengzhi_idx = get_datalist('../data/zhengzhi_useful.txt')
    # # zhengzhi_idx = get_datalist('../data/cotrain/pretrain.txt')
    # print 'length of pretrain txt sentences:', len(zhengzhi_idx)
    # count = 0
    # sentences_and_scores = {}
    # for x in zhengzhi_idx:
    #     scores = rnn.myclassify(
    #         numpy.asarray(contextwin(x, s['win'])).astype(
    #             'int32'))  # each word's 127 lebels score in each line[[label1_score,label2_score,..][label1_score,label2_score,..]]
    #     maxscores = map((lambda x: listmax(x)), scores)
    #     sentence_score = sum(maxscores)/len(maxscores)
    #     sentence_label = map(lambda x: idx2label[x],
    #                          rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))
    #     x = x.tolist()
    #     count += 1
    #     print 'predicting the %i sentences'%count
    #     sentences_and_scores[count] = [x, sentence_label,
    #                                    sentence_score]  # {1:[sentence,pred_label,score],2:[sentence,pred_label,score]}
    #
    # # sort the sentence_and_scores by score and save first k sentence into newHMM_train_data
    # # zhengzhi_predx = open('../data/cotrain/rnn_predx.txt', 'w') #zhengzhi_idx after sorted
    # # zhengzhi_predy = open('../data/cotrain/rnn_predy.txt', 'w')
    # zhengzhi_predx = open('../data/rnn_predx.txt', 'w') #zhengzhi_idx after sorted
    # zhengzhi_predy = open('../data/rnn_predy.txt', 'w')
    # sorted_dic = sorted(sentences_and_scores.items(), key=lambda d: d[1][2],
    #                     reverse=True)  # sort by value.[('china', 9), ('io', 4), ('ret', 2), ('me', 2)]
    #
    # topk=0
    # for sentence in sorted_dic:
    #     if topk>11111799: break       #predict top800
    #     assert len(sentence[1][0]) == len(sentence[1][1])  # sentence vs label
    #     zhengzhi_predx.write(transforx(str(sentence[1][0]))+'\n')
    #     zhengzhi_predy.write(transfory(str(sentence[1][1])) + '\n')
    #     topk+=1
    #
    # zhengzhi_predx.close()
    # zhengzhi_predy.close()
    # print 'prediction done!!!'