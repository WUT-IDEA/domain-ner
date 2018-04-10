import numpy
import time
import sys
import subprocess
import os
import random
from theano import tensor as T


from is13.data import load
from is13.rnn.my_elman import model
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

    #co-train
    '''
    # alltrain_x =get_datalist('../data/cotrain/trainx.txt')
    alltrain_y  =get_datalist('../data/cotrain/trainy.txt')
    train_lex=alltrain_x
    train_y=alltrain_y
    test_lex =get_datalist('../data/cotrain/testx.txt')
    test_y = get_datalist('../data/cotrain/testy.txt')
    '''
    #initial running, obtain zhengzhi trainset
    alltrain_x = get_datalist('../data/trainx_useful.txt')
    alltrain_y = get_datalist('../data/trainy_useful.txt')
    # alltrain_x = get_datalist('../data/top1674x.txt')
    # alltrain_y = get_datalist('../data/top1674y.txt')
    train_lex=alltrain_x[0:3855]
    # test_lex=alltrain_x[3855:]
    train_y = alltrain_y[0:3855]
    # test_y = alltrain_y[3855:]
    test_lex = get_datalist('../data/cotrain/testx.txt')
    test_y = get_datalist('../data/cotrain/testy.txt')

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

    # train with early stopping on validation set
    print 'train with set...'
    print time.localtime(time.time())
    best_f1 = -numpy.inf
    s['clr'] = s['lr']
    best_f1 = 0.0
    for e in xrange(s['nepochs']):
        predictions_test = []
        # shuffle
        shuffle([train_lex, train_y], s['seed'])
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
                # rnn.normalize()
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
        # evaluation // compute the accuracy using conlleval.pl
        print 'evaluation step2...compute the accuracy using conlleval.pl'
        # res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        res_test =my_conll(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')

        if res_test['f1'] > best_f1:
            rnn.save(folder)
            best_f1 = res_test['f1']
            print 'now,best_f1=', best_f1
            if s['verbose']:
                tempstr= 'NEW BEST: epoch '+str(e)+', best test P ,R, F1 '+str(res_test['p'])+'  '+str(res_test['r'])+'  '+str(res_test['f1'])
                f = open('result.txt', 'a')
                f.write(tempstr + '\n')
                f.close()
                print tempstr #'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20

            # s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
            s['tf1'], s['tp'], s['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
            s['be'] = e
           # subprocess.call(['rename', folder + '/current.test.txt', folder + '/best.test.txt']) #mv->rename
            #subprocess.call(['rename', folder + '/current.valid.txt', folder + '/best.valid.txt'])
            if os.path.isfile(folder+'/best.test.txt'):
                os.remove(folder+'/best.test.txt')
            os.rename(folder + '/current.test.txt', folder + '/best.test.txt')
            # if os.path.isfile(folder + '/best.valid.txt'):
            #     os.remove(folder + '/best.valid.txt')
            # os.rename(folder + '/current.valid.txt', folder + '/best.valid.txt')
            #print 'test.... test.... test.... test....'
        else:
            print ''

        # learning rate decay if no improvement in 10 epochs
        if s['decay'] and abs(s['be']-s['ce']) >= 10: s['clr'] *= 0.5
        if s['clr'] < 1e-5: break

    print 'BEST RESULT: epoch', s['be'], 'best test F1', s['tf1'], 'with the model', folder
    print time.localtime(time.time())
    print 'training done!!!'

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