import numpy
import time
import sys
import subprocess
import os
import random
import time
from theano import tensor as T
from prepare_data_for_rnn import get_dictionary,get_label2idx,get_datalist


from is13.data import load
from is13.rnn.elman import model
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle, minibatch, contextwin,writelist,listmax,saveIntoFile,get_word_posTagging

if __name__ == '__main__':

    s = {'fold':3, # 5 folds 0,1,2,3,4
         'lr':0.0627142536696559,
         'verbose':1,
         'decay':False, # decay on the learning rate if improvement stops
         'win':7, # number of words in the context window
         'bs':9, # number of backprop through time steps
         'nhidden':100, # number of hidden units
         'seed':345,
         'emb_dimension':100, # dimension of word embedding
         'nepochs':50}

    folder = os.path.basename(__file__).split('.')[0]
    print 'folder=', folder
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    print 'load the dataset...'
    train_set, valid_set, test_set, dic = load.atisfold(s['fold'])
    idx2label = dict((k, v) for v, k in dic['labels2idx'].iteritems())
    idx2word = dict((k, v) for v, k in dic['words2idx'].iteritems())

    # for k, v in idx2label.iteritems():
    #    saveIntoFile('dict_idx2label.txt', str(k) + ' ' + str(v))
    #  for k, v in idx2word.iteritems():
    #     saveIntoFile('dict_idx2word.txt', str(k) + ' ' + str(v))

    #train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y = test_set
    #pre_train=train_lex[796::]
    #train_lex=train_lex[0:796]
    #train_y=train_y[0:796]
    train_lex=get_datalist('co-train/trainX.txt')
    train_y = get_datalist('co-train/trainY.txt')
    test_lex=get_datalist('co-train/co-traintest_x.txt')
    test_y = get_datalist('co-train/co-traintest_y.txt')

    '''
    print "pre-train legth",len(pre_train)
    f=open('co-train/trainY.txt','w')
    for v in train_y:
        v=list(v)
        line='[ '
        for i in v:
            line+=str(i)+' '
        line+=']'
        f.write(line+'\n')
    f.close()
    '''
    # for v in train_y:
    #   saveIntoFile('train_y.txt', str(v))

    print 'train data length:',len(train_lex)
    print 'test data length:',len(test_lex)

    vocsize = len(dic['words2idx'])
    print 'vosize=', vocsize  # 572
    nclasses = len(dic['labels2idx'])
    #print nclasses  # 127
    nsentences = len(train_lex)
    #print nsentences  # 3983 to train;  test_lex:893

    print 'instanciate the model'
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(nh = s['nhidden'],nc = nclasses,ne = vocsize, de = s['emb_dimension'], cs = s['win'] )

    # train with early stopping on validation set
    print 'train with set...'
    best_f1 = -numpy.inf
    s['clr'] = s['lr']

    print time.localtime(time.time())

    for e in xrange(s['nepochs']):
        # shuffle
        shuffle([train_lex, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        for i in xrange(nsentences):
            #print 'i=', i
            cwords = contextwin(train_lex[i], s['win'])
            words  = map(lambda x: numpy.asarray(x).astype('int32'), minibatch(cwords, s['bs']))
            labels = train_y[i]
            #print 'label=', labels
            for word_batch , label_last_word in zip(words, labels):
                t=rnn.train(word_batch, label_last_word, s['clr'])
                rnn.normalize()
            if s['verbose']:
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        print 'evaluation step1: back into the real world : idx -> words'

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
                f = open('co-train_result.txt', 'a')
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

    print 'BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1']
    print 'epoch finished.\n',time.localtime(time.time())


    #co-train:  produce k number new high believable new train data for HMM

    pre_train_x=get_datalist('co-train/pre_train_x.txt')
    print 'before: length of pre_train:',len(pre_train_x)
    count=0
    sentences_and_scores={}
    pre_train=[]
    for x in pre_train_x:
        scores = rnn.myclassify(
            numpy.asarray(contextwin(x, s['win'])).astype('int32'))  # each word's 127 lebels score in each line[[label1_score,label2_score,..][label1_score,label2_score,..]]
        maxscores=map((lambda x:listmax(x)),scores)
        sentence_score=sum(maxscores)
        sentence_label =map(lambda x: idx2label[x],rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32')))
        x=x.tolist()
        count += 1
        pre_train.append(x)
        sentences_and_scores[count]=[x,sentence_label,sentence_score]#{1:[sentence,pred_label,score],2:[sentence,pred_label,score]}


    #sort the sentence_and_scores by score and save first k sentence into newHMM_train_data
    hmm = open('co-train/newHMM_train_data.txt', 'w')
    sorted_dic = sorted(sentences_and_scores.items(), key=lambda d: d[1][2], reverse=True)  # sort by value.[('china', 9), ('io', 4), ('ret', 2), ('me', 2)]
    i=0
    for sentence in sorted_dic:
        if i>300:break #k=300
        i=i+1
        pre_train.remove(sentence[1][0])
        assert len(sentence[1][0])==len(sentence[1][1]) #sentence vs label
        for index in xrange(0,len(sentence[1][0])):
            word = idx2word[sentence[1][0][index]]
            label= sentence[1][1][index]
            hmm.write(str(word) + ' ' + str(label) + '\n')
        hmm.write('\n')

    hmm.close()
    print 'after: length of pre_train:', len(pre_train)
    writelist(pre_train,'co-train/pre_train_x.txt')
    print '----done!!!-----'



