#coding:utf-8
import gzip
import cPickle
import urllib
import os
import random
import theano
import numpy
from os.path import isfile


# PREFIX = os.getenv('ATISDATA', '')



def download(origin):
    '''
    download the corresponding atis file
    from http://www-etud.iro.umontreal.ca/~mesnilgr/atis/
    '''
    print 'Downloading data from %s' % origin
    name = origin.split('/')[-1]
    urllib.urlretrieve(origin, name)


def load_dropbox(filename):
    if not isfile(filename):
        # download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/'+filename)
        print 'atis.plk is not found!'
    f = open(filename, 'rb')
    return f


def load_udem(filename):
    if not isfile(filename):
        print 'atis.fold.pkl.gz is not existed, downloading...'
        download('http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/' + filename)
    f = gzip.open(filename, 'rb')
    return f


def atisfull():
    f = load_dropbox('atis.pkl')
    train_set, test_set, dicts = cPickle.load(f)
    return train_set, test_set, dicts


def atisfold(fold):
    assert fold in range(5)
    f = load_udem('atis.fold' + str(fold) + '.pkl.gz')
    train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts

def toarray(string):
    line=string.strip('\n').split(' ')
    train_lex = []
    for e in line:
        if e=='':continue
        train_lex.append(float(e))

    row = numpy.array(train_lex)
    return row

if __name__ == '__main__':

    ''' visualize a few sentences '''

    '''
    w2ne, w2la = {}, {}
    train, test, dic = atisfull()
    train, _, test, dic = atisfold(3)

    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']

    idx2w = dict((v, k) for k, v in w2idx.iteritems())
    idx2ne = dict((v, k) for k, v in ne2idx.iteritems())
    idx2la = dict((v, k) for k, v in labels2idx.iteritems())

    test_x, test_ne, test_label = test
    train_x, train_ne, train_label = train
    all_x=train_x
    all_label=train_label
    for tx, tl in zip(test_x, test_label):
        all_x.append(tx)
        all_label.append(tl)
    wlength = 35
    i = 1
    words=0
    f = open('HMMtest.txt', 'w')

    for x, l in zip(all_x, all_label):#1001-2000
        if i<=1000:
            i += 1
            continue

        if i>2000:
            break


        for w, y in zip(x, l):
            f.write(idx2w[w] + ' ' + idx2la[y] + '\n')
            words+=1

            #f.write(idx2w[w] + '\n')
        f.write('\n')

        i+=1
        #print i
    print 'words:',words
    f.close()
    '''


    '''
    # remove the 300 high confidence level data produced by HMM and RNN from pre_train
    f=open('co-train/pre_train_x.txt')
    to_remove=open('co-train/hmm_train_x.txt')
    store=[]
    for line in iter(f):
        line=line.strip('\n')
        store.append(line)
    f.close()
    print 'before remove HMM train_x,length:', len(store)
    for remove in iter(to_remove):
        remove = remove.strip('\n')
        if remove in store:
            store.remove(remove)
    to_remove.close()
    print 'after remove HMM train_x,length:', len(store)
    f = open('co-train/pre_train_x.txt','w')
    for sentence in store:
        f.write(sentence+'\n')
    f.close()
    '''


    emb = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, \
                                                        (5 + 1, 9)).astype(theano.config.floatX))
    # print emb
    # print numpy.random.uniform(-1,1,(1,200))
    print '---------------------------------'
    s='dfdvdfd'
    print s.strip('d')
    print s

    for e in xrange(1):
        print 'a'
    l=[1,2,3]
    f=open('a.txt','w')
    f.write(str(l)+'\n')
    print 'open'
    f.close()

    for i in range(0,5):
        print i

    '''

    f=open("totaltxtemb")
    dic={}
    res=''
    for s in iter(f):
        s=s.strip('\n').strip(' ')
        key=s.split(' ')[0]
        dic[key]=s.strip(key).strip(' ')
    print len(dic)
    f.close()
    b = open("../data/dictionary.txt")
    count = 0
    for w in iter(b):
        w = w.strip('\n')
        if dic.has_key(w):
            res+=w+' '+dic.get(w)+'\n'
            count+=1
    b.close()
    print 'find emb:',count
    result=open('word_embedings.txt','w')
    result.write(res)
    result.close()
    '''







