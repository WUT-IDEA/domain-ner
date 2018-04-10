import gzip
import cPickle
import urllib
import os
import random

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

def getrnndata(begin,end):
    w2ne, w2la = {}, {}
    train, test, dic = atisfull()
    train, _, test, dic = atisfold(3)

    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']

    idx2w = dict((v, k) for k, v in w2idx.iteritems())
    idx2ne = dict((v, k) for k, v in ne2idx.iteritems())
    idx2la = dict((v, k) for k, v in labels2idx.iteritems())

    test_x, test_ne, test_label = test
    train_x, train_ne, train_label = train
    print len(test_x), len(train_label)
    print train_label[1001], train_x[1001]
    i = 0
    words = 0
    f = open('getdata.txt', 'w')
    all_x, all_label = train_x, train_label
    for tx, tl in zip(test_x, test_label):
        all_x.append(tx)
        all_label.append(tl)

    for x, l in zip(train_x, train_label):
        # for x, l in zip(test_x, test_label):

        if i < begin:
            i += 1
            continue
        if i >= end: break

        for w, y in zip(x, l):
            f.write(idx2w[w] + ' ' + idx2la[y] + '\n')
            words += 1
            # f.write(idx2w[w] + '\n')
        f.write('\n')
        i += 1
        print i
    print 'words:', words
    f.close()


train, _, test, dic = atisfold(3)

w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']

idx2w = dict((v, k) for k, v in w2idx.iteritems())
idx2ne = dict((v, k) for k, v in ne2idx.iteritems())
idx2la = dict((v, k) for k, v in labels2idx.iteritems())

test_x, test_ne, test_label = test
train_x, train_ne, train_label = train

all_data={}
all_x, all_label = train_x, train_label
for tx, tl in zip(test_x, test_label):
    all_x.append(tx)
    all_label.append(tl)


for x, l in zip(all_x, all_label):
    sen='[ '
    la='[ '
    for v,w in zip(x,l):
        sen+=str(v)+ ' '
        la+=str(w)+ ' '
    sen+=']'
    la+=']'
    all_data[sen]=la #{[23 34 67 33]:[126 126 3 6],}


def update_crfpre_train(file,sentence):
    assert sentence in all_data.keys()
    label=all_data[sentence]
    sentence=sentence.strip('[').strip(']').split(' ')
    sentence.remove('')
    sentence.remove('')
    label = label.strip('[').strip(']').split(' ')
    label.remove('')
    label.remove('')
    for sen,la in zip(sentence,label):
        file.write(idx2w[int(sen.strip('\n'))]+' '+idx2la[int(la.strip('\n'))]+'\n')
    file.write('\n')




def countword(input,output):
    f=open(input)
    o=open(output,'w')
    count=0
    l=[]
    for i in f:
        i=i.strip('\n')
        if i=='':
            count+=1
        else:
            w=i.split(' ')[1]
            if not w in l:
                l.append(w)
                o.write(w+'\n')
    print count, len(l)
    f.close()
    o.close()





if __name__ == '__main__':

    ''' visualize a few sentences '''


    # remove the 300 high confidence level data produced by HMM and RNN from pre_train
    #update rnn and crf 's pre_train
    # xx=open('co-train/pre_train_x.txt')
    # to_remove=open('crf/crf-produced300train-x')
    # store=[]
    # for line in iter(xx):
    #     line=line.strip('\n')
    #     store.append(line)
    # xx.close()
    # before=len(store)
    # print 'before remove crf-produced train_x,length:', before
    # for remove in iter(to_remove):
    #     remove = remove.strip('\n')
    #     if remove in store:
    #         store.remove(remove)
    # to_remove.close()
    # after=len(store)
    # print 'after remove crf-produced train_x,length:', after
    # f = open('co-train/pre_train_x.txt','w')
    # crf=open('crf/pre-train','w')
    # for sentence in store:
    #     f.write(sentence+'\n')
    #     update_crfpre_train(crf,sentence)
    # f.close()
    # crf.close()
    # print 'total remove sentences:', 300+before-after

    countword('../dataset2/alldata.txt','../dataset2/alllabels2.txt')



