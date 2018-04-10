import gzip
import cPickle
import urllib
import os
import random

from os.path import isfile

#PREFIX = os.getenv('ATISDATA', '')



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
        #download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/'+filename)
        print 'atis.plk is not found!'
    f = open(filename,'rb')
    return f

def load_udem(filename):
    if not isfile(filename):
        print 'atis.fold.pkl.gz is not existed, downloading...'
        download('http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/'+filename)
    f = gzip.open(filename,'rb')
    return f


def atisfull():
    f = load_dropbox('atis.pkl')
    train_set, test_set, dicts = cPickle.load(f)
    return train_set, test_set, dicts

def atisfold(fold):
    assert fold in range(5)
    f = load_udem('atis.fold'+str(fold)+'.pkl.gz')
    train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts
 
if __name__ == '__main__':
    
    ''' visualize a few sentences '''

    import pdb
    
    w2ne, w2la = {}, {}
    train, test, dic = atisfull()
    train, _, test, dic = atisfold(3)
    
    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']
    
    idx2w  = dict((v,k) for k,v in w2idx.iteritems())
    idx2ne = dict((v,k) for k,v in ne2idx.iteritems())
    idx2la = dict((v,k) for k,v in labels2idx.iteritems())

    test_x,  test_ne,  test_label  = test
    train_x, train_ne, train_label = train
    wlength = 35
    i=0
    f=open('train_x2word_withLabel.txt', 'a')
    for x,l in zip(train_x, train_label):
        f.write(str(x)+'\n')
        for w,y in zip(x,l):
            f.write(idx2w[w]+'/'+idx2la[y]+' ')
        f.write('\n')
    f.close()

    for e in ['train', 'test']:
        for sw, se, sl in zip(eval(e + '_x'), eval(e + '_ne'), eval(e + '_label')):
            print 'WORD'.rjust(wlength), 'LABEL'.rjust(wlength)
            for wx, la in zip(sw, sl):
                tempstr = idx2w[wx] + ' ' + idx2la[la]
                f = open('word2label.txt', 'a')
                f.write(tempstr + '\n')
                i = i + 1
                print idx2w[wx].rjust(wlength), idx2la[la].rjust(wlength)
            print '\n' + '**' * 30 + '\n'
            f.close()
    print i




       # pdb.set_trace()
