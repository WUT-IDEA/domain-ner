import random
import numpy

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [-1] + l + win/2 * [-1]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out


def x_contextwin(l, win):#use in keras
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [572] + l + win/2 * [572]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

def y_contextwin(l, win):#use in keras
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [127] + l + win/2 * [127]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

def bigram_contextwin(l, win):
    '''
       list:[11,2,6,8]
       contextwin:(win=7)
       [ [-1, -1, -1, 11, 2, 6, 8],
         [-1, -1, 11, 2, 6, 8, -1],
         [-1, 11, 2, 6, 8, -1, -1],
         [11, 2, 6, 8, -1, -1, -1]  ] #unigram
       bigram_contextwn:
       [
        [[-1, -1], [-1, -1], 11, [2, 6], [6, 8]],
        [[-1, -1], [-1, 11], 2, [6, 8], [8, -1]],
        [[-1, 11], [11, 2], 6, [8, -1], [-1, -1]],
        [[11, 2], [2, 6], 8, [-1, -1], [-1, -1]]   ]
    '''
    assert win >3
    w = win / 2
    bigram=[]
    temp = contextwin(l, win)

    for li in temp: #[12,3,4,23,5,8,2]--->[ [12,3],[3,4],23,[5,8],[8,2] ]
        item=[ li[i:i+2] for i in range(len(li)-1) ]
        assert len(item)==len(li)-1

        #del item[w]
        #item = numpy.asarray(item).astype("int32")
        #item[w - 1] = li[w]
        bigram.append(item)

    assert len(bigram) == len(l)
    #print "bigram", bigram
    return bigram

def saveIntoFile(filename,str):
    f = open(filename, 'a')
    f.write(str + '\n')
    f.close()

def listmax(list):
    index=0
    vmax=list[0]
    for i in range(len(list)):
        if list[i]>vmax:
            vmax=list[i]
            index=i
    return vmax

def fetchmax(list,limit):
    d={}
    for j in range(len(list)):
        index = 0
        vmax=list[j][0]
        for i in range(len(list[j])):
            if list[j][i] > vmax:
                vmax = list[j][i]
                index = i
        if(vmax>=limit):
            d[j]=index
    return d

def get_word_posTagging(wordlist,preword,word):
    if preword=='':
        for line in wordlist:
            line=line.split()
            if word in line:
                return line[1],line[2]
    else:
        for i in xrange(1,len(wordlist)):
            print i
            if((preword.lower()==wordlist[i-1].split()[0].lower()) & (word.lower()==wordlist[i].split()[0].lower())):
                return wordlist[i].split()[1],wordlist[i].split()[2]


def writelist(list,file):
    '''write a list[[12,34,2][12,34,2]] into file'''
    f = open(file, 'w')
    for v in list:
        line = '[ '
        for i in v:
            line += str(i) + ' '
        line += ']'
        f.write(line + '\n')
    f.close()

'''
a sample output of bigram_contextwin;  3dim
[
    [[-1, -1], [-1, -1], [-1, 554], [554, 136], [136, 194], [194, 208]],
    [[-1, -1], [-1, 554], [554, 136], [136, 194], [194, 208], [208, 102]],
    [[-1, 554], [554, 136], [136, 194], [194, 208], [208, 102], [102, 502]],
    [[554, 136], [136, 194], [194, 208], [208, 102], [102, 502], [502, 332]],
    [[136, 194], [194, 208], [208, 102], [102, 502], [502, 332], [332, 569]],
    [[194, 208], [208, 102], [102, 502], [502, 332], [332, 569], [569, 104]],
    [[208, 102], [102, 502], [502, 332], [332, 569], [569, 104], [104, 58]],
    [[102, 502], [502, 332], [332, 569], [569, 104], [104, 58], [58, 62]],
    [[502, 332], [332, 569], [569, 104], [104, 58], [58, 62], [62, 332]],
    [[332, 569], [569, 104], [104, 58], [58, 62], [62, 332], [332, 569]],
    [[569, 104], [104, 58], [58, 62], [62, 332], [332, 569], [569, 104]],
    [[104, 58], [58, 62], [62, 332], [332, 569], [569, 104], [104, 72]],
    [[58, 62], [62, 332], [332, 569], [569, 104], [104, 72], [72, 8]],
    [[62, 332], [332, 569], [569, 104], [104, 72], [72, 8], [8, 384]],
    [[332, 569], [569, 104], [104, 72], [72, 8], [8, 384], [384, 358]],
    [[569, 104], [104, 72], [72, 8], [8, 384], [384, 358], [358, 416]],
    [[104, 72], [72, 8], [8, 384], [384, 358], [358, 416], [416, -1]],
    [[72, 8], [8, 384], [384, 358], [358, 416], [416, -1], [-1, -1]],
    [[8, 384], [384, 358], [358, 416], [416, -1], [-1, -1], [-1, -1]]
]
'''