#coding=utf-8
__auther__='sunyueqing'

import re
import numpy
import time
import math
import random
pattern = re.compile(r'.*\d+')
dictionary={}
label2idx={'nr':0, 'ns':1, 'nt':2, 'nz':3, 'Tg':4, 'Ng':5, 'j':6, 'm':7, 'n':8, 'na':9, 't':10, 'o':11}
idx2label=dict((k,v) for v,k in label2idx.iteritems())
path="../data/"
#select all words except last 300 words(23624-300) into index save into dic{ word:idx, word:idx }
def get_dictionary(txt):
    dic={}
    i = 0
    s = open(path+txt)
    for line in iter(s):
        line = line[:-1]
        dic[line] = i
        i = i + 1
    dic['UNKNOWN'] = i
    s.close()
    return dic

def transfor2train(txt,xfile,labelfile):
    sentence="[ "
    label="[ "
    f=open(path+txt)
    for line in iter(f):
        if(line=='\n'):
            sentence+=']\n[ '
            label+= ']\n[ '
        else:
            temp=line.strip('\n').split(" ")
            if(dictionary.has_key(temp[0])):
                sentence += str(dictionary.get(temp[0])) + " "
            else:
                sentence += str(dictionary.get('UNKNOWN')) + " "
            if(label2idx.has_key(temp[1])):
                label += str(label2idx.get(temp[1])) + " "
            else:
                label += str(label2idx.get('o')) + " "

    f.close()
    x = open(path+xfile, 'w')
    x.write(sentence)
    y=open(path+labelfile,'w')
    y.write(label)
    x.close()
    y.close()

def transfor2idx(txt,outfile):

    f=open(path+txt)
    total=""
    for line in iter(f):
        sentence = "[ "
        splits=line.strip('\n').split(" ")
        if len(splits)<3:continue
        for s in splits:
            if s=="": continue
            if s==" ":continue
            else:
                if (dictionary.has_key(s)):
                    sentence+=str(dictionary.get(s))+" "
                else: sentence+=str(dictionary.get('UNKNOWN'))+" "
        sentence+="]"
        total+=sentence+'\n'
        # if(line=='\n'):
        #     sentence+=']\n[ '
        # else:
        #     temp=line.strip('\n').split(" ")
        #     if(dictionary.has_key(temp[0])):
        #         sentence += str(dictionary.get(temp[0])) + " "
        #     else:
        #         sentence += str(dictionary.get('UNKNOWN')) + " "
    f.close()
    x = open(path+outfile, 'w')
    x.write(total)
    x.close()

def split2sentence(file,outfile):
    f=open(path+file,'r')
    out=open(path+outfile,'a')
    for line in iter(f):
        line=line.strip('\n')
        if line.find('。')>-1:
            split=line.split('。')
            temp=''
            for s in split:
                if s=="": continue
                else: temp+=s+'\n'
            out.write(temp)
        else:
            out.write(line+'\n')
    f.close()
    out.close()

def get_datalist(file):
    x = open(file)
    train_lex = []
    count=0
    for line in iter(x):
        count+=1
        line = line[:-1]
        b = line.strip('\n').strip('[').strip(']').split(' ')
        #print count
        c=[]
        for i in range(0,len(b)):
            if b[i]=='':continue
            if b[i]== ' ': continue
            b[i]=b[i].strip('\n')
            c.append(int(b[i]))

        line = numpy.array(c)
        train_lex.append(line)
    x.close()
    return train_lex

def toarray(string):
    line=string.strip('\n').split(' ')
    train_lex = []
    for e in line:
        if e=='':continue
        train_lex.append(float(e))

    row = numpy.array(train_lex)
    return row

def get_emb(ne,de):
    embeding=numpy.empty((ne+1,de))
    f=open(path+'word_embedings.txt')
    i=0
    for line in iter(f):
        sp=line.strip('\n').strip(' ').split(' ')[0]
        emb=toarray(line.strip(sp))
        embeding[i]=emb
        i+=1
    embeding[i]=numpy.random.uniform(-1,1,(1,de))
    return embeding

#for each line in trainx_useful,each word corresponding a embedding,so a sentence is a matrix.
# return these vectors' mean
embedding=get_emb(13451,200)
def getSentenceMeanVector(sentence):
    l=[]
    sentence=sentence.strip('\n').strip('[').strip(']').split()
    for e in sentence:
        l.append(embedding[int(e)])
    matrix=numpy.array(l)
    return matrix.mean(axis=0)

def cos(vec1,vec2): #similarity
    l = numpy.sqrt(vec1.dot(vec1)) * numpy.sqrt(vec2.dot(vec2))
    return vec1.dot(vec2) / l

def getcorpus():
    f=open(path+'zhengzhi_useful.txt')
    l=[]
    for line in f:
        vector=getSentenceMeanVector(line)
        l.append(vector)
    f.close()
    return l
zhengzhicorpus=getcorpus()
#calculate similarity between source sentence and zhengzhi corpus(polynomial kernel)
def calSimilarity(sentence):
    sum=0.0
    assert len(zhengzhicorpus)==15584
    for i in zhengzhicorpus:
        similar=cos(getSentenceMeanVector(sentence),i)
        sum+=similar
        # print similar
    return sum/15584

def calSimilarityRBF(sentence):
    sum=0.0
    assert len(zhengzhicorpus)==15584
    for i in zhengzhicorpus:
        similar=math.exp(-numpy.linalg.norm(getSentenceMeanVector(sentence)-i)/12)
        sum+=similar
        # print similar
    return sum/15584

#取原始训练文件file的第start至end个sentence作为crf的train/test，存入outfile中
def get_crf(file, istrain, start, end, outfile):
    f=open(path+file)
    count=1
    s=''
    for line in iter(f):
        if line=='\n':
            count+=1
        if count>end+1:
            break
        if count>start-1:
            if istrain:
                s+=line
            else:
                if line=='\n': s+=line
                else: s+=line.split(' ')[0]+'\n'
    f.close()
    out=open(path+outfile,'w')
    out.write(s)
    out.close()

def get_crf_train():
    f=open(path+'train.txt')
    s=''
    for line in iter(f):
        if(line=='\n'): s+=line
        else:
            sp=line.strip('\n').split(' ')
            s+=sp[0]+' '+sp[1]+' '+sp[1]+'\n'
    f.close()
    o=open(path+'train3.txt','w')
    o.write(s)
    o.close()


def get_useful_train(trainx,trainy):
    x=open(path+trainx)
    y=open(path+trainy)
    dicx={}
    dicy={}
    i=0
    j=0
    for t in iter(x):
        dicx[i]=t.strip('\n')
        i+=1
    for z in iter(y):
        dicy[j]=z.strip('\n')
        j+=1
    assert i==j
    x.close()
    y.close()
    print 'before:length=', i
    xx=open(path+"trainx_useful.txt",'w')
    yy=open(path+"trainy_useful.txt",'w')
    for key in range(0,i):
        count=0.0
        temp=dicx.get(key).strip('\n').strip('[').strip(']').strip(" ").split(" ")
        if len(temp)<3: continue
        for c in range(0,len(temp)):
            if temp[c]=="13450": count+=1
        precent=count/len(temp)
        if precent>=0.5:continue
        xx.write(dicx.get(key)+'\n')
        yy.write(dicy.get(key)+'\n')
    xx.close()
    yy.close()


'''
 transfor the rnn_xpredict result into vocabulary number [12 3434 6435]
'''
def transforx(line):
    splits=line.strip('\n').strip('\r').strip('[').strip(']').split(' ')
    re = '[ '
    for k in splits:
        if k == '': continue
        re += k.strip(',') + ' '
    re += ']'
    return re

'''
 transfor the rnn_ypredict result into label number [11 6 8 2 11 11]
'''
def transfory(line):
    splits = line.strip('\n').strip('\r').strip('[').strip(']').split(' ')
    re = '[ '
    for k in splits:
        if k == '': continue
        temp= k.strip(',').strip("'")
        l=label2idx[temp]
        re+=str(l)+' '
    re += ']'
    return re


dictionary=get_dictionary('dictionary.txt')

def countRepeat(file1,file2):
    f1=open(file1)
    f2=open(file2)
    dic1={}
    count=0
    # for e1, e2 in zip(f1,f2):
    #     e1=e1.strip('\n').strip('\r')
    #     e2 = e2.strip('\n').strip('\r')
    #     if not dic1.has_key(e1):
    #         dic1[e1]=1
    #     if not dic1.has_key(e2):
    #         dic1[e2]=1
    #     else: count+=1
    # print count,len(dic1)
    for i in f1:
        i=i.strip('\n').strip('\r')
        dic1[i]=1
    for j in f2:
        j=j.strip('\n').strip('\r')
        if dic1.has_key(j):
            count+=1
        else: dic1[j]=1
    print count,len(dic1)
    assert count==1600-len(dic1)
    f1.close()
    f2.close()

def remove_repeat():
    hx=open('../data/human-trainx.txt')
    predx=open('../data/top4000')
    predy=open('../data/top4000y.txt')
    dic1={}
    count=0
    dic2={}
    for x in hx:
        dic1[x.strip('\n').strip('\r')]=1
    assert len(dic1)==3043
    hx.close()
    for tx,ty in zip(predx,predy):
        tx=tx.strip('\n').strip('\r')
        ty = ty.strip('\n').strip('\r')
        if dic1.has_key(tx):count+=1
        else:
            dic2[tx]=ty
    assert len(dic2)+count==4200
    print count
    predx.close()
    predy.close()

    rx=open('../data/topx.txt','w')
    ry=open('../data/topy.txt','w')
    for e in dic2.keys():
        rx.write(e+'\n')
        ry.write(dic2.get(e)+'\n')
    rx.close()
    ry.close()

def get_pretrain():
    big=open('../data/zhengzhi_useful-3043.txt')
    small=open('../data/topx.txt')
    r=open('../data/pretrain.txt','w')
    l=[]
    for i in big:
        l.append(i.strip('\n'))
    big.close()
    for j in small:
        j=j.strip('\n')
        assert j in l
        l.remove(j)
    print len(l)
    for k in l:
        r.write(k+'\n')
    r.close()

def getdata_byidx():
    x=open('scoreRBF.txt')
    idx=[]
    for i in x:
        tt=i.strip('\n').strip('\r').split()[0]
        assert int(tt) in range(1,4819)
        idx.append(tt)
    x.close()
    y=open(path+'trainy_useful.txt')
    print len(idx)
    dic={}
    count=1
    for line in y:
        dic[count]=line.strip('\n')
        count+=1
    assert count==4819
    assert len(dic)==4818
    y.close()
    sorted=open('sorted_yRBF.txt','w')
    for k in idx:
        sorted.write(dic[int(k)]+'\n')
    sorted.close()


if __name__=="__main__":
    print 'a'
    # transfor2train('train.txt','trainx2.txt','trainy2.txt')
    # transfor2idx('zhengzhi_segment.txt','zhengzhi_idx.txt')
    # split2sentence('zhengzhi.txt','zhengzhi_sentences.txt')
    # get_crf('train.txt',True,1,15587,'crf_train.txt')
    # get_crf_train()
    # get_useful_train('trainx.txt','trainy.txt')
    # transfor2train('autolabel.txt', 'human-trainx.txt', 'human-trainy.txt')
    # remove_repeat('zhengzhi_predy.txt','zhengzhi_predy2.txt')
    # remove_repeat()
    # get_pretrain()

    # f=open(path+'trainx_useful.txt')
    # dic={}
    # dic2={}
    # count=0
    # start=time.time()
    # for i in f:
    #     # if count>5:break
    #     count+=1
    #     t=calSimilarity(i)
    #     dic[i]=t
    #     dic2[count]=t
    #     print 'calculating the %d sentence...'%count
    # f.close()
    # print count
    # print 'end', time.localtime(time.time())
    # print 'costs seconds:', (time.time()-start)/60.0
    # sort_dicscore = sorted(dic2.items(), key=lambda d: d[1], reverse=True)  # sort by score
    # sort_dicsentence = sorted(dic.items(), key=lambda d: d[1], reverse=True)  # sort by score
    # s = open('score.txt', 'w')
    # for i in sort_dicscore:
    #     s.write(str(i[0]) + ' ' + str(i[1]) + '\n')
    # s.close()
    # y = open('sorted_sentences.txt', 'w')
    # for j in sort_dicsentence:
    #     y.write(str(j[0]))
    # y.close()
    # getdata_byidx()

    x=open('newx.txt')
    y=open('newy.txt')
    dic={}
    count=0
    for i,j in zip(x,y):
        i=i.strip('\n')
        j = j.strip('\n')
        count+=1
        dic[count]=[i,j]
    x.close()
    y.close()
    print len(dic)
    xx=open('newxfinal.txt','w')
    yy = open('newyfinal.txt', 'w')
    l=range(1,len(dic)+1)
    random.shuffle(l)
    for i in l:
        xx.write(dic.get(i)[0]+'\n')
        yy.write(dic.get(i)[1]+'\n')
    xx.close()
    yy.close()

    # f=open('z.txt')
    # z=open('zz.txt','w')
    # for i in f:
    #     i=i.strip('\n')
    #     for j in range(0,30):
    #         z.write(i+'\n')
    # z.close()
    # f.close()



