#coding=utf-8
__auther__='sunyueqing'

import re
import numpy

pattern = re.compile(r'.*\d+')
dictionary={}
label2idx={'nr':0, 'ns':1, 'nt':2, 'nz':3, 'Tg':4, 'Ng':5, 'j':6, 'm':7, 'n':8, 'na':9, 't':10, 'o':11}
# path="/usr/seands/syqExp/is13/data/"
path='is13/data/sougou/'
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
        b = line.strip('[').strip(']').split(' ')
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
def removeMeaninglessSen(trainx,trainy,newx,newy):
    x=open(path+trainx)
    y=open(path+trainy)
    nx=open(path+newx,'w')
    ny = open(path+newy, 'w')
    count=0.0
    county=1
    countx=0
    ly={}
    for j in y:
        ly[county]=j
        county+=1
    y.close()
    print 'trainy len:',len(ly)
    for i in x:
        countx+=1
        count = 0.0
        w=i.strip('\n').strip('[ ').strip(' ]').split(" ")
        if len(w)<5:continue
        for ww in w:
            if ww=='16514':
                count+=1
        if count/len(w)>0.4: continue
        nx.write(i)
        ny.write(ly.get(countx))
        print 'writing ',countx,'th sentences'
    assert countx+1==county
    x.close()
    nx.close()
    ny.close()



dictionary=get_dictionary('sougou-dictionary2.txt')


def getDicEmb(dic, embfile, embindic):
    f = open(path+embfile)
    d=open(path+dic)
    o=open(path+embindic,'w')
    emblist={}
    count=0
    for i in f:
        k=i.strip('\n').strip(' ').split(" ")[0]
        emblist[k]=i.strip('\n').strip(' ').strip(k).strip(" ")
    f.close()
    for word in iter(d):
        count+=1
        key=word.strip('\n')
        if not emblist.has_key(key):
            print key
            print 'line ',count
        else:
            o.write(key+" "+emblist.get(key)+'\n')
    o.close()
    d.close()




if __name__=="__main__":
    print 'a'
    # transfor2train('train.txt','trainx.txt','trainy.txt')
    # transfor2idx('zhengzhi_segment.txt','zhengzhi_idx.txt')
    # split2sentence('zhengzhi.txt','zhengzhi_sentences.txt')
    # get_crf('train.txt',True,1,15587,'crf_train.txt')
    # get_crf_train()
    # print getSentenceMeanVector('[12 3 45]')

    #sougou expriment
    print len(dictionary)
    # transfor2train('sougou-train.txt', 'sougou-trainx.txt', 'sougou-trainy.txt')
    # removeMeaninglessSen('sougou-trainx.txt', 'sougou-trainy.txt','trainx.txt','trainy.txt')
    getDicEmb("sougou-dictionary2.txt","sougouword_embedding.txt","embeddings.txt")