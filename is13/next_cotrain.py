#coding:utf-8
#handle the train data transformation between rnn and crf model
import math,array,numpy,nltk
def rnn2crf3(trainx, trainyy, crfresult):
    f=open(trainx)
    trainy = open(trainyy)
    r=open(crfresult,'w')
    count=0
    for x,y in zip(f,trainy):
        # print x,y
        col1=x.strip('\n').strip('\r').strip('[').strip(']').split()
        col2=y.strip('\n').strip('\r').strip('[').strip(']').split()
        for o,w in zip(col1,col2):
            r.write(o+' '+w+'\n')
        r.write('\n')
        count+=1
    r.close()
    f.close()
    trainy.close()
    print 'transfored %d sentences' % count

def rnn2crf2(trainx, crfresult):
    f=open(trainx)
    r=open(crfresult,'w')
    count=0
    for x in f:
        # print x,y
        col1=x.strip('\n').strip('\r').strip('[').strip(']').split()
        for o in col1:
            r.write(o+'\n')
        r.write('\n')
        count+=1
    r.close()
    f.close()
    print 'transfored %d sentences' % count

def crf2rnn(crf, trainx, trainy):
    f=open(crf)
    rnnx = open(trainx,'w')
    rnny = open(trainy,'w')
    sen=''
    label=''
    count=0
    for line in f:
        if '\n' != line:
            col=line.strip('\n').strip('\r').split('\t')
            sen+=col[0]+' '
            label+=col[1]+' '
        else:
            rnnx.write('[ '+sen+']'+'\n')
            rnny.write('[ ' + label + ']' + '\n')
            sen=''
            label=''
            count+=1
    rnnx.close()
    rnny.close()
    f.close()
    print 'transfored %d sentences' % count


#crf model:
def getK_high_score(pred_file, k):
    '''pred_file: CRF pred result with score;
       output: return sorted_dict{index:[sentence,labels,score]} that length=k'''
    f=open(pred_file)
    dic={} # {1:[sentence,label,score]}
    score=0
    sen=''
    label=''
    index=0

    for line in iter(f):
        if '\n'!= line:
            temp=line.strip('\n').split(' ')
            if temp[0]=='#': #score line
                score=temp[1]
                continue
            temp=line.strip('\n').split('\t')
            sen+=temp[0]+' '
            label+=temp[1].split('/')[0]+' '
        else:
            dic[index]=[sen.strip(' '), label.strip(' '), float(score)]
            index+=1
            # print sen,'--',label
            sen=''
            label=''
    f.close()
    sort_dic=sorted(dic.items(),key=lambda d: d[1][2],reverse=True) # sort by score
    print 'pre_train length:', len(sort_dic)
    if len(sort_dic)<k:
        return sort_dic
    else: return sort_dic[0:k] ## [(1,[sentence,label,score]),(),()]

def getcrf_pred(crfscorefile,topn):
    # sorted_list=getK_high_score('train ribao/zhengzhi-score-result', 2000)
    sorted_list = getK_high_score(crfscorefile, topn)
    x=open('data/exp3/crftop800trainx.txt','w')
    y=open('data/exp3/crftop800trainy.txt','w')
    for item in sorted_list:
        x.write('[ '+item[1][0]+' ]'+'\n')
        y.write('[ '+item[1][1]+' ]'+'\n')
    x.close()
    y.close()

def my_evaluate(resultfile):
    f=open(resultfile)
    total=0
    # list=['nr', 'ns', 'nt', 'nz', 'Tg', 'Ng', 'j', 'm', 'n', 'na', 't']
    # list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    list = ['B_Thing', 'I_Thing', 'B_Person', 'I_Person', 'B_Location', 'I_Location', 'B_Time', 'I_Time', 'B_Metric', 'I_Metric', 'B_Organization','I_Organization','B-Abstract','I_Abstract']
    all_identied_entity=0 #识别出的实体数
    correct_identied_entity=0 #正确识别的个体数
    total_entity=0  #实际上的所有实体数
    for line in iter(f):
        s=line.strip('\n').split()
        splits=[]
        for i in s:
            if i=='':continue
            splits.append(i)
        if len(splits)==3:
            total = total + 1
            if splits[2] in list:
                all_identied_entity+=1
                if splits[1]==splits[2]:
                    correct_identied_entity+=1
            if splits[1] in list:
                total_entity+=1

    precision=correct_identied_entity*1.0/all_identied_entity
    recall=correct_identied_entity*1.0/total_entity
    f1=2.0*precision*recall/(precision+recall)
    print 'correct_identied_entity:',correct_identied_entity,'all_identied_entity:',all_identied_entity,'total_entity:',total_entity
    print 'total:',total
    print 'precision:',precision,'recall:', recall,'F1:', f1
    f.close()
    return {'p':precision, 'r':recall, 'f1':f1}

def update_pretrain():
    f=open('data/cotrain/pretrain.txt')
    l=[]
    for i in f:
        l.append(i)
    print 'before remove predx,pretrain length=',len(l)
    f.close()
    rnn=open('data/cotrain/rnn_predx.txt')
    crf=open('data/crf/crftop800trainx.txt')
    for r,c in zip(rnn,crf):
        if r in l:
            l.remove(r)
        if c in l:
            l.remove(c)
    print 'after remove predx,pretrain length=', len(l)
    rnn.close()
    crf.close()
    new=open('data/cotrain/new_pretrain.txt','w')
    for i in l:
        new.write(i)
    new.close()

def update_pretrain2():
    f=open('data/exp3/new_pretrainx.txt')
    l=[]
    for i in f:
        l.append(i)
    print 'before remove predx, pretrain length=',len(l)
    f.close()
    rnn=open('data/exp3/rnn_predx.txt')
    crf=open('data/exp3/crftop800trainx.txt')
    for r,c in zip(rnn,crf):
        if r in l:
            l.remove(r)
        if c in l:
            l.remove(c)
    print 'after remove predx,pretrain length=', len(l)
    rnn.close()
    crf.close()
    new=open('data/exp3/new_pretrainx.txt','w')
    for i in l:
        new.write(i)
    new.close()

def count():
    rnn = open('data/exp3/rnn_predx.txt')
    crf = open('data/exp3/crftop800trainx.txt')
    # rnn = open('data/human-trainx.txt')
    # crf = open('data/top2000')
    l=[]
    count=0
    for i in rnn:
        l.append(i)
    # assert len(l)==800
    for j in crf:
        if j in l:
            count+=1
    print count
    rnn.close()
    crf.close()





if __name__=='__main__':

    path1='data/cotrain/a/'

    path4 = 'data/crf/cotrain-group4/originalRNN/'
    path44 = 'data/crf/cotrain-group4/modifiedRNN/'

    path5 = 'data/crf/cotrain-group5/originalRNN/'
    path55 = 'data/crf/cotrain-group5/modifiedRNN/'

    p='data/exp3/'

    # rnn2crf3(p+'trainx.txt',p+'trainy.txt',p+'train.txt')
    # rnn2crf2(p + 'new_pretrainx.txt', p+'pretraincrf.txt')

    # update_pretrain2()
    # count()

    #
    my_evaluate(p+'result1.txt') #evaluate crf result
    # getcrf_pred(p+'score-result.txt',800)