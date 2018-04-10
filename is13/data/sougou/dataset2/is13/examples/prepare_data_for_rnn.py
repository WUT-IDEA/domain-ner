#coding:utf-8
__auther__='sunyueqing'

import re
import numpy

pattern = re.compile(r'.*\d+')
dictionary={}
#select all words except last 300 words(23624-300) into index save into dic{ word:idx, word:idx }
# def get_dictionary():
#     i = 0
#     s = open('../data/all_words_in_trainset.txt')
#     for line in iter(s):
#         line = line[:-1]
#         if i > 23324: break
#         dictionary[line] = i
#         i = i + 1
#     dictionary['UNKNOWN'] = i
#     s.close()
#     return dictionary

# def get_label2idx():
#     dic={}
#     dic['I-ORG']=0
#     dic['B-ORG']=1
#     dic['I-LOC']=2
#     dic['B-LOC'] = 3
#     dic['I-MISC'] = 4
#     dic['B-MISC'] = 5
#     dic['I-PER'] = 6
#     dic['O'] = 7
#     return dic

def get_datalist(file):
    x = open(file)
    train_lex = []
    count=0
    for line in iter(x):
        count+=1
        line = line[:-1]
        b = line.strip('\n').strip('\r').strip('[').strip(']').split(' ')
        #print count
        c=[]
        for i in range(0,len(b)):
            if b[i]=='':continue
            if b[i]== ' ': continue
            b[i]=b[i].strip('\n').strip('\r')
            c.append(int(b[i]))

        line = numpy.array(c)
        train_lex.append(line)
    x.close()
    return train_lex



#for second dataset

def get_dictionary2():
    d={}
    i = 0
    s = open('allwords2.txt')
    for line in iter(s):
        line = line.strip('\n').strip('\r').strip(' ')
        d[line] = i
        i = i + 1
    s.close()
    return d
def get_label2idx2():
    d = {}
    i = 0
    s = open('alllabels.txt')
    for line in iter(s):
        line = line.strip('\n').strip('\r').strip(' ')
        d[line] = i
        i = i + 1
    s.close()
    return d


dictionary2=get_dictionary2()
label2idx=get_label2idx2()


def transfor2train(txt,xfile,labelfile):
    path='../dataset2/'
    sentence="[ "
    label="[ "
    f=open(path+txt)
    for line in iter(f):
        if(line=='\n'):
            sentence+=']\n[ '
            label+= ']\n[ '
        else:
            temp=line.strip('\n').split(" ")
            sentence += str(dictionary2.get(temp[0])) + " "
            label += str(label2idx.get(temp[1])) + " "

    f.close()
    x = open(path+xfile, 'w')
    x.write(sentence)
    y=open(path+labelfile,'w')
    y.write(label)
    x.close()
    y.close()
def my_evaluate(resultfile):
    f=open(resultfile)
    total=0
    # list=['nr', 'ns', 'nt', 'nz', 'Tg', 'Ng', 'j', 'm', 'n', 'na', 't']
    list1=['B_Time',
    'I_Time','B_Person','I_Person','B_Thing','I_Thing','B_Location','B_Metric','I_Metric','I_Location','B_Organization','I_Organization','B_Abstract','I_Abstract','B_Physical',
    'I_Physical','B_Term','I_Term','B_ABstract','I_ABstract']
    all_identied_entity=0 #识别出的实体数
    correct_identied_entity=0 #正确识别的个体数
    total_entity=0  #实际上的所有实体数
    for line in iter(f):
        splits=line.strip('\n').split()
        if len(splits)==3:
            total = total + 1
            if splits[2] in list1:
                all_identied_entity+=1
                if splits[1]==splits[2]:
                    correct_identied_entity+=1
            if splits[1] in list1:
                total_entity+=1

    precision=correct_identied_entity*1.0/all_identied_entity
    recall=correct_identied_entity*1.0/total_entity
    f1=2.0*precision*recall/(precision+recall)
    print 'correct_identied_entity:',correct_identied_entity,'all_identied_entity:',all_identied_entity,'total_entity:',total_entity
    print 'precision:',precision,'recall:', recall,'F1:', f1
    f.close()
    return {'p':precision, 'r':recall, 'f1':f1}

if __name__=='__main__':
    transfor2train('test.txt','testx.txt','testy.txt')
