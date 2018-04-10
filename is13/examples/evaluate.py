#coding=utf-8
#author=sunyueqing
def my_evaluate(resultfile):
    f=open(resultfile)
    total=0
    # list=['nr', 'ns', 'nt', 'nz', 'Tg', 'Ng', 'j', 'm', 'n', 'na', 't']
    list=['B_Time',
    'I_Time','B_Person','I_Person','B_Thing','I_Thing','B_Location','B_Metric','I_Metric','I_Location','B_Organization','I_Organization','B_Abstract','I_Abstract','B_Physical',
    'I_Physical','B_Term','I_Term','B_ABstract','I_ABstract']
    all_identied_entity=0 #识别出的实体数
    correct_identied_entity=0 #正确识别的个体数
    total_entity=0  #实际上的所有实体数
    for line in iter(f):
        splits=line.strip('\n').split()
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
    print 'precision:',precision,'recall:', recall,'F1:', f1
    f.close()
    return {'p':precision, 'r':recall, 'f1':f1}


def IOBevaluate(resultfile):
    f=open(resultfile)
    total=0
    list=[]
    la=open('../data/aa.txt')
    for l in iter(la):
        list.append(l.strip('\n').split()[1])
    la.close()
    list.remove('O')
    all_identied_entity=0 #识别出的实体数
    correct_identied_entity=0 #正确识别的个体数
    total_entity=0  #实际上的所有实体数 
    for line in iter(f):
        splits=line.strip('\n').split()
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
    print total
    print 'precision:',precision,'recall:', recall,'F1:', f1
    f.close()
    return {'p':precision, 'r':recall, 'f1':f1}

if __name__=='__main__':
    my_evaluate('best.test1.txt')