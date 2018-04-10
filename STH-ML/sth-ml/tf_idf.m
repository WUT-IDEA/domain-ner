function [trainX, testX] = tf_idf(trainTF, testTF)
% TF-IDF weighting
% ([1+log(tf)]*log[N/df])

[n,m] = size(trainTF);  % the number of (training) documents and terms

df = sum(trainTF>0);  % (training) document frequency

d = sum(df>0); % the number of dimensions, i.e., terms occurred in (training) documents
[dfY, dfI] = sort(df, 2, 'descend'); % dfY是按行降序排序后的1*m矩阵，dfI是矩阵dfY元素对应原矩阵df的索引位置，这里的意思是按照文档频率降序
trainTF = trainTF(:,dfI(1:d)); % 对于矩阵trainTF中的每一行，按照deI的索引顺序输出每一个列元素，即按照文档顺序降序
testTF = testTF(:,dfI(1:d)); % 对testTF（测试数据）的每一列，同样按照词频重新排序
idf = log(n./dfY(1:d)); % 求IDF，即文档数n分别除以1*m矩阵的每一个元素（上文求得的df）
IDF = sparse(1:d,1:d,idf); % 将行向量idF转换成对角阵

[trainI,trainJ,trainV] = find(trainTF); % 获取矩阵trainTF中的非0元素，trainI记录非零元素行号，trainJ记录列号，trainV记录对应的元素，都是n*1的矩阵
trainLogTF = sparse(trainI,trainJ,1+log(trainV),size(trainTF,1),size(trainTF,2));
[testI,testJ,testV] = find(testTF);
testLogTF = sparse(testI,testJ,1+log(testV),size(testTF,1),size(testTF,2)); % 生成一个矩阵，行数和列数与testTF矩阵相同，分别取testI,testJ作为新矩阵的行列坐标，值是1+log(testV)的计算结果

trainX = trainLogTF*IDF;
testX = testLogTF*IDF;

end
