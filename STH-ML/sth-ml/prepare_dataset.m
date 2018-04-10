function prepare_dataset(dataset,k)

%%
load(['data/',dataset]);

if (strcmp(dataset,'Reuters21578'))
    top10CatIdx = find(gnd<=10);  % only use the top 10 categories
    trainIdx = intersect(top10CatIdx,trainIdx);
    testIdx  = intersect(top10CatIdx,testIdx);
    [feaTrain,feaTest] = tf_idf(fea(trainIdx,:),fea(testIdx,:));
    feaTrain = normalize(feaTrain);
    feaTest  = normalize(feaTest);
    metric = 'Cosine';
    kernel = 'Linear';
end
if (strcmp(dataset,'20Newsgroups'))
    [feaTrain,feaTest] = tf_idf(fea(trainIdx,:),fea(testIdx,:));
    feaTrain = normalize(feaTrain);
    feaTest  = normalize(feaTest);
    metric = 'Cosine';
    kernel = 'Linear';
end
%if (strcmp(dataset,'TDT2'))
if (strcmp(dataset,'STH_data_7Dim'))
%if (strcmp(dataset,'australian'))
%if (strcmp(dataset,'german'))
%if (strcmp(dataset,'japanese'))
    split = 0.8;   % 60 percent training, 40 percent testing
    randIdx = rand(size(fea,1),1);          % 生成n*1的矩阵，每个元素取值（0,1）
    trainIdx = find(randIdx <= split);      % 获取值小于0.6的元素索引，n*1的矩阵
    testIdx  = find(randIdx >  split);      % 这两句的意思是：按照6:4的比例划分数据集
    [feaTrain,feaTest] = tf_idf(fea(trainIdx,:),fea(testIdx,:));
    feaTrain = normalize(feaTrain);
    feaTest  = normalize(feaTest);
    metric = 'Cosine';
    kernel = 'Linear';
end

matrix2txt(fea(trainIdx,:),'feaTrain');
matrix2txt(fea(testIdx,:),'feaTest');

clear fea;
numTrain = size(feaTrain,1);
numTest  = size(feaTest,1);
gndTrain = uint8(gnd(trainIdx));
gndTest  = uint8(gnd(testIdx));

%%

% the ground-truth k nearest neighbours
trueTrainTest = knnMat(EuDist2(feaTrain,feaTest),k,false);

% the documents in the same category 
cateTrainTest = (repmat(gndTrain,1,numTest) == repmat(gndTest,1,numTrain)');

save(['testbed/',dataset],'feaTrain','feaTest','gndTrain','gndTest','k','metric','kernel','trueTrainTest','cateTrainTest');
clear;

end
