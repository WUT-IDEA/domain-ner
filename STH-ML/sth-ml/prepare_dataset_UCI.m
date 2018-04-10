function prepare_dataset_UCI(dataset,k)

%%
load(['data/',dataset]);

%if (strcmp(dataset,'TDT2'))
if (strcmp(dataset,'STH_data_7Dim'))
%if (strcmp(dataset,'Australian'))
%if (strcmp(dataset,'German'))
%if (strcmp(dataset,'Japanese'))
    [feaTrain,feaTest] = tf_idf(feaTrain,feaTest);
    feaTrain = normalize(feaTrain);
    feaTest  = normalize(feaTest);
    metric = 'Cosine';
    kernel = 'Linear';
end

clear fea;
numTrain = size(feaTrain,1);
numTest  = size(feaTest,1);


%%

% the ground-truth k nearest neighbours
trueTrainTest = knnMat(EuDist2(feaTrain,feaTest),k,false);

% the documents in the same category 
cateTrainTest = (repmat(gndTrain,1,numTest) == repmat(gndTest,1,numTrain)');

save(['testbed/',dataset],'feaTrain','feaTest','gndTrain','gndTest','k','metric','kernel','trueTrainTest','cateTrainTest');
clear;

end
