function STH_LEAD(dataset, alg)
% STH_LEAD: Learning to Hash. 第一阶段使用STH生成二进制编码，第二阶段使用多标签分类算法LEAD来生成测试集标签

%% 
load(['testbed/',dataset]);
f_learn = eval(['@',alg,'_learn']);
f_compress = eval(['@',alg,'_compress']);

%%
codeLen = 4:2:32;        % 编码长度（1*16的矩阵），从4开始到64为止的等差数列，公差是4
hammRadius = 0;        % [0,1,2,3]
maxbits = codeLen(end);  % 64
tic;                     % 保存当前时间
[model, codeTrain] = f_learn(feaTrain, gndTrain, k, metric, kernel, maxbits);
timeTrain = toc;


tic;
svm.type = 'Linear';
svm.para = []; 
DAG = zeros(maxbits,maxbits);
% Outputs存储预测的结果，矩阵的每一位都是double类型的值，表示属于此类的概率
[Outputs,Pre_Labels]=LEAD(feaTrain,codeTrain',feaTest,zeros(size(feaTest,1),maxbits)',DAG,svm);
codeTest = Outputs>0.5;          % 将大于0.5的值转换为1，小于等于0.5的变成0
codeTest = codeTest';    % 求得的结果是维度*n的矩阵，需要转置
timeTest = toc;
disp([timeTrain, timeTest]);

%%
m = length(codeLen);
n = length(hammRadius);
trueP = zeros(m,n);
trueR = zeros(m,n);
cateP = zeros(m,n);
cateR = zeros(m,n);
cateA = zeros(m,n);

generateLineNumber = zeros(m,size(gndTest,1));
matrix2txt(codeTrain,'codeTrain');
% matrix2txt(codeTest,'codeTest')
matrix2txt(gndTrain,'gndTrain.txt');
times = zeros(1,m);

for i = 1:m
    tic;
    nbits = codeLen(i);
    cbTrain = compactbit(codeTrain(:,1:nbits));
    cbTest  = compactbit(codeTest(:,1:nbits));
    hammTrainTest  = hammingDist(cbTest,cbTrain)';
    matrix2txt(cbTrain,'cbtrain');
    matrix2txt(cbTest,'cbtest');
    [line,col] = size(hammTrainTest);
    disp([num2str(line),'----',num2str(col)])  
    for j = 1:n
        Ret = (hammTrainTest <= hammRadius(j)+0.00001); % 将汉明距离小于等于给定值的位置元素置为1
        gndTrain = gndTrain';
        for k=1:col
            f1 = find(Ret(:,k));  % 找出每一列有多少元素满足汉明距离条件
            if(length(f1) == 0)   % 不存在汉明距离为0的情况
                g1 = 1;
            else g1 = f1(1);           % 获取第一个满足条件的元素的行号
            end
            Ret(:,k) = zeros(1,line);
            Ret(g1,k) = 1;        % 只保留一个满足条件的位置，将其他位置置为0，避免后续复制出现元素个数不一致
            if(length(gndTrain(Ret(:,k)))>0)
                generateLineNumber(i,k) = gndTrain(Ret(:,k));
            else generateLineNumber(i,k) = 0;
            end
        end        
        %save(['data/','linenumber.txt'],'generateLineNumber')
        %[trueP(i,j), trueR(i,j)] = evaluate_macro(trueTrainTest, Ret);
        %[cateP(i,j), cateR(i,j)] = evaluate_macro(cateTrainTest, Ret);
        %cateA(i,j) = evaluate_classification(gndTrain, gndTest, Ret);
    end
    times(i) = toc;
end

matrix2txt(generateLineNumber','generatedLineNumber.txt');   % 将生成的行号写入文件
matrix2txt(gndTest,'test.txt');  % 将测试集行号写入文件
matrix2txt(times','time');

%trueF1 = F1_measure(trueP, trueR);
%cateF1 = F1_measure(cateP, cateR);

%%
%clear feaTrain feaTest;
%clear gndTrain gndTest;
%clear trueTrainTest cateTrainTest;
%clear hammTrainTest Ret;
%save(['results/',dataset,'_',alg]);
%clear;

end
