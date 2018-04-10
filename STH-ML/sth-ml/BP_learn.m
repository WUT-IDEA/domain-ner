function BP_learn(dataset, alg)

% GRNNN_learn : 将STH的第二步——训练k个SVM分类器——修改成GRNN，一次性生成k位数字

%% 
load(['testbed/',dataset]);
f_learn = eval(['@',alg,'_learn']);
f_compress = eval(['@',alg,'_compress']);

%%
codeLen = 4:2:40;        % 编码长度（1*16的矩阵），从4开始到64为止的等差数列，公差是4
hammRadius = 0;        % [0,1,2,3]
maxbits = codeLen(end);  % 64

% 一次性生成最大位数的二进制码，后面按需截取
tic;
[model, codeTrain] = f_learn(feaTrain, gndTrain, k, metric, kernel, maxbits);
timeTrain = toc;

% BP神经网络预测输出
% net = feedforwardnet(10,'trainlm');
% net = train(net,feaTrain',codeTrain');
% net.trainParam.goal=0.001;    % goal是最小均方误差的训练目标
% net.trainParam.epochs=500;    % epochs是较大训练次数（迭代次数）
% code_test_d = sim(net,feaTest');
% codeTest = code_test_d>0.5;          % 将大于0.5的值转换为1，小于等于0.5的变成0
% codeTest = codeTest';    % 求得的结果是 维度*n 的矩阵，需要转置

tic;
net = newff(feaTrain',codeTrain');
net = train(net,feaTrain',codeTrain');
code_test_d = sim(net,feaTest');
codeTest = code_test_d > 0.5;
codeTest = codeTest';
timeTest = toc;
disp([timeTrain, timeTest]);

%%
m = length(codeLen);
n = length(hammRadius);

generateLineNumber = zeros(m,size(gndTest,1));
% matrix2txt(codeTrain,'codeTrain');
% matrix2txt(codeTest,'codeTest')
matrix2txt(gndTrain,'gndTrain.txt');
times = zeros(1,m);

for i = 1:m
    tic;
    nbits = codeLen(i);    

    cbTrain = compactbit(codeTrain(:,1:nbits));
    cbTest  = compactbit(codeTest(:,1:nbits)); 
    hammTrainTest  = hammingDist(cbTest,cbTrain)';
    % matrix2txt(cbTrain,'cbtrain');
    % matrix2txt(cbTest,'cbtest');
    [line,col] = size(hammTrainTest);
    disp([num2str(line),'----',num2str(col)])  
    for j = 1:n
        Ret = (hammTrainTest <= hammRadius(j)+0.00001); % 将汉明距离小于等于给定值的位置元素置为1
        gndTrain = gndTrain';
        for k=1:col
            f1 = find(Ret(:,k));  % 找出每一列有多少元素满足汉明距离条件
            if(length(f1) == 0)
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
    end
    times(i) = toc;
end

matrix2txt(generateLineNumber','generatedLineNumber.txt');   % 将生成的行号写入文件
matrix2txt(gndTest,'test.txt');  % 将测试集行号写入文件
matrix2txt(times','time');

%%
%clear feaTrain feaTest;
%clear gndTrain gndTest;
%clear trueTrainTest cateTrainTest;
%clear hammTrainTest Ret;
%save(['results/',dataset,'_',alg]);
%clear;

end
