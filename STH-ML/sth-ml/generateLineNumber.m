function generateLineNumber(cbTrain, cbTest, gndTrain)

% 生成映射文件的行号

m = 1;
generatedLineNumber = zeros(m,6038);
hammRadius = 0;        % [0,1,2,3]
hammTrainTest  = hammingDist(cbTest,cbTrain)';
n = length(hammRadius);
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
	        generatedLineNumber(1,k) = gndTrain(Ret(:,k));
	    else generatedLineNumber(1,k) = 0;
	    end
	end        
end

matrix2txt(generatedLineNumber','generatedLineNumber.txt');   % 将生成的行号写入文件

end
