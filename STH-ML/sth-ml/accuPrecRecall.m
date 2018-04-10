function accuPrecRecall(A,B)
% calculate accuracy, precision, recall etc.
% A : generated number
% B : test number

m = size(A,1);

dif = unique(B);        % 类别数
col = size(A,2);       % 不同编码长度的个数
accs = zeros(col,1);
spaces = '    ';
for j = 1:col
	% 分别统计每一类别的precision和recall
	disp([spaces,'code: ',num2str(j)]);
	ERR = nnz(A(:,j)-B);
	accuracy = 1-ERR/m;
	accs(j) = accuracy;
	disp([spaces,'accuracy: ',num2str(accuracy)]);
	for i = 1:length(dif)
		n = dif(i);                    % n表示第n类
		p_num = size(find(A(:,j) == n),1);  % TP+FP
		B_indexs_n = find(B == n);     % 找出测试集中所有第n类的位置索引
		A_n = A(B_indexs_n,j);
		B_n = B(B_indexs_n);
		e = nnz(A_n-B_n);           % 分类错误的个数
		tp = length(B_indexs_n)-e;  % TP
		precision = tp/p_num;
		recall = tp/length(B_indexs_n);
		% disp([spaces,spaces,num2str(n),'-precision: ',num2str(precision)]);
		% disp([spaces,spaces,num2str(n),'-recall: ',num2str(recall)]);
	end;
end
matrix2txt(accs,'accuracy');

end