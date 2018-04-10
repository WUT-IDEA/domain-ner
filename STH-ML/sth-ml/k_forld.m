function k_forld(k,A)

[row,col] = size(A);
randIdx = rand(row,1);
randValue = cell(k,1);
for i = 1:k
	randValue{i} = A(randIdx <= i/k & randIdx > (i-1)/k,:);
end

for i = 1:k
	B = zeros(size(A));	
	sizeI = size(randValue{i},1);
	fl_i = zeros(size(sizeI,1),2);
	ma = [0 1;1 0];
	for l = 1:sizeI
		if(randValue{i}(l,col) == 0)
			fl_i(l,:) = ma(1,:);
		else
			fl_i(l,:) = ma(2,:);
		end
	end
	unlabeled = zeros(row-sizeI,1);
	B(1:sizeI,:) = randValue{i};

	s = 0;
	for j = 1:i-1		
		sizeJ = size(randValue{j},1);
		B(sizeI+1:sizeI+sizeJ,:) = randValue{j};
		unlabeled(s+1:s+sizeJ) = randValue{j}(1:sizeJ,col);
		sizeI = sizeI+sizeJ;
		s = s+sizeJ;
	end
	for j = i+1:k
		sizeJ = size(randValue{j},1);
		B(sizeI+1:sizeJ+sizeI,:) = randValue{j};
		unlabeled(s+1:s+sizeJ) = randValue{j}(1:sizeJ,col);
		sizeI = sizeI+sizeJ;
		s = s+sizeJ;
	end

	[W, elapse] = constructW(B);
	[fu,fu_CMN] = harmonic_functionm(W, fl_i);
	C = zeros(size(fu,1));
	D = zeros(size(fu_CMN,1));

	% 处理预测的结果
	for l = 1:size(C,1)
		if(fu(l,1) > fu(l,2))
			% 1 0
			C(l) = 1;
		else C(l) = 0;
		end
		if(fu_CMN(l,1) > fu_CMN(l,2))
			D(l) = 1;
		else D(l) = 0;
		end
	end
	[m1,n1] = size(fu);
	n_fu_p = length(find(C == 1));  % TP+FP
	n_fucmn_p = length(find(D == 1));

	n1_fu_p = length(find(unlabeled == 1));     % TP+FN
	n2_fucmn_p = length(find(unlabeled == 1));
	index1 = find(unlabeled == 1);
	E = unlabeled(index1);
	F1 = fu(index1);
	F2 = fu_CMN(index1);
	err_fu = length(find(E-F1));
	err_fucmn = length(find(E-F2));

	precision_fu = (m1-err_fu)/n_fu_p;
	precision_fucmn = (m1-err_fucmn)/n_fucmn_p;

	recall_fu = (m1-err_fu)/n1_fu_p;
	recall_fucmn = (m1-err_fucmn)/n2_fucmn_p;

	acc_fu = (m1-length(find(unlabeled-fu)))/m1;
	acc_fucmn = (m1-length(find(unlabeled-fu_CMN)))/m1;

end

end