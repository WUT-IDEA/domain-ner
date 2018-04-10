function M = knnMat(D,k,bSelf)
%knnMat Create the k-nearest-neighbours matrix based on the distance matrix
%   D:  the nTrain x nTest distance matrix
%   k:  the number of nearest neighbours
%   bSelf:  whether self is excluded (default false)

if ~exist('bSelf','var')
    bSelf = false;
end
if (size(D,1) ~= size(D,2))
    bSelf = false;
end

if bSelf
    D(logical(eye(size(D)))) = Inf; % eye(size(D))：转换成m*n的单位矩阵，logical-将矩阵中的非零元素变成0，那么整句话的意思是将矩阵D中对应单位矩阵的位置的元素置为Inf(无穷大)
end

[Y,I] = sort(D); % 按列排序，Y是排序后的结果矩阵，I是对应元素排序前所在的行号

M = false(size(D));
for j = 1:size(D,2)
    M(I(1:k,j),j) = true;
end

end
