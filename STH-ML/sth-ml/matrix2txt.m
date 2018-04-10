function matrix2txt(A,filename)
% 将矩阵写入txt文件中

B = mat2str(A);
% 将空格替换为逗号

index = find(B == ';');
% 将分号变成空格
B(index) = ' ';

% 拼接文件路径
outfile = ['data/',filename];
f = fopen(outfile,'w');
h = 2;
% 分别输出B矩阵中的每一行
for fi = 1:length(index)
    fprintf(f,'%s\r\n',B(h:index(fi)));
    h = index(fi) + 1;
end
fprintf(f,'%s\r\n',B(h:end-1));
fclose(f);

end