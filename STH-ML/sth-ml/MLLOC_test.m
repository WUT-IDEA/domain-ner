function [test_labels,test_outputs]=MLLOC_test(test_data,test_targets,train_data,train_feature,models,kernel_type,sigma)

[n_test,T]=size(test_targets);

test_feature=get_test_feature(test_data,train_data,train_feature);

if(strcmp(kernel_type,'linear'))
    kernelX=test_data*train_data';
elseif(strcmp(kernel_type,'RBF'))
    kernelX=RBF_kernel(test_data,sigma,train_data);
end
kernelS=test_feature*train_feature';

% predict
test_labels=zeros(n_test,T);
test_outputs=zeros(n_test,T);
for i=1:T
    kernel_sum=[(1:n_test)',kernelX+kernelS];
    [test_label,accuracy,test_output]=svmpredict(test_targets(:,i),kernel_sum,models{i});
    if(test_label(1)*test_output(1)<0)
        test_output=-test_output;
    end
    test_labels(:,i)=test_label;
    test_outputs(:,i)=test_output;
end

