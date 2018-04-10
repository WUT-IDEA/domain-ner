function [test_labels,test_outputs]=MLLOC(train_data,train_targets,test_data,test_targets,lambda1,lambda2,m,num_rounds,kernel_type,sigma)

% Inputs:
%   train_data      - A n x m array, The i-th instance is stored in train_data(i,:)
%   train_target    - A n x T array, T is the number of possible labels, train_target(i,j) is 1 if the i-th instance has the j-th label, and -1 otherwise
%   test_data       - A n_test x m array, n_test is the number of test instances
%   test_target     - A n_test x T array
%   lambda1         - parameter in Eq. (6), default 1
%   lambda2         - parameter in Eq. (6), default 100
%   m               - length of the loc code, default 15
%   num_rounds      - maximum rounds to perform if not converged, you can set to 1 for efficiency
%   kernel_type     - which kernel to use, options: 'RBF' and 'linear'
%   sigma           - sigma of the RBF kernel, if kernel_type='RBF'

% Outputs:
%   test_labels          - A n_test x T array, the predicted labels of the test instances
%   test_outputs         - A n_test x T array, the prediction values of the test instances

[models,train_feature]=MLLOC_train(train_data,train_targets,lambda1,lambda2,m,num_rounds,kernel_type,sigma);
    
[test_labels,test_outputs]=MLLOC_test(test_data,test_targets,train_data,train_feature,models,kernel_type,sigma);