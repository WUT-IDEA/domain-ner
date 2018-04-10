function [models,S]=MLLOC_train(train_data,train_targets,lambda1,lambda2,m,num_rounds,kernel_type,sigma)

[n,T]=size(train_targets);

if(strcmp(kernel_type,'linear'))
    kernelX=train_data*train_data';
elseif(strcmp(kernel_type,'RBF'))
    kernelX=RBF_kernel(train_data,sigma);
end

models=cell(1,T);

% initilize S and C
[idx,C]=kmeans(train_targets,m,'emptyaction','singleton');
S=zeros(n,m);
for i=1:m
    S(idx==i,i)=1;
end

last_obj=inf;
Ws=zeros(m,T);
for t=1:num_rounds
    % fix S and C, solve W
    kernelS=S*S';
    dec_vals=zeros(n,T);
    for i=1:T
        kernel_sum=[(1:n)',kernelX+kernelS+0.0001*eye(size(kernelX))];
        model=svmtrain(train_targets(:,i),kernel_sum,['-t 4 -c ',num2str(lambda1)]);
        [tmp1,tmp2,dec_val]=svmpredict(train_targets(:,i),kernel_sum,model);
        models{i}=model;
        % calculate Ws
        alpha_SV=model.sv_coef;
        SVs=full(model.SVs);
        tmpWs=S(SVs,:)'*alpha_SV;
        if(model.Label(1)==-1)
            tmpWs=-tmpWs;
        end
        Ws(:,i)=tmpWs;
        if(tmp1(1)*dec_val(1)<0)
            dec_val=-dec_val;
        end
        dec_val=dec_val-S*tmpWs;
        dec_vals(:,i)=dec_val;
    end
    
    % fix W and C, solve S
    for i=1:n
        % x=[xi_i,S_i1,...S_im]'
        f=[ones(T,1);lambda2*sum((repmat(train_targets(i,:),m,1)-C).^2,2)];
        A=-[eye(T),repmat(train_targets(i,:)',1,m).*Ws'];
        b=(train_targets(i,:).*dec_val(i,:))'-1;
        Aeq=[zeros(1,T),ones(1,m)];
        beq=1;
        lb=zeros(T+m,1);
        ub=[inf(T,1);ones(m,1)];
        [x,fval] = linprog(f,A,b,Aeq,beq,lb,ub);
        S(i,:)=x(T+1:end);
    end
    
    % update C
    for i=1:m
        C(i,:)=sum(repmat(S(:,i),1,T).*train_targets)./sum(S(:,i));
    end
    if(abs(last_obj-fval)<0.001)
        break;
    end
    last_obj=fval;
end

kernelS=S*S';
for i=1:T
    kernel_sum=[(1:n)',kernelX+kernelS+0.0001*eye(size(kernelX))];
    model=svmtrain(train_targets(:,i),kernel_sum,['-t 4 -c ',num2str(lambda1)]);
    models{i}=model;
    
end