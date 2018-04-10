function [model, B] = LCH_learn(A, gnd, k, metric, kernel, maxbits)

[Nsamples,Nfeatures] = size(A);
k = maxbits;

d1 = sqrt(sum(A,2))';
d1(d1>0) = 1./d1(d1>0);
D1 = speye(Nsamples,Nsamples);
D1(D1>0) = d1;

d2 = sqrt(sum(A));
d2(d2>0) = 1./d2(d2>0);
D2 = speye(Nfeatures,Nfeatures);
D2(D2>0) = d2;

X = D1*A*D2;

[U,S,V] = svds(X,k);

model.V = V;
model.invS = pinv(S);
model.D2 = D2;
model.medU = median(D1*U);

Z = repmat(model.medU,Nsamples,1);
B = (D1*U > Z);

end
