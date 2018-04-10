function [model, B] = LSI_learn(X, gnd, k, metric, kernel, maxbits)

Nsamples = size(X,1);
k = maxbits;

[U,S,V] = svds(X,k);

model.V = V;
model.invS = pinv(S);
model.medU = median(U);

Z = repmat(model.medU,Nsamples,1);
B = (U>Z);

end
