function [model, B] = SpH_learn(X, gnd, k, metric, kernel, maxbits)

[Nsamples Ndim] = size(X);

X0 = X;

% 1) PCA

npca = min(maxbits, Ndim);
% The following line has been replaced as it is very slow in comparison to 
% the PCA code by Deng Cai
% [pc, l] = eigs(cov(X), npca);
[pc, l] = PCA(X./sqrt(Nsamples-1),struct('ReducedDim',npca));
X = X * pc; % no need to remove the mean

% 2) fit uniform distribution

mn = prctile(X, 5);  mn = min(X)-eps;
mx = prctile(X,95);  mx = max(X)+eps;

% 3) enumerate eigenfunctions

R=(mx-mn);
maxMode = ceil((maxbits+1)*R/max(R));

nModes = sum(maxMode)-length(maxMode)+1;
modes = ones([nModes npca]);
m = 1;
for i=1:npca
    modes(m+1:m+maxMode(i)-1,i) = 2:maxMode(i);
    m = m+maxMode(i)-1;
end
modes = modes - 1;
omega0 = pi./R;
omegas = modes.*repmat(omega0, [nModes 1]);
eigVal = -sum(omegas.^2,2);
[yy,ii]= sort(-eigVal);
modes=modes(ii(2:maxbits+1),:);

% 4) store model paramaters

model.pc = pc;
model.mn = mn;
model.mx = mx;
model.mx = mx;
model.modes = modes;

% 5) compute code for X

B = SpH_compress(X0, model, kernel, maxbits);

end
