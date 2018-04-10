function B = LCH_compress(A, model, kernel, maxbits)

Nsamples = size(A,1);

d1 = sqrt(sum(A,2))';
d1(d1>0) = 1./d1(d1>0);
D1 = speye(Nsamples,Nsamples);
D1(D1>0) = d1;

D2 = model.D2;

X = D1*A*D2;

Y = X * model.V * model.invS;

Z = repmat(model.medU,Nsamples,1);
B = (D1*Y > Z);

end
