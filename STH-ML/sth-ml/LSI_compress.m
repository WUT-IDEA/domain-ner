function B = LSI_compress(X, model, kernel, maxbits)

Nsamples = size(X,1);

Y = X * model.V * model.invS;
Z = repmat(model.medU,Nsamples,1);
B = (Y>Z);

end
