function B = SpH_compress(X, model, kernel, maxbits)

Nsamples = size(X,1);

X = X*model.pc;
X = X-repmat(model.mn, [Nsamples 1]);
omega0=pi./(model.mx-model.mn);
omegas=model.modes.*repmat(omega0, [maxbits 1]);

U = zeros([Nsamples maxbits]);
for i=1:maxbits
    omegai = repmat(omegas(i,:), [Nsamples 1]);
    ys = sin(X.*omegai+pi/2);
    yi = prod(ys,2);
    U(:,i)=yi;    
end

B = (U>0);

end
