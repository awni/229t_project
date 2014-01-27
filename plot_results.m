grad = load('grad.txt');
grad=bsxfun(@minus,grad, mean(grad));
[u,s,v]=svds(grad,2);
cost = load('cost.txt');
param = load('param.txt');
t = cost';
tmp=param*v;
tmp = [tmp t(:)];

hold on;
for i = 1:floor(size(tmp,1)/70),
    x=tmp((i-1)*70+1:i*70,:);
    mesh([x(:,1),x(:,1)],[x(:,2),x(:,2)],[x(:,3),x(:,3)]);
end