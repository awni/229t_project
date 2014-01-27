grad = load('grad.txt');
grad=bsxfun(@minus,grad, mean(grad));
[u,s,v]=svds(cov(grad),2);
disp(diag(s));
cost = load('cost.txt');
param = load('param.txt');
t = cost';
tmp=param*v;
tmp = [tmp t(:)];

points_per_trial = 70;

hold on;
for i = 1:floor(size(tmp,1)/points_per_trial),
    x=tmp((i-1)*points_per_trial+1:i*points_per_trial,:);
    mesh([x(:,1),x(:,1)],[x(:,2),x(:,2)],[x(:,3),x(:,3)]);
end