grad = load('grad.txt');
grad=bsxfun(@minus,grad, mean(grad));
cost = load('cost.txt');
param = load('param.txt');
[u,s,v]=svds(cov(grad),13);
disp(diag(s));
t = cost';
tmp=param*v;
tmp = [tmp t(:)];

points_per_trial = 1120;

for f = 1:3,
    figure;
    title(sprintf('eigs: %d and %d', f, f+1));
    hold on;
    for i = 1:floor(size(tmp,1)/points_per_trial),
        x=tmp((i-1)*points_per_trial+1:i*points_per_trial,:);
        mesh([x(:,f),x(:,f)],[x(:,f+1),x(:,f+1)],[x(:,end),x(:,end)]);
    end
end